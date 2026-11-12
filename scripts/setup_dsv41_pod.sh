#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-worker}"
RERUN_ROOT="/models/dsv41-v7x-bench"
STACK_DIR="/models/dsv41-stack"
GCS_BUCKET="YOUR_GCS_BUCKET"
MODEL_PREFIX="models/deepseek-ai/DeepSeek-V4.1-Flash-1017/"

echo "[DSV41-SETUP] Starting setup for role=${ROLE} on $(hostname) at $(date -u +%FT%TZ)"

# Record real GCE internal IP BEFORE mock_metadata.py or /etc/hosts changes anything
REAL_IP=$(curl -sf -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/ip || hostname -I | awk '{print $1}')
REAL_HOSTNAME=$(curl -sf -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/hostname || hostname)
echo "${REAL_IP}" > /tmp/real_host_ip
echo "${REAL_HOSTNAME}" > /tmp/real_hostname
echo "[DSV41-SETUP] Real host IP=${REAL_IP}, hostname=${REAL_HOSTNAME}"

mount -o remount,size=850G /dev/shm || true

# Kill stale processes and clean stale caches from previous pods on hostPath /dev/shm
for p in $(pgrep -f 'in_pod_serve.sh|dsv41_serve_role.sh|vllm serve|VLLM::|EngineCore|toy_proxy_server|mock_metadata.py' || true); do
  [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null || true
done
rm -rf /dev/shm/dsv41_fp4_cache /dev/shm/vllm_cache_27arr /dev/shm/xla_cache_27arr \
       /dev/shm/profiles /dev/shm/vllm_dp8_scale.log* /dev/shm/dsv41_runner.log* \
       /dev/shm/pd_* /dev/shm/psm_* /dev/shm/sem.* /tmp/libengram_gather_omp_*.so || true
mkdir -p /dev/shm/tmp /workspace
export TMPDIR=/dev/shm/tmp

echo "[DSV41-SETUP] Extracting stack_27be32f509754e88.tar into /workspace ..."
tar --overwrite --keep-directory-symlink -C / -xf "${STACK_DIR}/stack_27be32f509754e88.tar"
cp -f "${STACK_DIR}/toy_proxy_server.py" /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py
cp -f "${STACK_DIR}/launch_dsv41_disagg_role.sh" /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh
cp -f "${RERUN_ROOT}/scripts/launch_pd_role.sh" /workspace/launch_pd_role.sh
cp -f "${STACK_DIR}/pd_gates.py" /workspace/pd_gates.py
chmod +x /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh /workspace/launch_pd_role.sh

cd /workspace
(gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl) \
  || (apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null && \
      gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl)

for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
git config --global --add safe.directory '*' || true

# Write reusable 32-process HTTP Range parallel checkpoint downloader (/workspace/dsv41_ensure_ckpt.py)
cat <<'PY' > /workspace/dsv41_ensure_ckpt.py
#!/usr/bin/env python3
import concurrent.futures
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BUCKET = os.environ.get("DSV41_GCS_BUCKET", "YOUR_GCS_BUCKET")
PREFIX = os.environ.get("DSV41_GCS_PREFIX", "models/deepseek-ai/DeepSeek-V4.1-Flash-1017/")
DST = "/dev/shm/DeepSeek-V4.1-Flash"
CHUNK = 2 << 30
BUF = 16 << 20

os.makedirs(DST, exist_ok=True)
marker = f"{DST}/.done_all"
if os.path.exists(marker) and len(glob.glob(f"{DST}/model-*-of-00048.safetensors")) == 48:
    print("[CKPT] 48/48 shards already present in /dev/shm/DeepSeek-V4.1-Flash", flush=True)
    sys.exit(0)
if os.path.exists(marker):
    os.remove(marker)

def _vm_token():
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["access_token"]

token = _vm_token()
items, pt = [], None
while True:
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=" + urllib.parse.quote(PREFIX)
    if pt:
        url += "&pageToken=" + urllib.parse.quote(pt)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    items += d.get("items", [])
    pt = d.get("nextPageToken")
    if not pt:
        break

blobs = [
    (it["name"][len(PREFIX):], int(it["size"]))
    for it in items
    if not it["name"].endswith((".pdf", ".png", ".jpeg", ".patch", ".md", "/", ".tail"))
]

def dl(a):
    rel, dst, s, e = a
    for r in range(6):
        try:
            tk = _vm_token() if r > 0 else token
            u = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(PREFIX + rel, safe='')}?alt=media"
            req = urllib.request.Request(u, headers={"Authorization": f"Bearer {tk}", "Range": f"bytes={s}-{e-1}"})
            fd = os.open(dst, os.O_WRONLY)
            try:
                pos = s
                with urllib.request.urlopen(req, timeout=180) as resp:
                    while pos < e:
                        b = resp.read(min(BUF, e - pos))
                        if not b:
                            break
                        os.pwrite(fd, b, pos)
                        pos += len(b)
            finally:
                os.close(fd)
            return e - s
        except Exception:
            if r == 5:
                raise
            time.sleep(2 * (r + 1))

t0 = time.time()
tasks = []
total = 0
for rel, sz in blobs:
    if not rel:
        continue
    dst = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == sz:
        continue
    fd = os.open(dst, os.O_CREAT | os.O_WRONLY, 0o644)
    os.ftruncate(fd, sz)
    os.close(fd)
    total += sz
    for s in range(0, max(1, sz), CHUNK):
        tasks.append((rel, dst, s, min(s + CHUNK, sz)))

print(f"[CKPT] Downloading {total/1e9:.1f} GB in {len(tasks)} chunks from gs://{BUCKET}/{PREFIX} ...", flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
    list(ex.map(dl, tasks))
assert len(glob.glob(f"{DST}/model-*-of-00048.safetensors")) == 48, "Missing safetensor shards!"
open(marker, "w").close()
print(f"[CKPT] CKPT_READY in {time.time()-t0:.1f}s", flush=True)
PY
chmod +x /workspace/dsv41_ensure_ckpt.py

# Write reusable role supervisor (/workspace/dsv41_serve_role.sh)
cat <<'SH' > /workspace/dsv41_serve_role.sh
#!/usr/bin/env bash
set -uo pipefail
ENV_FILE="${1:?Usage: dsv41_serve_role.sh <env_file> [log_file]}"
LOG_FILE="${2:-/dev/shm/pd_role.log}"

# Stop any existing vLLM / EngineCore processes on this host before starting fresh
for p in $(pgrep -f 'launch_pd_role.sh|launch_dsv41_disagg_role.sh|vllm serve|VLLM::|EngineCore' || true); do
  [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null || true
done
sleep 2

# Ensure all 48 checkpoint shards are staged in /dev/shm (LD_PRELOAD="" so real GCE metadata token works)
LD_PRELOAD="" python3 -u /workspace/dsv41_ensure_ckpt.py

source "${ENV_FILE}"
: > "${LOG_FILE}"
bash /workspace/launch_pd_role.sh >> "${LOG_FILE}" 2>&1 &
SRV=$!
echo "${SRV}" > /dev/shm/pd_active_role.pid
deleted=0
EXPECTED_RANKS="${DP_SIZE_LOCAL:-8}"
while kill -0 "${SRV}" 2>/dev/null; do
  sleep 8
  if [ "${deleted}" = 0 ] && [ "${PD_DELETE_CKPT:-1}" = 1 ]; then
    c=$(grep -c "Memory statistics |" "${LOG_FILE}" || true)
    if [ "${c:-0}" -ge "${EXPECTED_RANKS}" ]; then
      rm -f /dev/shm/DeepSeek-V4.1-Flash/*.safetensors /dev/shm/DeepSeek-V4.1-Flash/.done_*
      echo "[PD-SERVE] deleted checkpoint shards after ${EXPECTED_RANKS}x Memory statistics" >> "${LOG_FILE}"
      deleted=1
    fi
  fi
done
wait "${SRV}"
echo "[PD-SERVE] server exited rc=$?" >> "${LOG_FILE}"
SH
chmod +x /workspace/dsv41_serve_role.sh

# Install pip wheels in background while staging 48 safetensors to /dev/shm
(
  LIBTPU_VER="0.0.48"
  pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
  pip install --no-cache-dir --no-deps "libtpu==${LIBTPU_VER}" portpicker==1.6.0 compressed-tensors==0.15.0 sympy aiohttp >/dev/null
  pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
  if ls /workspace/whls_pd/*.whl >/dev/null 2>&1; then
    pip install --no-cache-dir --no-index --no-deps --force-reinstall /workspace/whls_pd/*.whl >/dev/null
    echo "[PIP] pd override wheels: $(ls /workspace/whls_pd)"
  fi
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty \
    pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
  for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf /workspace/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
  done
  for d in /usr/local/lib/python3.12/*-packages/vllm; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm/vllm)" ] && cp -rf /workspace/vllm/vllm/* "$d/" || true
  done
  find /workspace -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  python3 -c "import importlib.metadata as m; v=m.version('libtpu'); assert v=='${LIBTPU_VER}', v; print('[PIP] libtpu', v, 'vllm', m.version('vllm'), 'torch_tpu', m.version('torch_tpu'), 'tpu_sync_torch', m.version('tpu-sync-torch'))"
) &
PIP_PID=$!

LD_PRELOAD="" DSV41_GCS_BUCKET="${GCS_BUCKET}" DSV41_GCS_PREFIX="${MODEL_PREFIX}" python3 -u /workspace/dsv41_ensure_ckpt.py
wait "${PIP_PID}"

mkdir -p /workspace/dsv41_meta_backup /data
for mf in config.json generation_config.json tokenizer.json tokenizer_config.json; do
  [ -f "/dev/shm/DeepSeek-V4.1-Flash/${mf}" ] && cp -f "/dev/shm/DeepSeek-V4.1-Flash/${mf}" "/workspace/dsv41_meta_backup/${mf}" || true
done
cp -f "${RERUN_ROOT}/inputs/requests_mocked.jsonl.xz" /data/requests_mocked.jsonl.xz
cp -f "${RERUN_ROOT}/inputs/requests_mocked.jsonl" /data/requests_mocked.jsonl
cp -f "${RERUN_ROOT}/scripts/benchmark_agentic_isl.py" /workspace/benchmark_agentic_isl.py
cp -f "${RERUN_ROOT}/scripts/bench_glm_ds_standalone.py" /workspace/bench_glm_ds_standalone.py

echo "PD_SETUP_OK" > /tmp/pd_setup_status
echo "[DSV41-SETUP] Completed PD_SETUP_OK on $(hostname) (${REAL_IP}) at $(date -u +%FT%TZ)"
