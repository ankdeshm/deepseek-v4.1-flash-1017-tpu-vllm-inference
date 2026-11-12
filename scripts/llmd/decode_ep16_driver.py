#!/usr/bin/env python3
"""Step 2 Pool B EP=16 (2x2x2, 16 TensorCores) Decode Driver & Supervisor.

Runs inside LeaderWorkerSet `llmd-decode-ep16` on `YOUR_TPU_V7X_POOL_B`:
  - Leader (`<gke-node>`, `<private-ip>`, worker 0): DP ranks 0..7 + HTTP :9400 + sidecar :8000
  - Worker (`<gke-node>`, `<private-ip>`, worker 1): DP ranks 8..15 (--headless)
"""

import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

GCS_BUCKET = "YOUR_GCS_BUCKET"
MODEL_PREFIX = "models/deepseek-ai/DeepSeek-V4.1-Flash-1017/"
STEP2_GCS_PREFIX = "dsv41-v7x-llmd-20261106/step2_llmd"
CACHE_GCS_OBJECT = "dsv41-v7x-llmd-20261106/compile_cache/compile_cache_decode_ep16_xla.tar.gz"
STACK_TAR = "/models/dsv41-stack/stack_27be32f509754e88.tar"
LAUNCH_SH_SRC = "/models/dsv41-v7x-llmd-20261106/step2_llmd/launch_dsv41_decode_ep16.sh"
LOG_PATH = "/dev/shm/pd_decode_ep16.log"
BARRIER_PORT = 29599


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def mds_get(path: str, default: str = "") -> str:
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{path}",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return default


def vm_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["access_token"]


def gcs_upload_bytes(obj_name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    for attempt in range(4):
        try:
            tok = vm_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj_name, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {tok}",
                    "Content-Type": content_type,
                },
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
            return
        except Exception as e:
            if attempt == 3:
                log(f"[WARN] GCS upload failed for {obj_name}: {e}")
            else:
                time.sleep(2 * (attempt + 1))


def gcs_upload_file(obj_name: str, local_path: str, content_type: str = "application/octet-stream") -> None:
    sz = os.path.getsize(local_path)
    for attempt in range(4):
        try:
            tok = vm_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj_name, safe='')}"
            )
            with open(local_path, "rb") as fp:
                req = urllib.request.Request(
                    url,
                    data=fp,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "Content-Type": content_type,
                        "Content-Length": str(sz),
                    },
                )
                with urllib.request.urlopen(req, timeout=300) as r:
                    r.read()
            return
        except Exception as e:
            if attempt == 3:
                log(f"[WARN] GCS file upload failed for {obj_name}: {e}")
            else:
                time.sleep(2 * (attempt + 1))


CKPT_DL_SCRIPT = r'''#!/usr/bin/env python3
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
'''


PIP_INSTALL_SH = r'''#!/usr/bin/env bash
set -euo pipefail
LIBTPU_VER="0.0.48"
export TMPDIR=/dev/shm/tmp
mkdir -p /dev/shm/tmp
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
'''


def tcp_barrier(node_role: str, leader_ip: str, port: int, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    if node_role == "leader":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(4)
        srv.settimeout(5.0)
        log(f"[BARRIER] Leader listening on 0.0.0.0:{port} waiting for worker...")
        try:
            while time.time() < deadline:
                try:
                    conn, addr = srv.accept()
                    msg = conn.recv(256).decode("utf-8", errors="replace")
                    log(f"[BARRIER] Leader accepted worker from {addr}: {msg}")
                    conn.sendall(b"GO\n")
                    conn.close()
                    return
                except socket.timeout:
                    continue
            raise TimeoutError("Timed out waiting for worker at TCP barrier")
        finally:
            srv.close()
    else:
        log(f"[BARRIER] Worker dialing leader {leader_ip}:{port}...")
        while time.time() < deadline:
            try:
                s = socket.create_connection((leader_ip, port), timeout=5.0)
                s.sendall(b"WORKER_READY\n")
                resp = s.recv(64)
                s.close()
                if b"GO" in resp:
                    log("[BARRIER] Worker received GO from leader")
                    return
            except Exception:
                time.sleep(1.5)
        raise TimeoutError("Timed out dialing leader TCP barrier")


def inspect_torch_tpu_dist() -> str:
    code = r'''
import inspect, os
out = []
try:
    import torch_tpu
    pkg_dir = os.path.dirname(torch_tpu.__file__)
    out.append(f"torch_tpu.__file__ = {torch_tpu.__file__}")
    for root, _, files in os.walk(pkg_dir):
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                txt = open(p, errors="replace").read()
                if "LOCAL_WORLD_SIZE" in txt or "SLICEBUILDER" in txt or "TPU_VISIBLE_CHIPS" in txt:
                    out.append(f"\n=== {p} ===\n" + txt[:12000])
except Exception as e:
    out.append(f"ERR: {e}")
print("\n".join(out))
'''
    r = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr


def parse_diagnostics(log_text: str) -> dict:
    mem_lines = [ln for ln in log_text.splitlines() if "Memory statistics |" in ln]
    ep_mesh_lines = [ln for ln in log_text.splitlines() if "Built EP mesh" in ln]
    moe_lines = [ln for ln in log_text.splitlines() if "Fused EP MoE armed" in ln]
    kv_lines = [ln for ln in log_text.splitlines() if "GPU KV cache size" in ln or "Available KV cache memory" in ln]
    raiden_lines = [ln for ln in log_text.splitlines() if "Raiden" in ln or "ReshardStore" in ln or "topology=dp" in ln]
    compile_lines = [ln for ln in log_text.splitlines() if "Graph compilation takes" in ln or "compile" in ln.lower() and "takes" in ln.lower()]
    hbm_post_lines = [ln for ln in log_text.splitlines() if "Post-warmup HBM" in ln or "HBM" in ln and "GiB" in ln]
    attn_lines = [ln for ln in log_text.splitlines() if "Using " in ln and "backend" in ln or "kv_cache_dtype" in ln or "fp8" in ln.lower() and ("kv" in ln.lower() or "attn" in ln.lower())]
    return {
        "memory_statistics_count": len(mem_lines),
        "memory_statistics_sample": mem_lines[:16],
        "ep_mesh_lines": ep_mesh_lines[:8],
        "fused_ep_moe_lines": moe_lines[:8],
        "kv_cache_lines": kv_lines[:32],
        "raiden_lines": raiden_lines[:32],
        "compile_lines": compile_lines[:16],
        "hbm_lines": hbm_post_lines[:16],
        "precision_attn_lines": attn_lines[:16],
    }


def archive_compile_cache() -> dict:
    t0 = time.time()
    tar_dst = "/dev/shm/compile_cache_decode_ep16_xla.tar.gz"
    cache_root = "/dev/shm/pd_cache_decode"
    paths_to_pack = []
    if os.path.isdir(f"{cache_root}/xla"):
        paths_to_pack.append("xla")
    tc_root = f"{cache_root}/vllm/torch_compile_cache"
    if os.path.isdir(tc_root):
        for hash_dir in sorted(os.listdir(tc_root)):
            r0 = os.path.join(tc_root, hash_dir, "rank_0_0")
            if os.path.isdir(r0):
                paths_to_pack.append(f"vllm/torch_compile_cache/{hash_dir}/rank_0_0")
                break
        if len(paths_to_pack) == 1:
            # Fallback: include first entry under torch_compile_cache if structured differently
            subs = sorted(os.listdir(tc_root))
            if subs:
                paths_to_pack.append(f"vllm/torch_compile_cache/{subs[0]}")
    if not paths_to_pack:
        return {"archived": False, "reason": "no cache dirs"}
    cmd = ["tar", "-C", cache_root, "-I", "gzip -1", "-cf", tar_dst] + paths_to_pack
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(tar_dst)
    t_pack = time.time() - t0
    t_up0 = time.time()
    gcs_upload_file(CACHE_GCS_OBJECT, tar_dst, content_type="application/gzip")
    t_up = time.time() - t_up0
    try:
        os.remove(tar_dst)
    except OSError:
        pass
    return {
        "archived": True,
        "gcs_uri": f"gs://{GCS_BUCKET}/{CACHE_GCS_OBJECT}",
        "packed_paths": paths_to_pack,
        "size_bytes": sz,
        "size_mb": round(sz / (1024 * 1024), 2),
        "pack_seconds": round(t_pack, 2),
        "upload_seconds": round(t_up, 2),
    }


def http_check(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")[:500]
    except Exception as e:
        return 0, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-role", required=True, choices=["leader", "worker"])
    ap.add_argument("--leader-ip", default="<private-ip>")
    ap.add_argument("--run-id", default="llmd_ep16")
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--barrier-port", type=int, default=BARRIER_PORT)
    args = ap.parse_args()

    t_pod_start = time.time()
    real_ip = mds_get("instance/network-interfaces/0/ip") or socket.gethostbyname(socket.gethostname())
    real_hostname = mds_get("instance/hostname") or socket.gethostname()
    worker_num = mds_get("instance/attributes/agent-worker-number", "unknown")
    log(
        f"[SETUP] Starting Step 2 EP=16 Decode driver: node_role={args.node_role} "
        f"real_ip={real_ip} hostname={real_hostname} gce_agent_worker_number={worker_num} skip_setup={args.skip_setup}"
    )

    fix_src = "/models/dsv41-v7x-llmd-20261106/step2_llmd/ep16_multihost_dp_fix.py"
    if args.skip_setup:
        t_setup_done = 0.0
        for pat in ["/dev/shm/psm_*", "/dev/shm/sem.*", "/tmp/libengram_gather_omp_*.so", "/tmp/tpu_logs/vllm-decode/*", "/tmp/smoke16_logs_*"]:
            for p in glob.glob(pat):
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.unlink(p) if os.path.exists(p) else None
        if not os.path.exists("/workspace/launch_dsv41_decode_ep16.sh") and os.path.exists(LAUNCH_SH_SRC):
            shutil.copyfile(LAUNCH_SH_SRC, "/workspace/launch_dsv41_decode_ep16.sh")
        os.chmod("/workspace/launch_dsv41_decode_ep16.sh", 0o755)
        if os.path.exists("/workspace/ep16_multihost_dp_fix.py"):
            subprocess.run(["python3", "/workspace/ep16_multihost_dp_fix.py"], check=True)
        elif os.path.exists(fix_src):
            shutil.copyfile(fix_src, "/workspace/ep16_multihost_dp_fix.py")
            subprocess.run(["python3", "/workspace/ep16_multihost_dp_fix.py"], check=True)
        shards = len(glob.glob("/dev/shm/DeepSeek-V4.1-Flash/model-*-of-00048.safetensors"))
        assert shards == 48, f"Expected 48 shards in /dev/shm/DeepSeek-V4.1-Flash, found {shards}"
        log(f"[SETUP] Warm relaunch verified: {shards}/48 shards present in /dev/shm")
    else:
        subprocess.run(["mount", "-o", "remount,size=850G", "/dev/shm"], check=False)
        for pat in ["/dev/shm/pd_*", "/dev/shm/psm_*", "/dev/shm/sem.*", "/tmp/libengram_gather_omp_*.so"]:
            for p in glob.glob(pat):
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.unlink(p) if os.path.exists(p) else None
        os.makedirs("/dev/shm/tmp", exist_ok=True)
        os.makedirs("/workspace", exist_ok=True)

        # Wait up to 30s for gcsfuse /models mount
        for _ in range(30):
            if os.path.exists(STACK_TAR) and os.path.exists(LAUNCH_SH_SRC):
                break
            time.sleep(1)
        assert os.path.exists(STACK_TAR), f"Missing {STACK_TAR}"
        assert os.path.exists(LAUNCH_SH_SRC), f"Missing {LAUNCH_SH_SRC}"

        log(f"[SETUP] Extracting {STACK_TAR} to / ...")
        subprocess.run(
            ["tar", "--overwrite", "--keep-directory-symlink", "-C", "/", "-xf", STACK_TAR],
            check=True,
        )
        shutil.copyfile(LAUNCH_SH_SRC, "/workspace/launch_dsv41_decode_ep16.sh")
        os.chmod("/workspace/launch_dsv41_decode_ep16.sh", 0o755)

        for sc_dst in [
            "/usr/local/lib/python3.12/sitecustomize.py",
            "/usr/local/lib/python3.12/site-packages/sitecustomize.py",
            "/usr/lib/python3.12/sitecustomize.py",
        ]:
            os.makedirs(os.path.dirname(sc_dst), exist_ok=True)
            shutil.copyfile("/workspace/sitecustomize_bench/sitecustomize.py", sc_dst)
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)

        with open("/workspace/dsv41_ensure_ckpt.py", "w") as f:
            f.write(CKPT_DL_SCRIPT)
        os.chmod("/workspace/dsv41_ensure_ckpt.py", 0o755)

        with open("/workspace/install_pip_stack.sh", "w") as f:
            f.write(PIP_INSTALL_SH)
        os.chmod("/workspace/install_pip_stack.sh", 0o755)

        log("[SETUP] Launching parallel pip stack install + 48-shard /dev/shm checkpoint staging...")
        t_setup0 = time.time()
        pip_proc = subprocess.Popen(["bash", "/workspace/install_pip_stack.sh"])
        ckpt_env = os.environ.copy()
        ckpt_env["DSV41_GCS_BUCKET"] = GCS_BUCKET
        ckpt_env["DSV41_GCS_PREFIX"] = MODEL_PREFIX
        ckpt_proc = subprocess.Popen(["python3", "-u", "/workspace/dsv41_ensure_ckpt.py"], env=ckpt_env)

        pip_rc = pip_proc.wait()
        assert pip_rc == 0, f"pip stack install failed rc={pip_rc}"
        if os.path.exists(fix_src):
            shutil.copyfile(fix_src, "/workspace/ep16_multihost_dp_fix.py")
            subprocess.run(["python3", "/workspace/ep16_multihost_dp_fix.py"], check=True)
        log(f"[SETUP] pip stack install + ep16_multihost_dp_fix finished in {time.time() - t_setup0:.1f}s")

        if args.node_role == "leader":
            inspect_txt = inspect_torch_tpu_dist()
            gcs_upload_bytes(f"{STEP2_GCS_PREFIX}/torch_tpu_dist_inspect.txt", inspect_txt.encode("utf-8"), "text/plain")
            log("[SETUP] Uploaded torch_tpu_dist_inspect.txt to GCS")

        ckpt_rc = ckpt_proc.wait()
        assert ckpt_rc == 0, f"checkpoint download failed rc={ckpt_rc}"
        t_setup_done = time.time() - t_setup0
        log(f"[SETUP] Both pip + 48-shard checkpoint ready in {t_setup_done:.1f}s")

        os.makedirs("/workspace/dsv41_meta_backup", exist_ok=True)
        for mf in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
            src_mf = f"/dev/shm/DeepSeek-V4.1-Flash/{mf}"
            if os.path.exists(src_mf):
                shutil.copyfile(src_mf, f"/workspace/dsv41_meta_backup/{mf}")

    # Synchronize Leader + Worker right before launching vllm serve
    tcp_barrier(args.node_role, args.leader_ip, args.barrier_port, timeout_s=600)

    # Launch vllm serve
    serve_env = os.environ.copy()
    serve_env.update(
        {
            "NODE_ROLE": args.node_role,
            "LEADER_IP": args.leader_ip,
            "ADVERTISE_HOST": real_ip,
            "RUN_ID": args.run_id,
            "PORT": "9400",
            "DP_SIZE": "16",
            "DP_SIZE_LOCAL": "8",
            "MAX_NUM_SEQS": "48",
            "MAX_NUM_BATCHED_TOKENS": "288",
            "COMPILE_SIZES": "[16, 40, 48, 64, 96, 192, 288]",
            "ENABLE_DSPARK": "1",
            "DSPARK_TOKENS": "5",
            "ASYNC_SCHEDULING": "1",
            "ENABLE_PREFIX_CACHING": "0",
            "GPU_MEMORY_UTILIZATION": "0.85",
        }
    )

    t_vllm_start = time.time()
    log_fp = open(LOG_PATH, "w", buffering=1)
    vllm_proc = subprocess.Popen(
        ["bash", "/workspace/launch_dsv41_decode_ep16.sh"],
        env=serve_env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    with open("/dev/shm/pd_active_role.pid", "w") as pf:
        pf.write(str(vllm_proc.pid))
    log(f"[SERVE] Started launch_dsv41_decode_ep16.sh pid={vllm_proc.pid}, logging to {LOG_PATH}")

    ckpt_deleted = False
    ready_recorded = False
    last_log_pos = 0
    last_gcs_sync = 0.0

    while True:
        time.sleep(5)
        # Stream new log lines to container stdout for `kubectl logs` visibility
        try:
            with open(LOG_PATH, "r", errors="replace") as rf:
                rf.seek(last_log_pos)
                chunk = rf.read()
                last_log_pos = rf.tell()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except Exception:
            pass

        # Read full log for state transitions
        try:
            with open(LOG_PATH, "r", errors="replace") as rf:
                full_log = rf.read()
        except Exception:
            full_log = ""

        # Checkpoint eviction once all 8 local workers print "Memory statistics |"
        if not ckpt_deleted:
            mem_cnt = full_log.count("Memory statistics |")
            if mem_cnt >= 8:
                removed = 0
                for sf in glob.glob("/dev/shm/DeepSeek-V4.1-Flash/*.safetensors") + glob.glob("/dev/shm/DeepSeek-V4.1-Flash/.done_*"):
                    try:
                        os.remove(sf)
                        removed += 1
                    except OSError:
                        pass
                ckpt_deleted = True
                msg = f"[EP16-SUPERVISOR] Deleted {removed} checkpoint files from /dev/shm after {mem_cnt}x 'Memory statistics |' at +{time.time() - t_vllm_start:.1f}s\n"
                log_fp.write(msg)
                log_fp.flush()
                log(msg.strip())

        # Periodically upload log snapshot to GCS (every 30s while booting, every 120s once ready)
        now = time.time()
        sync_interval = 120.0 if ready_recorded else 30.0
        if now - last_gcs_sync >= sync_interval and full_log:
            gcs_upload_bytes(
                f"{STEP2_GCS_PREFIX}/vllm_decode_ep16_{args.node_role}.log",
                full_log.encode("utf-8", errors="replace"),
                content_type="text/plain",
            )
            last_gcs_sync = now

        # Check HTTP readiness on Leader (or worker diagnostics after ckpt eviction + warmup)
        if args.node_role == "leader" and not ready_recorded:
            st_9400, body_9400 = http_check("http://127.0.0.1:9400/v1/models")
            st_health, _ = http_check("http://127.0.0.1:9400/health")
            st_8000, body_8000 = http_check("http://127.0.0.1:8000/v1/models")
            st_8000_h, _ = http_check("http://127.0.0.1:8000/health")
            if st_9400 == 200 and st_health == 200:
                cold_boot_s = time.time() - t_vllm_start
                log(
                    f"[READY] Leader vLLM :9400 READY in {cold_boot_s:.1f}s! "
                    f"sidecar :8000/v1/models={st_8000} :8000/health={st_8000_h}"
                )
                cache_info = archive_compile_cache()
                log(f"[CACHE] Archived EP=16 decode XLA compile cache: {json.dumps(cache_info)}")
                diag = parse_diagnostics(full_log)
                shm_stat = shutil.disk_usage("/dev/shm")
                report = {
                    "node_role": args.node_role,
                    "hostname": real_hostname,
                    "ip": real_ip,
                    "gce_agent_worker_number": worker_num,
                    "run_id": args.run_id,
                    "setup_seconds": round(t_setup_done, 2),
                    "cold_boot_vllm_seconds": round(cold_boot_s, 2),
                    "total_pod_to_ready_seconds": round(time.time() - t_pod_start, 2),
                    "http_9400_models_status": st_9400,
                    "http_9400_models_body": body_9400,
                    "http_8000_sidecar_models_status": st_8000,
                    "http_8000_sidecar_health_status": st_8000_h,
                    "shm_total_gib": round(shm_stat.total / (1024**3), 2),
                    "shm_used_gib": round(shm_stat.used / (1024**3), 2),
                    "shm_free_gib": round(shm_stat.free / (1024**3), 2),
                    "compile_cache_archive": cache_info,
                    "diagnostics": diag,
                }
                gcs_upload_bytes(
                    f"{STEP2_GCS_PREFIX}/decode_ep16_boot_metrics_leader.json",
                    json.dumps(report, indent=2).encode("utf-8"),
                    content_type="application/json",
                )
                gcs_upload_bytes(
                    f"{STEP2_GCS_PREFIX}/vllm_decode_ep16_leader.log",
                    full_log.encode("utf-8", errors="replace"),
                    content_type="text/plain",
                )
                ready_recorded = True

        elif args.node_role == "worker" and not ready_recorded and ckpt_deleted:
            # Check if Leader's :9400/v1/models is up so Worker also writes its final boot metrics JSON
            st_leader, _ = http_check(f"http://{args.leader_ip}:9400/v1/models")
            if st_leader == 200:
                cold_boot_s = time.time() - t_vllm_start
                diag = parse_diagnostics(full_log)
                shm_stat = shutil.disk_usage("/dev/shm")
                report = {
                    "node_role": args.node_role,
                    "hostname": real_hostname,
                    "ip": real_ip,
                    "gce_agent_worker_number": worker_num,
                    "run_id": args.run_id,
                    "setup_seconds": round(t_setup_done, 2),
                    "cold_boot_vllm_seconds": round(cold_boot_s, 2),
                    "shm_total_gib": round(shm_stat.total / (1024**3), 2),
                    "shm_used_gib": round(shm_stat.used / (1024**3), 2),
                    "shm_free_gib": round(shm_stat.free / (1024**3), 2),
                    "diagnostics": diag,
                }
                gcs_upload_bytes(
                    f"{STEP2_GCS_PREFIX}/decode_ep16_boot_metrics_worker.json",
                    json.dumps(report, indent=2).encode("utf-8"),
                    content_type="application/json",
                )
                gcs_upload_bytes(
                    f"{STEP2_GCS_PREFIX}/vllm_decode_ep16_worker.log",
                    full_log.encode("utf-8", errors="replace"),
                    content_type="text/plain",
                )
                log(f"[READY] Worker recorded EP=16 boot diagnostics in {cold_boot_s:.1f}s")
                ready_recorded = True

        # Detect unexpected vllm exit, upload full log immediately, and stay alive for inspection/relaunch
        rc = vllm_proc.poll()
        if rc is not None:
            log(f"[WARN] vllm_proc exited with rc={rc}! Uploading log and keeping pod alive for inspection...")
            if full_log:
                gcs_upload_bytes(
                    f"{STEP2_GCS_PREFIX}/vllm_decode_ep16_{args.node_role}.log",
                    full_log.encode("utf-8", errors="replace"),
                    content_type="text/plain",
                )
            while True:
                time.sleep(30)


if __name__ == "__main__":
    main()
