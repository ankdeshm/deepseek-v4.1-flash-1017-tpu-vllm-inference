#!/usr/bin/env python3
"""Step 0 Pre-Flight Verification & Cross-Pool 1P1D Smoke Test for DeepSeek-V4.1-Flash-1017 on TPU v7x.

Executes across 4 standalone 4-chip pods (pinned to all 4 hosts across Pool A and Pool B):
  - poola-h0 (gke-tpu-51ea19d7-rqkr, 192.168.100.43): Prefill server (DP=8, TP=1, EP=8, CED=1, :8400) + Proxy (:8000) + Smoke driver
  - poola-h1 (gke-tpu-51ea19d7-xq6s, 192.168.100.38): Concurrent independent single-host 4-chip (DP=8) libtpu/torch_tpu verification on Pool A slice
  - poolb-h0 (gke-tpu-771db71a-059c, 192.168.100.198): Decode server (DP=8, TP=1, EP=8, max_seqs=48, mnbt=288, DSpark K=5, async, :9400)
  - poolb-h1 (gke-tpu-771db71a-cz6l, 192.168.100.199): Concurrent independent single-host 4-chip (DP=8) libtpu/torch_tpu verification on Pool B slice + git SHA / EP=16 code audit
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request


GCS_BUCKET = "ank-usc1-bucket-314837540096"
STACK_PREFIX = "dsv41-stack"
MODEL_PREFIX = "models/deepseek-ai/DeepSeek-V4.1-Flash-1017/"
SERVED_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

PROBE_PORTS = [8000, 8400, 9400, 14579, 9100, 9200, 9600, 27000, 27100, 28000, 28100]

SMOKE_PROMPTS = [
    "Explain how Multi-head Latent Attention (MLA) reduces KV cache memory bandwidth during autoregressive decoding, and compute the bytes per token in FP8.",
    "Write a concise Python implementation of consistent prefix hashing with bounded load for routing LLM requests across prefill replicas.",
    "Compare Expert Parallelism EP=8 versus EP=16 for a 384-expert top-6 MoE model: how do weight bytes read per step and all-to-all dispatch volume change?",
    "Summarize the three main phases of Kubernetes Gateway API Inference Extension endpoint selection when Prefill/Decode disaggregation is enabled.",
    "Derive the expected tokens generated per verification step in speculative decoding with draft length K=5 if per-position acceptance probabilities are [0.82, 0.71, 0.60, 0.49, 0.38].",
    "List 12 major rivers in North America, the mountain ranges or lakes where they originate, and the bodies of water into which they drain.",
    "Write an ANSI SQL query using window functions to compute rolling 7-day P50 and P95 request latency per cluster from a request_logs table.",
    "Explain why block-scaled microscaling formats (MXFP4 with UE8M0 32-element shared exponents) preserve dynamic range better than uniform INT4.",
    "Describe the TCP congestion control differences between CUBIC and BBRv2 on intra-datacenter 200 Gbps links carrying bursty KV cache transfers.",
    "Provide a step-by-step checklist for diagnosing XLA HLO compilation cache misses across multi-process Data Parallel workers on TPU v7x.",
]


def log(role_id: str, msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [{role_id}] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["access_token"]


def gcs_download_file(bucket: str, obj_name: str, dst: Path) -> float:
    t0 = time.time()
    tok = vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj_name, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=300) as resp, open(dst, "wb") as f:
        while True:
            chunk = resp.read(16 << 20)
            if not chunk:
                break
            f.write(chunk)
    return time.time() - t0


def check_http(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def get_primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def unpack_stack_and_clean_shm(role_id: str, need_ckpt: bool) -> None:
    log(role_id, "Cleaning stale /dev/shm artifacts and unpacking dsv41 stack...")
    # Clean any leftover GLM-5.3 or older caches in /dev/shm
    clean_sh = r"""
set -euo pipefail
mount -o remount,size=850G /dev/shm || true
for p in /dev/shm/*; do
  [ -e "$p" ] || continue
  base=$(basename "$p")
  if [ "$base" = "DeepSeek-V4.1-Flash" ]; then
    continue
  fi
  rm -rf "$p" || true
done
mkdir -p /dev/shm/tmp
"""
    subprocess.run(["bash", "-lc", clean_sh], check=True)

    gcs_download_file(GCS_BUCKET, f"{STACK_PREFIX}/stack_27be32f509754e88.tar", Path("/tmp/stack.tar"))
    for fn in [
        "in_pod_setup.sh",
        "in_pod_serve.sh",
        "pd_gates.py",
        "bench_empirical_isl_dsv41.py",
        "toy_proxy_server.py",
        "launch_dsv41_disagg_role.sh",
    ]:
        gcs_download_file(GCS_BUCKET, f"{STACK_PREFIX}/{fn}", Path(f"/tmp/{fn}"))

    stage_sh = r"""
set -euo pipefail
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar && rm -f /tmp/stack.tar
cd /workspace
(gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl) \
  || (apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null && \
      gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl)
cp -f /tmp/toy_proxy_server.py /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py
cp -f /tmp/launch_dsv41_disagg_role.sh /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh
chmod +x /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh /tmp/in_pod_setup.sh /tmp/in_pod_serve.sh
git config --global --add safe.directory '*' || true
"""
    subprocess.run(["bash", "-lc", stage_sh], check=True)

    if not need_ckpt:
        # Install wheels only (skip 510GB download on indep_verify helper pods to save network bandwidth)
        pip_only_sh = r"""
set -euo pipefail
LIBTPU_VER="0.0.48"
for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
export TMPDIR=/dev/shm/tmp
pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
pip install --no-cache-dir --no-deps "libtpu==${LIBTPU_VER}" portpicker==1.6.0 compressed-tensors==0.15.0 sympy >/dev/null
pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
if ls /workspace/whls_pd/*.whl >/dev/null 2>&1; then
  pip install --no-cache-dir --no-index --no-deps --force-reinstall /workspace/whls_pd/*.whl >/dev/null
fi
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
"""
        subprocess.run(["bash", "-lc", pip_only_sh], check=True)


def collect_provenance_and_ep16_audit() -> dict:
    def git_info(repo: str) -> dict:
        if not Path(repo).exists():
            return {"exists": False}
        if not (Path(repo) / ".git").exists():
            top_files = [p.name for p in sorted(Path(repo).iterdir())[:25]]
            return {"exists": True, "has_dot_git": False, "top_files": top_files}
        try:
            sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
            branch = subprocess.check_output(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
            log1 = subprocess.check_output(["git", "-C", repo, "log", "-n", "3", "--oneline"], text=True, stderr=subprocess.STDOUT).strip()
            dirty = subprocess.check_output(["git", "-C", repo, "status", "--short"], text=True, stderr=subprocess.STDOUT).strip()
            return {"exists": True, "has_dot_git": True, "sha": sha, "branch": branch, "recent_commits": log1.splitlines(), "dirty_files": dirty.splitlines()[:30]}
        except Exception as e:
            return {"exists": True, "git_error": repr(e)}

    # Also measure GCS FUSE read throughput on 1 shard (4-worker read test) for comparison
    fuse_shard = Path("/models/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/model-00001-of-00048.safetensors")
    fuse_probe = {}
    if fuse_shard.exists():
        t0 = time.time()
        sz = fuse_shard.stat().st_size
        read_bytes = 0
        with open(fuse_shard, "rb") as f:
            while read_bytes < min(sz, 1 << 30):
                b = f.read(16 << 20)
                if not b:
                    break
                read_bytes += len(b)
        dt_s = time.time() - t0
        fuse_probe = {
            "shard": str(fuse_shard),
            "shard_size_bytes": sz,
            "sample_read_bytes": read_bytes,
            "sample_read_s": round(dt_s, 3),
            "single_stream_mb_s": round((read_bytes / 1e6) / max(dt_s, 1e-6), 1),
        }

    # Audit EP=16 / multi-host support in /workspace/vllm-torchtpu
    ep_hits = []
    vtt_src = Path("/workspace/vllm-torchtpu")
    for pyf in sorted(vtt_src.rglob("*.py")):
        txt = pyf.read_text(errors="replace")
        for i, ln in enumerate(txt.splitlines(), 1):
            if any(k in ln for k in ("Built EP mesh", "ep_rank_at_mesh_index", "data_parallel_size_local", "SLICEBUILDER", "TPU_WORKER_HOSTNAMES", "Fused EP MoE armed")):
                ep_hits.append(f"{pyf}:{i}: {ln.strip()[:200]}")
    disagg_scripts = [str(p) for p in sorted((vtt_src / "examples").rglob("*")) if p.is_file()] if (vtt_src / "examples").exists() else []

    pkg_vers = subprocess.check_output(
        [
            "python3",
            "-c",
            "import importlib.metadata as m, json; "
            "pkgs=['vllm','torch','torch_tpu','libtpu','tpu-sync-torch','compressed-tensors']; "
            "print(json.dumps({p: m.version(p) for p in pkgs}))",
        ],
        text=True,
    ).strip()

    ref_dst = Path("/models/dsv41-v7x-llmd-20261106/step0/ref_files")
    ref_dst.mkdir(parents=True, exist_ok=True)
    for rel_ref in [
        "MERGE_NOTES.md",
        "examples/multihost/README.md",
        "examples/disagg/launch_dsv4_disagg_multihost.sh",
        "src/vllm_torchtpu/distributed/tpu_mp_multihost.py",
        "src/vllm_torchtpu/distributed/ep_mesh.py",
        "src/vllm_torchtpu/worker/tpu_rank_binding.py",
    ]:
        sp = vtt_src / rel_ref
        if sp.exists():
            shutil.copy2(sp, ref_dst / sp.name)

    return {
        "vllm_git": git_info("/workspace/vllm"),
        "vllm_torchtpu_git": git_info("/workspace/vllm-torchtpu"),
        "python_packages": json.loads(pkg_vers),
        "gcsfuse_single_stream_probe": fuse_probe,
        "ep16_code_audit_hits": ep_hits[:80],
        "examples_files": disagg_scripts[:60],
    }


def run_tcp_echo_servers(ports: list[int], stop_evt: threading.Event) -> list[socket.socket]:
    socks = []
    for p in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", p))
        s.listen(8)
        s.settimeout(0.5)
        socks.append(s)

        def _serve(lsock: socket.socket, port: int) -> None:
            while not stop_evt.is_set():
                try:
                    conn, _ = lsock.accept()
                    with conn:
                        data = conn.recv(256)
                        conn.sendall(b"ACK:" + str(port).encode() + b":" + data)
                except socket.timeout:
                    continue
                except Exception:
                    break
            try:
                lsock.close()
            except Exception:
                pass

        threading.Thread(target=_serve, args=(s, p), daemon=True).start()
    return socks


def probe_remote_ports(target_ip: str, ports: list[int]) -> dict:
    results = {}
    for p in ports:
        t0 = time.perf_counter()
        try:
            with socket.create_connection((target_ip, p), timeout=5.0) as c:
                c.sendall(b"PING")
                resp = c.recv(256).decode(errors="replace")
            rtt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            results[str(p)] = {"ok": resp.startswith(f"ACK:{p}:"), "rtt_ms": rtt_ms, "resp": resp}
        except Exception as e:
            results[str(p)] = {"ok": False, "error": repr(e)}
    return results


def run_independent_singlehost_tpu_verify(role_id: str) -> dict:
    """Proves this host initializes its own independent 4-chip (8 TensorCore) 2x2x1 runtime while its slice neighbor is active."""
    indep_py = r"""
import os, sys, time, json, traceback
try:
    import torch
    import torch.distributed as dist
    import torch_tpu

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    t0 = time.time()
    n_dev = torch.tpu.device_count()
    dev_idx = 0 if n_dev == 1 else rank
    torch.tpu.set_device(dev_idx)
    dev = torch.device(f"tpu:{dev_idx}")
    dist.init_process_group(backend="tpu", init_method="env://")
    a = torch.ones((2048, 2048), dtype=torch.bfloat16, device=dev) * (rank + 1)
    b = torch.eye(2048, dtype=torch.bfloat16, device=dev)
    c = torch.matmul(a, b)
    dist.all_reduce(c)
    torch.tpu.synchronize()
    expected = float(sum(range(1, world + 1)))
    val = float(c[0, 0].cpu().item())
    assert val == expected, f"rank {rank} expected {expected}, got {val}"
    if rank == 0:
        print("INDEP_TPU_PASS " + json.dumps({
            "world_size": world,
            "device_count_per_proc": n_dev,
            "all_reduce_sum": val,
            "expected_sum": expected,
            "elapsed_s": round(time.time() - t0, 3),
        }), flush=True)
    dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    Path("/tmp/indep_test.py").write_text(indep_py)

    verify_sh = r"""
set -euo pipefail
grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >>/etc/hosts
pgrep -f mock_metadata.py >/dev/null || (LD_PRELOAD="" python3 /workspace/mock_metadata.py >/workspace/mock_metadata.log 2>&1 &)
sleep 1

unset TPU_PROCESS_PORT TPU_PROCESS_ADDRESSES TPU_WORKER_ID TPU_VISIBLE_CHIPS \
  TPU_RUNTIME_METRICS_PORTS TPU_SKIP_MDS_QUERY VBAR_CONTROL_SERVICE_URL \
  TPU_WORKER_HOSTNAMES TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT

SB_PORT_BASE=29600
sb=""
for i in 0 1 2 3 4 5 6 7; do sb+="${sb:+,}localhost:$((SB_PORT_BASE + i))"; done
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29599
export TORCH_TPU_SLICEBUILDER_ADDRESSES="${sb}"
export TORCH_TPU_TOPOLOGY="2,2,1,2"
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1 TOPOLOGY=2x2x1 TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_VISIBLE_DEVICES=0,1,2,3
[ -S /var/run/tpu-plugin/vbar.sock ] && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"
export TPU_RUNTIME_METRICS_PORTS="8431,8432,8433,8434,8435,8436,8437,8438"
export GCE_METADATA_HOST=127.0.0.1:80 GCE_METADATA_IP=127.0.0.1
export GOOGLE_CLOUD_PROJECT=tpu-prod-env-one-vm GCLOUD_PROJECT=tpu-prod-env-one-vm
export LD_PRELOAD=/workspace/metadata_hook.so
export PJRT_DEVICE=TPU

torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29599 /tmp/indep_test.py
"""
    t0 = time.time()
    proc = subprocess.run(["bash", "-lc", verify_sh], capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    m = re.search(r"INDEP_TPU_PASS (\{.*\})", out)
    return {
        "role_id": role_id,
        "hostname": socket.gethostname(),
        "ip": get_primary_ip(),
        "passed": proc.returncode == 0 and m is not None,
        "returncode": proc.returncode,
        "total_wall_s": round(time.time() - t0, 2),
        "details": json.loads(m.group(1)) if m else None,
        "log_tail": out[-3500:],
    }


def write_dl_and_setup(role: str, host_ip: str, run_id: str) -> dict:
    # Ensure fresh download measurement by removing any old safetensors first
    shutil.rmtree("/dev/shm/DeepSeek-V4.1-Flash", ignore_errors=True)

    dl_py = r"""
import os, sys, json, time, glob, urllib.request, urllib.parse, concurrent.futures
BUCKET, PREFIX = os.environ['DSV41_GCS_BUCKET'], os.environ['DSV41_GCS_PREFIX']
DST = '/dev/shm/DeepSeek-V4.1-Flash'
CHUNK, BUF = 2 << 30, 16 << 20
os.makedirs(DST, exist_ok=True)
def _vm_token():
    req = urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
                                 headers={'Metadata-Flavor': 'Google'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())['access_token']
token = _vm_token()
items, pt = [], None
while True:
    url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=' + urllib.parse.quote(PREFIX)
    if pt: url += '&pageToken=' + urllib.parse.quote(pt)
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'}), timeout=30).read())
    items += d.get('items', []); pt = d.get('nextPageToken')
    if not pt: break
blobs = [(it['name'][len(PREFIX):], int(it['size'])) for it in items
         if not it['name'].endswith(('.pdf', '.png', '.jpeg', '.patch', '.md', '/', '.tail'))]
def dl(a):
    rel, dst, s, e = a
    for r in range(6):
        try:
            tk = _vm_token() if r > 0 else token
            u = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(PREFIX + rel, safe="")}?alt=media'
            req = urllib.request.Request(u, headers={'Authorization': f'Bearer {tk}', 'Range': f'bytes={s}-{e-1}'})
            fd = os.open(dst, os.O_WRONLY)
            try:
                pos = s
                with urllib.request.urlopen(req, timeout=180) as resp:
                    while pos < e:
                        b = resp.read(min(BUF, e - pos))
                        if not b: break
                        os.pwrite(fd, b, pos); pos += len(b)
            finally:
                os.close(fd)
            return e - s
        except Exception:
            if r == 5: raise
            time.sleep(2 * (r + 1))
t0 = time.time(); tasks = []; total = 0
for rel, sz in blobs:
    if not rel: continue
    dst = os.path.join(DST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd = os.open(dst, os.O_CREAT | os.O_WRONLY, 0o644); os.ftruncate(fd, sz); os.close(fd)
    total += sz
    for s in range(0, max(1, sz), CHUNK): tasks.append((rel, dst, s, min(s + CHUNK, sz)))
print(f'Downloading {total/1e9:.2f} GB in {len(tasks)} chunks from gs://{BUCKET}/{PREFIX} ...', flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex: list(ex.map(dl, tasks))
open(f'{DST}/.done_all', 'w').close()
elapsed = time.time() - t0
print(f'CKPT_READY bytes={total} elapsed_s={elapsed:.2f} gb_per_s={(total/1e9)/max(elapsed,1e-6):.2f}', flush=True)
"""
    Path("/tmp/dsv41_dl.py").write_text(dl_py)
    tok = vm_access_token()
    Path("/tmp/dsv41_env.sh").write_text(
        f"export GCS_TOKEN='{tok}' DSV41_GCS_BUCKET='{GCS_BUCKET}' "
        f"DSV41_GCS_PREFIX='{MODEL_PREFIX}' DSV41_SHARDS='all'\n"
    )
    if role == "prefill":
        pd_env = {
            "ADVERTISE_HOST": host_ip,
            "RUN_ID": run_id,
            "CED_PREFILL": "1",
            "MAX_NUM_SEQS": "64",
            "MAX_NUM_BATCHED_TOKENS": "8192",
            "COMPILE_SIZES": "[512, 2048, 8192]",
            "ENABLE_DSPARK": "0",
            "ASYNC_SCHEDULING": "0",
            "GPU_MEMORY_UTILIZATION": "0.85",
        }
    else:
        # Known-good decode settings per user prompt:
        # --max-num-seqs 48 --max-num-batched-tokens 288 (compile bucket includes 288), DSpark K=5, --async-scheduling
        pd_env = {
            "ADVERTISE_HOST": host_ip,
            "RUN_ID": run_id,
            "CED_PREFILL": "0",
            "MAX_NUM_SEQS": "48",
            "MAX_NUM_BATCHED_TOKENS": "288",
            "COMPILE_SIZES": "[16, 40, 48, 64, 96, 192, 288]",
            "ENABLE_DSPARK": "1",
            "DSPARK_TOKENS": "5",
            "ASYNC_SCHEDULING": "1",
            "GPU_MEMORY_UTILIZATION": "0.85",
        }
    Path("/tmp/pd_env.sh").write_text("".join(f"export {k}='{v}'\n" for k, v in pd_env.items()))

    t0 = time.time()
    subprocess.run(["bash", "-lc", "bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup.log 2>&1"], check=True)
    setup_wall_s = round(time.time() - t0, 2)
    setup_txt = Path("/dev/shm/pd_setup.log").read_text(errors="replace")
    if "PD_SETUP_OK" not in setup_txt:
        raise RuntimeError(f"in_pod_setup.sh failed:\n{setup_txt[-1500:]}")

    m_dl = re.search(r"CKPT_READY bytes=(\d+) elapsed_s=([0-9.]+) gb_per_s=([0-9.]+)", setup_txt)
    ckpt_info = {
        "setup_wall_s": setup_wall_s,
        "ckpt_bytes": int(m_dl.group(1)) if m_dl else None,
        "ckpt_gcs_to_shm_s": float(m_dl.group(2)) if m_dl else None,
        "ckpt_gcs_gb_per_s": float(m_dl.group(3)) if m_dl else None,
        "pd_env": pd_env,
    }
    if Path("/dev/shm/DeepSeek-V4.1-Flash/config.json").exists():
        ckpt_info["model_config_quantization"] = json.loads(
            Path("/dev/shm/DeepSeek-V4.1-Flash/config.json").read_text()
        ).get("quantization_config")
        ckpt_info["model_config_dtype"] = json.loads(
            Path("/dev/shm/DeepSeek-V4.1-Flash/config.json").read_text()
        ).get("dtype")
    if Path("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json").exists():
        shutil.copy2("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json", "/tmp/dsv41_tokenizer.json")
    return ckpt_info


def extract_precision_and_load_stats(log_path: Path) -> dict:
    txt = log_path.read_text(errors="replace")
    load_times = [float(m.group(1)) for m in re.finditer(r"Loading weights took ([0-9.]+) seconds", txt)]
    precision_lines = []
    patterns = (
        "Detected quantization_config",
        "Using fp8 data type to store kv cache",
        "Setting TPU quantization config",
        "expert_dtype resolved",
        "FP8 linear weights requantized",
        "Fused EP MoE armed",
        "MXFP4 weights processed",
        "MXFP4 fused EP W4A8 enabled",
        "Memory statistics |",
        "TPU KV cache size",
        "Raiden DSv4 pool",
    )
    seen = set()
    for ln in txt.splitlines():
        for pat in patterns:
            if pat in ln:
                # Deduplicate per pattern prefix to keep summary concise
                key = pat + ":" + ln[-90:]
                if key not in seen and len(precision_lines) < 35:
                    seen.add(key)
                    precision_lines.append(ln.strip())
                break
    launch_cmd = ""
    for ln in txt.splitlines():
        if ln.startswith("[PD-LAUNCH] cmd:"):
            launch_cmd = ln
            break
    return {
        "launch_cmd": launch_cmd,
        "vllm_weight_load_seconds_per_worker": load_times,
        "vllm_weight_load_seconds_min": min(load_times) if load_times else None,
        "vllm_weight_load_seconds_max": max(load_times) if load_times else None,
        "precision_and_memory_log_lines": precision_lines,
    }


def scrape_decode_metrics(decode_url: str) -> dict:
    with urllib.request.urlopen(f"{decode_url}/metrics", timeout=10) as r:
        text = r.read().decode("utf-8", errors="replace")

    def _sum_metric(pattern: str) -> float:
        return sum(float(m.group(1)) for m in re.finditer(pattern, text, re.M))

    drafts = _sum_metric(r"^vllm:spec_decode_num_drafts_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
    draft_tokens = _sum_metric(r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
    accepted = _sum_metric(r"^vllm:spec_decode_num_accepted_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
    preemptions = _sum_metric(r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
    kv_vals = [
        float(m.group(1))
        for m in re.finditer(r"^vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
    ]
    per_pos: dict[str, float] = {}
    for m in re.finditer(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)',
        text,
        re.M,
    ):
        pos = m.group(1)
        per_pos[pos] = per_pos.get(pos, 0.0) + float(m.group(2))
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted": accepted,
        "per_pos": per_pos,
        "preemptions": preemptions,
        "kv_cache_usage_mean": (sum(kv_vals) / len(kv_vals)) if kv_vals else 0.0,
    }


def run_smoke_requests(proxy_url: str, decode_url: str) -> dict:
    # First run 2 warmup requests so any one-time proxy/Raiden handshake is warmed
    def _one_chat(idx_prompt: tuple[int, str], max_tokens: int = 192) -> dict:
        idx, prompt = idx_prompt
        payload = json.dumps(
            {
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        req = urllib.request.Request(
            f"{proxy_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        ttft = None
        chunks_txt = []
        usage = {}
        with urllib.request.urlopen(req, timeout=180) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                obj = json.loads(data_str)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks_txt.append(piece)
        wall = time.perf_counter() - t0
        full_text = "".join(chunks_txt).strip()
        comp_toks = usage.get("completion_tokens") or max(1, len(chunks_txt))
        prompt_toks = usage.get("prompt_tokens")
        decode_dur = max(1e-6, wall - (ttft or 0.0))
        return {
            "idx": idx,
            "status": status,
            "prompt": prompt[:100],
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "ttft_ms": round((ttft or wall) * 1000.0, 2),
            "wall_s": round(wall, 3),
            "per_user_tok_s": round(comp_toks / decode_dur, 2),
            "output_preview": full_text[:280],
            "sane": status == 200 and len(full_text) >= 40,
        }

    # Warmup 2 requests
    for i in range(2):
        _one_chat((i, SMOKE_PROMPTS[i]), max_tokens=64)

    # Measured 10-request cross-pool 1P1D smoke test (concurrency=5 across 10 requests)
    m0 = scrape_decode_metrics(decode_url)
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda ip: _one_chat(ip, 192), enumerate(SMOKE_PROMPTS)))
    total_wall_s = time.perf_counter() - t_start
    m1 = scrape_decode_metrics(decode_url)

    d_drafts = max(0.0, m1["num_drafts"] - m0["num_drafts"])
    d_accepted = max(0.0, m1["num_accepted"] - m0["num_accepted"])
    tau = (1.0 + d_accepted / d_drafts) if d_drafts > 0 else 1.0
    per_pos_rates = {
        k: round((m1["per_pos"].get(k, 0.0) - m0["per_pos"].get(k, 0.0)) / d_drafts, 4)
        for k in sorted(m1["per_pos"], key=lambda x: int(x))
    } if d_drafts > 0 else {}

    total_out_toks = sum(r["completion_tokens"] or 0 for r in results)
    return {
        "num_requests": len(results),
        "all_http_200": all(r["status"] == 200 for r in results),
        "all_sane_output": all(r["sane"] for r in results),
        "failures": sum(0 if r["sane"] else 1 for r in results),
        "wall_s": round(total_wall_s, 3),
        "total_completion_tokens": total_out_toks,
        "aggregate_out_tok_s": round(total_out_toks / max(total_wall_s, 1e-6), 2),
        "dspark_num_drafts": d_drafts,
        "dspark_num_accepted": d_accepted,
        "dspark_tau": round(tau, 4),
        "dspark_per_pos_accept_rates": per_pos_rates,
        "preemptions_delta": m1["preemptions"] - m0["preemptions"],
        "requests": results,
    }


def archive_compile_cache(role: str, cache_dst_dir: Path) -> dict:
    src = Path(f"/dev/shm/pd_cache_{role}")
    if not src.exists():
        return {"archived": False}
    cache_dst_dir.mkdir(parents=True, exist_ok=True)
    tar_name = f"compile_cache_{role}_dp8.tar.gz"
    local_tar = Path(f"/dev/shm/{tar_name}")
    t0 = time.time()
    subprocess.run(
        ["tar", "-C", str(src), "-czf", str(local_tar), "vllm", "xla"],
        check=False,
    )
    if local_tar.exists():
        sz = local_tar.stat().st_size
        shutil.copy2(local_tar, cache_dst_dir / tar_name)
        local_tar.unlink(missing_ok=True)
        return {"archived": True, "path": str(cache_dst_dir / tar_name), "size_mb": round(sz / 1e6, 2), "archive_s": round(time.time() - t0, 2)}
    return {"archived": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-role", required=True, choices=["poola-h0", "poola-h1", "poolb-h0", "poolb-h1"])
    ap.add_argument("--out-dir", default="/models/dsv41-v7x-llmd-20261106/step0")
    ap.add_argument("--run-id", default="step0_preflight_20261106")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    coord_dir = out_dir / "coord"
    coord_dir.mkdir(parents=True, exist_ok=True)

    my_ip = get_primary_ip()
    my_host = socket.gethostname()
    (coord_dir / f"host_{args.pod_role}.json").write_text(
        json.dumps({"pod_role": args.pod_role, "hostname": my_host, "ip": my_ip, "ts": time.time()}, indent=2)
    )
    log(args.pod_role, f"Started on host={my_host} ip={my_ip}")

    need_ckpt = args.pod_role in ("poola-h0", "poolb-h0")
    unpack_stack_and_clean_shm(args.pod_role, need_ckpt=need_ckpt)

    # --- Helper Pods (poola-h1 and poolb-h1): Provenance audit + Concurrent Single-Host TPU Verify ---
    if args.pod_role in ("poola-h1", "poolb-h1"):
        if args.pod_role == "poolb-h1":
            log(args.pod_role, "Collecting git SHAs, package versions, GCS FUSE probe, and EP=16 code audit...")
            prov = collect_provenance_and_ep16_audit()
            (out_dir / "provenance_and_ep16_audit.json").write_text(json.dumps(prov, indent=2))
            log(args.pod_role, f"Saved provenance_and_ep16_audit.json (vllm={prov['vllm_git'].get('sha')}, vtt={prov['vllm_torchtpu_git'].get('sha')})")

        neighbor = "poola-h0" if args.pod_role == "poola-h1" else "poolb-h0"
        log(args.pod_role, f"Waiting for slice neighbor {neighbor} to initialize its 8-TensorCore TPU mesh...")
        t_wait = time.time()
        while not (coord_dir / f"tpu_mesh_up_{neighbor}.json").exists():
            if time.time() - t_wait > 2400:
                raise TimeoutError(f"Timed out waiting for {neighbor} TPU mesh init")
            time.sleep(5)
        log(args.pod_role, f"Neighbor {neighbor} TPU mesh is LIVE! Running concurrent single-host 8-TensorCore torch_tpu test on {my_host}...")
        indep_res = run_independent_singlehost_tpu_verify(args.pod_role)
        (out_dir / f"indep_singlehost_{args.pod_role}.json").write_text(json.dumps(indep_res, indent=2))
        log(args.pod_role, f"Concurrent single-host TPU verification passed={indep_res['passed']} in {indep_res['total_wall_s']}s!")
        return 0 if indep_res["passed"] else 1

    # --- Main Serving Pods: poola-h0 (Prefill) and poolb-h0 (Decode) ---
    while not ((coord_dir / "host_poola-h0.json").exists() and (coord_dir / "host_poolb-h0.json").exists()):
        time.sleep(2)
    peer_a_ip = json.loads((coord_dir / "host_poola-h0.json").read_text())["ip"]
    peer_b_ip = json.loads((coord_dir / "host_poolb-h0.json").read_text())["ip"]

    # Step 0.4a: Cross-pool TCP port reachability test before vLLM binds ports
    stop_echo = threading.Event()
    run_tcp_echo_servers(PROBE_PORTS, stop_echo)
    (coord_dir / f"echo_ready_{args.pod_role}.json").write_text("1")
    while not ((coord_dir / "echo_ready_poola-h0.json").exists() and (coord_dir / "echo_ready_poolb-h0.json").exists()):
        time.sleep(1)
    time.sleep(1)
    target_ip = peer_b_ip if args.pod_role == "poola-h0" else peer_a_ip
    tcp_res = probe_remote_ports(target_ip, PROBE_PORTS)
    (out_dir / f"tcp_probe_from_{args.pod_role}.json").write_text(
        json.dumps({"from_ip": my_ip, "to_ip": target_ip, "ports": tcp_res}, indent=2)
    )
    (coord_dir / f"echo_done_{args.pod_role}.json").write_text("1")
    while not ((coord_dir / "echo_done_poola-h0.json").exists() and (coord_dir / "echo_done_poolb-h0.json").exists()):
        time.sleep(1)
    stop_echo.set()
    time.sleep(2)
    log(args.pod_role, f"Cross-pool TCP probe -> {target_ip} all_ok={all(v['ok'] for v in tcp_res.values())}")

    # Step 0.5: Stage checkpoint on both pools and record load time
    vllm_role = "prefill" if args.pod_role == "poola-h0" else "decode"
    srv_port = 8400 if vllm_role == "prefill" else 9400
    log(args.pod_role, f"Staging 48-shard checkpoint to /dev/shm and installing wheels ({vllm_role})...")
    ckpt_info = write_dl_and_setup(vllm_role, my_ip, args.run_id)
    (out_dir / f"ckpt_stage_{args.pod_role}.json").write_text(json.dumps(ckpt_info, indent=2))
    log(
        args.pod_role,
        f"Checkpoint staged: {ckpt_info.get('ckpt_gcs_to_shm_s')}s ({ckpt_info.get('ckpt_gcs_gb_per_s')} GB/s)",
    )

    # Launch vLLM role server
    log(args.pod_role, f"Launching vLLM {vllm_role} server on :{srv_port}...")
    t_boot = time.time()
    srv_proc = subprocess.Popen(["bash", "/tmp/in_pod_serve.sh", vllm_role])
    srv_log = Path(f"/dev/shm/pd_{vllm_role}.log")
    seen_lines = 0
    mesh_signaled = False
    progress_keys = (
        "[PD-LAUNCH]",
        "Loading weights took",
        "Built EP mesh",
        "Fused EP MoE armed",
        "MXFP4 fused EP W4A8",
        "Memory statistics",
        "TPU KV cache size",
        "Raiden DSv4 pool",
        "Application startup complete",
        "deleted checkpoint",
    )
    fail_keys = ("Engine core initialization failed", "RuntimeBufferAllocationFailure", "EngineCore failed", "[PD-SERVE] server exited")

    while True:
        if srv_proc.poll() is not None:
            tail = srv_log.read_text(errors="replace")[-4000:] if srv_log.exists() else ""
            raise RuntimeError(f"{vllm_role} exited early (rc={srv_proc.returncode}):\n{tail}")
        if srv_log.exists():
            lines = srv_log.read_text(errors="replace").splitlines()
            for ln in lines[seen_lines:]:
                if any(k in ln for k in progress_keys):
                    log(args.pod_role, f"[{vllm_role}] {ln[:320]}")
                if (not mesh_signaled) and ("Built EP mesh" in ln or "Memory statistics |" in ln):
                    (coord_dir / f"tpu_mesh_up_{args.pod_role}.json").write_text(
                        json.dumps({"ts": time.time(), "host": my_host, "ip": my_ip})
                    )
                    mesh_signaled = True
                if any(fk in ln for fk in fail_keys):
                    tail = "\n".join(lines[-80:])
                    (out_dir / f"{args.pod_role}_boot_fail.log").write_text(tail)
                    raise RuntimeError(f"{vllm_role} boot failure:\n{ln}\n{tail}")
            seen_lines = len(lines)
        if check_http(f"http://127.0.0.1:{srv_port}/health"):
            boot_s = round(time.time() - t_boot, 2)
            if not mesh_signaled:
                (coord_dir / f"tpu_mesh_up_{args.pod_role}.json").write_text(json.dumps({"ts": time.time()}))
            prec_stats = extract_precision_and_load_stats(srv_log)
            prec_stats["boot_healthy_s"] = boot_s
            (out_dir / f"vllm_boot_{args.pod_role}.json").write_text(json.dumps(prec_stats, indent=2))
            (coord_dir / f"healthy_{args.pod_role}.json").write_text(json.dumps({"boot_s": boot_s, "ip": my_ip}))
            log(args.pod_role, f"{vllm_role.upper()} HEALTHY on :{srv_port} in {boot_s}s!")
            break
        if time.time() - t_boot > 4200:
            raise TimeoutError(f"{vllm_role} boot exceeded 4200s")
        time.sleep(12)

    # Wait for both Prefill (Pool A) and Decode (Pool B) to be healthy
    while not ((coord_dir / "healthy_poola-h0.json").exists() and (coord_dir / "healthy_poolb-h0.json").exists()):
        if srv_proc.poll() is not None:
            raise RuntimeError(f"{vllm_role} died waiting for peer")
        time.sleep(5)

    if args.pod_role == "poolb-h0":
        log(args.pod_role, "Decode server active; waiting for poola-h0 to run cross-pool 1P1D smoke test...")
        while not (coord_dir / "smoke_done.json").exists():
            if srv_proc.poll() is not None:
                raise RuntimeError("Decode server exited during smoke test")
            time.sleep(5)
        log(args.pod_role, "Archiving Decode XLA/vLLM compile cache to /models...")
        cache_info = archive_compile_cache("decode", out_dir.parent / "compile_cache")
        (out_dir / "compile_cache_decode.json").write_text(json.dumps(cache_info, indent=2))
        shutil.copy2(srv_log, out_dir / "poolb_h0_pd_decode.log")
        shutil.copy2("/dev/shm/pd_setup.log", out_dir / "poolb_h0_pd_setup.log")
        srv_proc.terminate()
        return 0

    # --- poola-h0: Start toy_proxy_server (:8000) -> Pool A Prefill (:8400) + Pool B Decode (:9400) ---
    log(args.pod_role, f"Starting toy_proxy_server on :8000 -> P=127.0.0.1:8400, D={peer_b_ip}:9400 ...")
    proxy_log_f = open("/dev/shm/pd_proxy.log", "w")
    proxy_proc = subprocess.Popen(
        [
            "python3",
            "/workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--prefiller-hosts", "127.0.0.1",
            "--prefiller-ports", "8400",
            "--prefiller-dp-size", "8",
            "--decoder-hosts", peer_b_ip,
            "--decoder-ports", "9400",
            "--decoder-dp-size", "8",
        ],
        stdout=proxy_log_f,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(30):
            if check_http("http://127.0.0.1:8000/healthcheck"):
                break
            time.sleep(2)
        else:
            raise RuntimeError("toy_proxy_server failed /healthcheck")

        log(args.pod_role, "Running 10-request Cross-Pool (Pool A -> Pool B) 1P1D Smoke + KV Transfer + DSpark tau check...")
        smoke_res = run_smoke_requests("http://127.0.0.1:8000", f"http://{peer_b_ip}:9400")
        (out_dir / "smoke_1p1d_cross_pool.json").write_text(json.dumps(smoke_res, indent=2))
        log(
            args.pod_role,
            f"SMOKE DONE: all_200={smoke_res['all_http_200']} sane={smoke_res['all_sane_output']} "
            f"tau={smoke_res['dspark_tau']} out_tok/s={smoke_res['aggregate_out_tok_s']}",
        )
    finally:
        (coord_dir / "smoke_done.json").write_text(json.dumps({"ts": time.time()}))
        log(args.pod_role, "Archiving Prefill XLA/vLLM compile cache to /models...")
        cache_info = archive_compile_cache("prefill", out_dir.parent / "compile_cache")
        (out_dir / "compile_cache_prefill.json").write_text(json.dumps(cache_info, indent=2))
        shutil.copy2(srv_log, out_dir / "poola_h0_pd_prefill.log")
        shutil.copy2("/dev/shm/pd_setup.log", out_dir / "poola_h0_pd_setup.log")
        if Path("/dev/shm/pd_proxy.log").exists():
            shutil.copy2("/dev/shm/pd_proxy.log", out_dir / "poola_h0_pd_proxy.log")
        proxy_proc.terminate()
        srv_proc.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
