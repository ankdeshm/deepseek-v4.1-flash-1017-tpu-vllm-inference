#!/usr/bin/env python3
"""Step 1 R0 Reproducibility Anchor In-Pod Boot & Service Supervisor on Pool A.

Brings up 1P1D on ank-tpu-v7x-8-pool-a (single-host 4-chip 2x2x1 isolation on 2x2x2 slice):
  - prefill (r0-prefill-poola-h0 on gke-tpu-51ea19d7-rqkr, 192.168.100.43):
      DP=8, TP=1, EP=8, CED_PREFILL=1, MAX_NUM_SEQS=64, MNBT=8192,
      ENABLE_PREFIX_CACHING=1, PREFIX_CACHE_RETENTION_INTERVAL=1024, :8400
      + toy_proxy_server.py on :8000
  - decode  (r0-decode-poola-h1 on gke-tpu-51ea19d7-xq6s, 192.168.100.38):
      DP=8, TP=1, EP=8, MAX_NUM_SEQS=48, MNBT=288,
      COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 288], ENABLE_DSPARK=1 (K=5),
      ASYNC_SCHEDULING=1, ENABLE_PREFIX_CACHING=0, :9400
Restores Step 0 warm XLA/vLLM compile caches before launching vLLM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request


GCS_BUCKET = "ank-usc1-bucket-314837540096"
STACK_PREFIX = "dsv41-stack"
MODEL_PREFIX = "models/deepseek-ai/DeepSeek-V4.1-Flash-1017/"
CACHE_PREFIX = "dsv41-v7x-llmd-20261106/compile_cache"
PREFILL_EXPECTED_IP = "192.168.100.43"
DECODE_EXPECTED_IP = "192.168.100.38"


def log(role: str, msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [r0-{role}] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["access_token"]


def gcs_download_file(bucket: str, obj_name: str, dst: Path) -> float:
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    for att in range(5):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                f"{urllib.parse.quote(obj_name, safe='')}?alt=media"
            )
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(dst, "wb") as f:
                while True:
                    chunk = resp.read(16 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return time.time() - t0
        except Exception:
            if att == 4:
                raise
            time.sleep(2 * (att + 1))
    return time.time() - t0


def check_http(url: str, timeout: float = 4.0) -> bool:
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


def unpack_stack_and_clean_shm(role: str) -> None:
    log(role, "Cleaning stale /dev/shm files (preserving valid DeepSeek-V4.1-Flash shards if present) and unpacking stack...")
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
    files = [
        "in_pod_setup.sh",
        "in_pod_serve.sh",
        "pd_gates.py",
        "toy_proxy_server.py",
        "launch_dsv41_disagg_role.sh",
    ]
    if role == "prefill":
        files.append("requests_mocked.jsonl")
    for fn in files:
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


def stage_checkpoint_and_wheels(role: str, host_ip: str, run_id: str) -> dict:
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
t0 = time.time(); tasks = []; total = 0; reused = 0
for rel, sz in blobs:
    if not rel: continue
    dst = os.path.join(DST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == sz:
        reused += sz
        continue
    fd = os.open(dst, os.O_CREAT | os.O_WRONLY, 0o644); os.ftruncate(fd, sz); os.close(fd)
    total += sz
    for s in range(0, max(1, sz), CHUNK): tasks.append((rel, dst, s, min(s + CHUNK, sz)))
if tasks:
    print(f'Downloading {total/1e9:.2f} GB in {len(tasks)} chunks (reused {reused/1e9:.2f} GB) from gs://{BUCKET}/{PREFIX} ...', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex: list(ex.map(dl, tasks))
else:
    print(f'Reused all {reused/1e9:.2f} GB already size-verified in {DST}', flush=True)
open(f'{DST}/.done_all', 'w').close()
elapsed = time.time() - t0
tot_all = total + reused
print(f'CKPT_READY bytes={tot_all} elapsed_s={elapsed:.2f} gb_per_s={(total/1e9)/max(elapsed,1e-6):.2f}', flush=True)
"""
    Path("/tmp/dsv41_dl.py").write_text(dl_py)
    tok = vm_access_token()
    Path("/tmp/dsv41_env.sh").write_text(
        f"export GCS_TOKEN='{tok}' DSV41_GCS_BUCKET='{GCS_BUCKET}' "
        f"DSV41_GCS_PREFIX='{MODEL_PREFIX}' DSV41_SHARDS='all'\n"
    )
    if role == "prefill":
        pd_env = {
            "PYTHONHASHSEED": "0",
            "ADVERTISE_HOST": host_ip,
            "RUN_ID": run_id,
            "PORT": "8400",
            "CED_PREFILL": "1",
            "ENABLE_PREFIX_CACHING": "1",
            "PREFIX_CACHE_RETENTION_INTERVAL": "1024",
            "MAX_NUM_SEQS": "64",
            "MAX_NUM_BATCHED_TOKENS": "8192",
            "COMPILE_SIZES": "[512, 2048, 8192]",
            "ENABLE_DSPARK": "0",
            "ASYNC_SCHEDULING": "0",
            "GPU_MEMORY_UTILIZATION": "0.85",
            "TPU_RAIDEN_TRANSFER_PARALLELISM": "1",
        }
    else:
        pd_env = {
            "PYTHONHASHSEED": "0",
            "ADVERTISE_HOST": host_ip,
            "RUN_ID": run_id,
            "PORT": "9400",
            "CED_PREFILL": "0",
            "ENABLE_PREFIX_CACHING": "0",
            "PREFIX_CACHE_RETENTION_INTERVAL": "0",
            "MAX_NUM_SEQS": "48",
            "MAX_NUM_BATCHED_TOKENS": "288",
            "COMPILE_SIZES": "[16, 40, 48, 64, 96, 192, 288]",
            "ENABLE_DSPARK": "1",
            "DSPARK_TOKENS": "5",
            "ASYNC_SCHEDULING": "1",
            "GPU_MEMORY_UTILIZATION": "0.85",
            "TPU_RAIDEN_TRANSFER_PARALLELISM": "1",
        }
    Path("/tmp/pd_env.sh").write_text("".join(f"export {k}='{v}'\n" for k, v in pd_env.items()))

    t0 = time.time()
    subprocess.run(["bash", "-lc", "bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup.log 2>&1"], check=True)
    setup_wall_s = round(time.time() - t0, 2)
    setup_txt = Path("/dev/shm/pd_setup.log").read_text(errors="replace")
    if "PD_SETUP_OK" not in setup_txt:
        raise RuntimeError(f"in_pod_setup.sh failed:\n{setup_txt[-2000:]}")

    # Verify aiohttp is installed for bench_empirical_isl_dsv41_rerun.py
    subprocess.run(
        ["bash", "-lc", "python3 -c 'import aiohttp' 2>/dev/null || pip install --no-cache-dir aiohttp >/dev/null"],
        check=True,
    )

    m_dl = re.search(r"CKPT_READY bytes=(\d+) elapsed_s=([0-9.]+) gb_per_s=([0-9.]+)", setup_txt)
    return {
        "setup_wall_s": setup_wall_s,
        "ckpt_bytes": int(m_dl.group(1)) if m_dl else None,
        "ckpt_gcs_to_shm_s": float(m_dl.group(2)) if m_dl else None,
        "ckpt_gcs_gb_per_s": float(m_dl.group(3)) if m_dl else None,
        "pd_env": pd_env,
    }


def restore_warm_compile_cache(role: str) -> dict:
    tar_obj = f"{CACHE_PREFIX}/compile_cache_{role}_dp8.tar.gz"
    local_tar = Path(f"/dev/shm/compile_cache_{role}_dp8.tar.gz")
    cache_root = Path(f"/dev/shm/pd_cache_{role}")
    log(role, f"Downloading Step 0 warm compile cache gs://{GCS_BUCKET}/{tar_obj} ...")
    dl_s = gcs_download_file(GCS_BUCKET, tar_obj, local_tar)
    tar_size_mb = round(local_tar.stat().st_size / 1e6, 2)

    top_lines = subprocess.check_output(
        ["bash", "-lc", f"tar -tzf {local_tar} 2>/dev/null | head -n 5 || true"],
        text=True,
    ).strip().splitlines()
    log(role, f"Tar top-level entries: {top_lines} (size={tar_size_mb} MB, dl={dl_s:.2f}s)")

    cache_root.mkdir(parents=True, exist_ok=True)
    t_ext = time.time()
    first = top_lines[0] if top_lines else ""
    if first.startswith(f"pd_cache_{role}/"):
        target_dir = Path("/dev/shm")
    elif first.startswith(f"/dev/shm/pd_cache_{role}/"):
        target_dir = Path("/")
    else:
        target_dir = cache_root

    proc = subprocess.run(
        ["tar", "-C", str(target_dir), "-xzvf", str(local_tar)],
        capture_output=True,
        text=True,
        check=False,
    )
    ext_s = round(time.time() - t_ext, 2)
    local_tar.unlink(missing_ok=True)

    dropped_partial = None
    if proc.returncode != 0:
        # Step 0 GCSFuse flush truncated the final ~KB of the tar.gz when the Step 0 pod exited.
        # Every file printed BEFORE the final member in `tar -xzvf` stdout was completely extracted;
        # remove the final member so no truncated/corrupt cache entry remains.
        extracted = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip() and not ln.strip().endswith("/")]
        if extracted:
            last_rel = extracted[-1].lstrip("/")
            cand = target_dir / last_rel
            if cand.exists() and cand.is_file():
                cand.unlink(missing_ok=True)
                dropped_partial = str(cand)
        log(
            role,
            f"NOTE: tar exited rc={proc.returncode} on trailing archive EOF; removed incomplete final member={dropped_partial}",
        )

    vllm_files = sum(1 for p in (cache_root / "vllm").rglob("*") if p.is_file()) if (cache_root / "vllm").exists() else 0
    xla_files = sum(1 for p in (cache_root / "xla").rglob("*") if p.is_file()) if (cache_root / "xla").exists() else 0
    log(
        role,
        f"Warm compile cache restored to {cache_root} in {ext_s}s: vllm_files={vllm_files}, xla_files={xla_files}",
    )
    return {
        "tar_obj": tar_obj,
        "tar_size_mb": tar_size_mb,
        "download_s": round(dl_s, 2),
        "extract_s": ext_s,
        "tar_returncode": proc.returncode,
        "dropped_trailing_partial_file": dropped_partial,
        "top_entries": top_lines,
        "vllm_files": vllm_files,
        "xla_files": xla_files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-role", required=True, choices=["prefill", "decode"])
    ap.add_argument("--out-dir", default="/models/dsv41-v7x-llmd-20261106/step1_r0")
    ap.add_argument("--run-id", default="step1_r0_20261106")
    args = ap.parse_args()

    role = args.pod_role
    out_dir = Path(args.out_dir)
    coord_dir = out_dir / "coord"
    coord_dir.mkdir(parents=True, exist_ok=True)

    my_ip = get_primary_ip()
    my_host = socket.gethostname()
    expected_ip = PREFILL_EXPECTED_IP if role == "prefill" else DECODE_EXPECTED_IP
    log(role, f"Started on host={my_host} ip={my_ip} (expected_ip={expected_ip})")
    if my_ip != expected_ip:
        log(role, f"WARNING: host IP {my_ip} differs from expected {expected_ip}; using actual {my_ip}")

    (coord_dir / f"host_{role}.json").write_text(
        json.dumps({"role": role, "hostname": my_host, "ip": my_ip, "ts": time.time()}, indent=2)
    )

    unpack_stack_and_clean_shm(role)
    ckpt_info = stage_checkpoint_and_wheels(role, my_ip, args.run_id)
    cache_info = restore_warm_compile_cache(role)

    (out_dir / f"setup_{role}.json").write_text(
        json.dumps({"host": my_host, "ip": my_ip, "ckpt": ckpt_info, "warm_cache": cache_info}, indent=2)
    )

    srv_port = 8400 if role == "prefill" else 9400
    log(role, f"Launching vLLM {role} server on :{srv_port} (warm cache enabled)...")
    t_boot = time.time()
    srv_proc = subprocess.Popen(["bash", "/tmp/in_pod_serve.sh", role])
    srv_log = Path(f"/dev/shm/pd_{role}.log")
    seen_lines = 0
    progress_keys = (
        "[PD-LAUNCH]",
        "Loading weights took",
        "Built EP mesh",
        "Fused EP MoE armed",
        "MXFP4 fused EP W4A8",
        "Memory statistics",
        "TPU KV cache size",
        "Raiden DSv4 pool admission complete",
        "Application startup complete",
        "deleted checkpoint",
    )
    fail_keys = (
        "Engine core initialization failed",
        "RuntimeBufferAllocationFailure",
        "EngineCore failed",
        "[PD-SERVE] server exited",
    )

    while True:
        if srv_proc.poll() is not None:
            tail = srv_log.read_text(errors="replace")[-4000:] if srv_log.exists() else ""
            raise RuntimeError(f"{role} exited early (rc={srv_proc.returncode}):\n{tail}")
        if srv_log.exists():
            lines = srv_log.read_text(errors="replace").splitlines()
            for ln in lines[seen_lines:]:
                if any(k in ln for k in progress_keys):
                    log(role, ln[:320])
                if any(fk in ln for fk in fail_keys):
                    tail = "\n".join(lines[-80:])
                    (out_dir / f"{role}_boot_fail.log").write_text(tail)
                    raise RuntimeError(f"{role} boot failure:\n{ln}\n{tail}")
            seen_lines = len(lines)
        if check_http(f"http://127.0.0.1:{srv_port}/health"):
            boot_s = round(time.time() - t_boot, 2)
            (coord_dir / f"healthy_{role}.json").write_text(
                json.dumps({"role": role, "boot_s": boot_s, "ip": my_ip, "host": my_host, "ts": time.time()}, indent=2)
            )
            log(role, f"{role.upper()} HEALTHY on :{srv_port} in {boot_s}s!")
            break
        if time.time() - t_boot > 2400:
            raise TimeoutError(f"{role} boot exceeded 2400s")
        time.sleep(8)

    proxy_proc = None
    if role == "prefill":
        log(role, "Waiting for decode peer healthy_decode.json...")
        while not (coord_dir / "healthy_decode.json").exists():
            if srv_proc.poll() is not None:
                raise RuntimeError("Prefill server exited while waiting for decode peer")
            time.sleep(4)
        dec_info = json.loads((coord_dir / "healthy_decode.json").read_text())
        dec_ip = dec_info["ip"]
        log(role, f"Starting toy_proxy_server.py on :8000 -> P=127.0.0.1:8400 (dp=8), D={dec_ip}:9400 (dp=8)...")
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
                "--decoder-hosts", dec_ip,
                "--decoder-ports", "9400",
                "--decoder-dp-size", "8",
            ],
            stdout=proxy_log_f,
            stderr=subprocess.STDOUT,
        )
        for _ in range(30):
            if check_http("http://127.0.0.1:8000/healthcheck"):
                break
            time.sleep(2)
        else:
            raise RuntimeError("toy_proxy_server failed /healthcheck on :8000")
        (coord_dir / "stack_ready.json").write_text(
            json.dumps(
                {
                    "prefill_url": "http://127.0.0.1:8400",
                    "decode_url": f"http://{dec_ip}:9400",
                    "proxy_url": "http://127.0.0.1:8000",
                    "ready_ts": time.time(),
                },
                indent=2,
            )
        )
        log(role, "ALL THREE ENDPOINTS HEALTHY (:8400/health, :9400/health, :8000/healthcheck) — READY FOR R0 BENCHMARK!")

    stop_requested = False

    def _on_sig(signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    try:
        while not stop_requested and not Path("/tmp/stop_r0").exists():
            if srv_proc.poll() is not None:
                log(role, f"ERROR: vLLM {role} exited unexpectedly with rc={srv_proc.returncode}")
                return 1
            if proxy_proc is not None and proxy_proc.poll() is not None:
                log(role, f"ERROR: toy_proxy_server exited unexpectedly with rc={proxy_proc.returncode}")
                return 1
            time.sleep(5)
    finally:
        log(role, "Persisting logs to /models and shutting down...")
        for src_p, dst_name in [
            (srv_log, f"pd_{role}.log"),
            (Path("/dev/shm/pd_setup.log"), f"pd_setup_{role}.log"),
            (Path("/dev/shm/pd_proxy.log"), "pd_proxy.log"),
        ]:
            if src_p.exists():
                try:
                    shutil.copy2(src_p, out_dir / dst_name)
                except Exception as exc:
                    log(role, f"Log copy warning ({src_p}): {exc}")
        if proxy_proc is not None:
            proxy_proc.terminate()
        srv_proc.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
