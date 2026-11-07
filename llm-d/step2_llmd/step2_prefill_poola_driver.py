#!/usr/bin/env python3
"""Step 2b: llm-d Prefill (2 x 4-chip DP=8, EP=8 pods) In-Pod Driver & Supervisor on Pool A.

Brings up two independent single-host 4-chip (2x2x1 isolation on 2x2x2 slice) Prefill servers
that register directly into InferencePool `pd-disaggregation` on HTTP :8000 (no sidecar):
  - h0 (`llmd-prefill-poola-h0` on `gke-tpu-51ea19d7-rqkr`, `192.168.100.43`):
      RUN_ID=llmd_h0, TPU_KV_TRANSFER_NAMESPACE=prefill_llmd_h0, TPU_RAIDEN_ENGINE_ID=prefill-engine-llmd-h0
  - h1 (`llmd-prefill-poola-h1` on `gke-tpu-51ea19d7-xq6s`, `192.168.100.38`):
      RUN_ID=llmd_h1, TPU_KV_TRANSFER_NAMESPACE=prefill_llmd_h1, TPU_RAIDEN_ENGINE_ID=prefill-engine-llmd-h1

Both pods serve `deepseek-ai/DeepSeek-V4.1-Flash` on :8000 with Raiden Stage-3 `kv_producer`,
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`, `PYTHONHASHSEED=0`, `CED_PREFILL=1`, `ENABLE_PREFIX_CACHING=1`,
`PREFIX_CACHE_RETENTION_INTERVAL=1024`, `MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=8192`,
`COMPILE_SIZES=[512, 2048, 8192]`, `GPU_MEMORY_UTILIZATION=0.85`, FP8 KV cache, and tool/reasoning parsers.
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
import urllib.error
import urllib.parse
import urllib.request


GCS_BUCKET = "ank-usc1-bucket-314837540096"
STACK_PREFIX = "dsv41-stack"
MODEL_PREFIX = "models/deepseek-ai/DeepSeek-V4.1-Flash-1017/"
CACHE_PREFIX = "dsv41-v7x-llmd-20261106/compile_cache"
CLEAN_CACHE_OBJ = f"{CACHE_PREFIX}/compile_cache_prefill_dp8_phs0.tar.gz"
STEP0_CACHE_OBJ = f"{CACHE_PREFIX}/compile_cache_prefill_dp8.tar.gz"
SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"

SHARD_SPECS = {
    "h0": {
        "expected_ip": "192.168.100.43",
        "expected_node": "gke-tpu-51ea19d7-rqkr",
        "run_id": "llmd_h0",
        "kv_namespace": "prefill_llmd_h0",
        "engine_id": "prefill-engine-llmd-h0",
    },
    "h1": {
        "expected_ip": "192.168.100.38",
        "expected_node": "gke-tpu-51ea19d7-xq6s",
        "run_id": "llmd_h1",
        "kv_namespace": "prefill_llmd_h1",
        "engine_id": "prefill-engine-llmd-h1",
    },
}


def log(shard: str, msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [llmd-prefill-{shard}] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["access_token"]


def gcs_object_exists(bucket: str, obj_name: str) -> tuple[bool, int]:
    try:
        tok = vm_access_token()
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj_name, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            meta = json.loads(resp.read())
            return True, int(meta.get("size", 0))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False, 0
        return False, 0
    except Exception:
        return False, 0


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


def gcs_upload_file_streaming(bucket: str, obj_name: str, local_path: Path, content_type: str = "application/gzip") -> dict:
    sz = local_path.stat().st_size
    t0 = time.time()
    for att in range(4):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
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
                with urllib.request.urlopen(req, timeout=900) as resp:
                    meta = json.loads(resp.read())
            remote_sz = int(meta.get("size", 0))
            if remote_sz != sz:
                raise RuntimeError(f"Uploaded size mismatch: remote={remote_sz} != local={sz}")
            return {
                "gcs_uri": f"gs://{bucket}/{obj_name}",
                "size_bytes": sz,
                "remote_size_bytes": remote_sz,
                "md5Hash": meta.get("md5Hash"),
                "upload_s": round(time.time() - t0, 2),
            }
        except Exception:
            if att == 3:
                raise
            time.sleep(3 * (att + 1))
    raise RuntimeError("Unreachable")


def get_primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def check_models_ready(port: int = 8000, expected_model: str = SERVED_MODEL_NAME) -> tuple[bool, dict | None]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=4.0) as resp:
            if resp.status != 200:
                return False, None
            payload = json.loads(resp.read())
            ids = [m.get("id") for m in payload.get("data", [])]
            return expected_model in ids, payload
    except Exception:
        return False, None


def run_local_smoke_completion(port: int = 8000) -> dict:
    body = json.dumps(
        {
            "model": SERVED_MODEL_NAME,
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return {
            "status": resp.status,
            "latency_s": round(time.time() - t0, 3),
            "id": data.get("id"),
            "model": data.get("model"),
            "text": (data.get("choices") or [{}])[0].get("text", ""),
            "usage": data.get("usage"),
        }


def unpack_stack_and_patch_launcher(shard: str) -> None:
    log(shard, "Cleaning stale /dev/shm files (preserving DeepSeek-V4.1-Flash if present) and unpacking stack...")
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
    for fn in ["in_pod_setup.sh", "in_pod_serve.sh", "launch_dsv41_disagg_role.sh"]:
        gcs_download_file(GCS_BUCKET, f"{STACK_PREFIX}/{fn}", Path(f"/tmp/{fn}"))

    stage_sh = r"""
set -euo pipefail
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar && rm -f /tmp/stack.tar
cd /workspace
(gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl) \
  || (apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null && \
      gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl)
cp -f /tmp/launch_dsv41_disagg_role.sh /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh
chmod +x /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh /tmp/in_pod_setup.sh /tmp/in_pod_serve.sh
git config --global --add safe.directory '*' || true
"""
    subprocess.run(["bash", "-lc", stage_sh], check=True)

    # Patch launch_dsv41_disagg_role.sh so explicit TPU_KV_TRANSFER_NAMESPACE,
    # TPU_RAIDEN_ENGINE_ID, VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120, and PYTHONHASHSEED=0 are honored verbatim.
    launcher_path = Path("/workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh")
    txt = launcher_path.read_text()
    txt = txt.replace(
        'export TPU_KV_TRANSFER_NAMESPACE="${ROLE}_${RUN_ID}"',
        'export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-120}"\n'
        'export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"\n'
        'export TPU_KV_TRANSFER_NAMESPACE="${TPU_KV_TRANSFER_NAMESPACE:-${ROLE}_${RUN_ID}}"',
    )
    txt = txt.replace(
        'export TPU_RAIDEN_ENGINE_ID="${ROLE}-engine-${RUN_ID}"',
        'export TPU_RAIDEN_ENGINE_ID="${TPU_RAIDEN_ENGINE_ID:-${ROLE}-engine-${RUN_ID}}"',
    )
    launcher_path.write_text(txt)
    assert "VLLM_HTTP_TIMEOUT_KEEP_ALIVE" in launcher_path.read_text(), "Failed to patch launch_dsv41_disagg_role.sh"


def stage_checkpoint_and_wheels(shard: str, host_ip: str, spec: dict) -> dict:
    dl_py = r"""
import os, sys, json, time, urllib.request, urllib.parse, concurrent.futures
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
    pd_env = {
        "PYTHONHASHSEED": "0",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
        "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
        "ADVERTISE_HOST": host_ip,
        "RUN_ID": spec["run_id"],
        "TPU_KV_TRANSFER_NAMESPACE": spec["kv_namespace"],
        "TPU_RAIDEN_ENGINE_ID": spec["engine_id"],
        "PORT": "8000",
        "KV_ROLE": "kv_producer",
        "KV_PORT": "14579",
        "CTRL_PORT": "27000",
        "KV_TRANSFER_PORT": "9100",
        "SIDE_CHANNEL_PORT": "9600",
        "TPU_RAIDEN_TRANSFER_PARALLELISM": "1",
        "CED_PREFILL": "1",
        "ENABLE_PREFIX_CACHING": "1",
        "PREFIX_CACHE_RETENTION_INTERVAL": "1024",
        "MAX_NUM_SEQS": "64",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "COMPILE_SIZES": "[512, 2048, 8192]",
        "ENABLE_DSPARK": "0",
        "ASYNC_SCHEDULING": "0",
        "GPU_MEMORY_UTILIZATION": "0.85",
    }
    Path("/tmp/pd_env.sh").write_text("".join(f"export {k}='{v}'\n" for k, v in pd_env.items()))

    t0 = time.time()
    subprocess.run(["bash", "-lc", "bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup.log 2>&1"], check=True)
    setup_wall_s = round(time.time() - t0, 2)
    setup_txt = Path("/dev/shm/pd_setup.log").read_text(errors="replace")
    if "PD_SETUP_OK" not in setup_txt:
        raise RuntimeError(f"in_pod_setup.sh failed:\n{setup_txt[-2000:]}")

    m_dl = re.search(r"CKPT_READY bytes=(\d+) elapsed_s=([0-9.]+) gb_per_s=([0-9.]+)", setup_txt)
    return {
        "setup_wall_s": setup_wall_s,
        "ckpt_bytes": int(m_dl.group(1)) if m_dl else None,
        "ckpt_gcs_to_shm_s": float(m_dl.group(2)) if m_dl else None,
        "ckpt_gcs_gb_per_s": float(m_dl.group(3)) if m_dl else None,
        "pd_env": pd_env,
    }


def restore_warm_compile_cache(shard: str) -> dict:
    has_clean, clean_sz = gcs_object_exists(GCS_BUCKET, CLEAN_CACHE_OBJ)
    tar_obj = CLEAN_CACHE_OBJ if (has_clean and clean_sz > 10_000_000) else STEP0_CACHE_OBJ
    local_tar = Path("/dev/shm/compile_cache_prefill_restore.tar.gz")
    cache_root = Path("/dev/shm/pd_cache_prefill")

    log(shard, f"Downloading warm prefill compile cache gs://{GCS_BUCKET}/{tar_obj} ...")
    dl_s = gcs_download_file(GCS_BUCKET, tar_obj, local_tar)
    tar_size_mb = round(local_tar.stat().st_size / 1e6, 2)

    top_lines = subprocess.check_output(
        ["bash", "-lc", f"tar -tzf {local_tar} 2>/dev/null | head -n 5 || true"],
        text=True,
    ).strip().splitlines()
    log(shard, f"Tar top-level entries: {top_lines} (size={tar_size_mb} MB, dl={dl_s:.2f}s)")

    cache_root.mkdir(parents=True, exist_ok=True)
    t_ext = time.time()
    first = top_lines[0] if top_lines else ""
    if first.startswith("pd_cache_prefill/"):
        target_dir = Path("/dev/shm")
    elif first.startswith("/dev/shm/pd_cache_prefill/"):
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
        extracted = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip() and not ln.strip().endswith("/")]
        if extracted:
            last_rel = extracted[-1].lstrip("/")
            cand = target_dir / last_rel
            if cand.exists() and cand.is_file():
                cand.unlink(missing_ok=True)
                dropped_partial = str(cand)
        log(
            shard,
            f"NOTE: tar exited rc={proc.returncode} on trailing archive EOF; removed incomplete final member={dropped_partial}",
        )

    vllm_files = sum(1 for p in (cache_root / "vllm").rglob("*") if p.is_file()) if (cache_root / "vllm").exists() else 0
    xla_files = sum(1 for p in (cache_root / "xla").rglob("*") if p.is_file()) if (cache_root / "xla").exists() else 0
    log(
        shard,
        f"Warm compile cache restored to {cache_root} in {ext_s}s: vllm_files={vllm_files}, xla_files={xla_files}",
    )
    return {
        "tar_obj": tar_obj,
        "tar_size_mb": tar_size_mb,
        "download_s": round(dl_s, 2),
        "extract_s": ext_s,
        "tar_returncode": proc.returncode,
        "dropped_trailing_partial_file": dropped_partial,
        "vllm_files": vllm_files,
        "xla_files": xla_files,
    }


def archive_clean_phs0_prefill_cache(shard: str) -> dict:
    """Packs clean PYTHONHASHSEED=0 prefill compile cache on h0 and streams to GCS."""
    local_tar = Path("/dev/shm/compile_cache_prefill_dp8_phs0.tar.gz")
    local_tar.unlink(missing_ok=True)
    log(shard, "Packing clean PYTHONHASHSEED=0 /dev/shm/pd_cache_prefill -> temporary tarball in /dev/shm ...")
    t0 = time.time()
    try:
        subprocess.run(
            [
                "tar",
                "-C",
                "/dev/shm",
                "--exclude=pd_cache_prefill/tmp",
                "--exclude=pd_cache_prefill/plan_dump",
                "-I",
                "gzip -1",
                "-cf",
                str(local_tar),
                "pd_cache_prefill",
            ],
            check=True,
        )
        pack_s = round(time.time() - t0, 2)
        # Verify archive integrity locally before uploading (`gzip -t`)
        subprocess.run(["gzip", "-t", str(local_tar)], check=True)
        sz_mb = round(local_tar.stat().st_size / 1e6, 2)
        log(shard, f"Packed & verified clean archive ({sz_mb} MB in {pack_s}s); streaming to gs://{GCS_BUCKET}/{CLEAN_CACHE_OBJ} ...")
        up_info = gcs_upload_file_streaming(GCS_BUCKET, CLEAN_CACHE_OBJ, local_tar)
        up_info["pack_s"] = pack_s
        up_info["gzip_verified"] = True
        log(shard, f"Uploaded clean PYTHONHASHSEED=0 prefill cache: {up_info}")
        return up_info
    finally:
        local_tar.unlink(missing_ok=True)
        log(shard, "Cleaned up temporary /dev/shm/compile_cache_prefill_dp8_phs0.tar.gz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, choices=["h0", "h1"])
    ap.add_argument("--out-dir", default="/models/dsv41-v7x-llmd-20261106/step2_llmd/prefill_poola")
    args = ap.parse_args()

    shard = args.shard
    spec = SHARD_SPECS[shard]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    my_ip = get_primary_ip()
    my_host = socket.gethostname()
    log(shard, f"Starting on host={my_host} ip={my_ip} (expected={spec['expected_node']} {spec['expected_ip']})")
    if my_ip != spec["expected_ip"]:
        log(shard, f"WARNING: primary IP {my_ip} differs from expected {spec['expected_ip']}; advertising {my_ip}")

    unpack_stack_and_patch_launcher(shard)
    ckpt_info = stage_checkpoint_and_wheels(shard, my_ip, spec)
    cache_info = restore_warm_compile_cache(shard)

    (out_dir / f"setup_prefill_{shard}.json").write_text(
        json.dumps(
            {
                "shard": shard,
                "host": my_host,
                "ip": my_ip,
                "spec": spec,
                "ckpt": ckpt_info,
                "warm_cache_restore": cache_info,
                "ts": time.time(),
            },
            indent=2,
        )
    )

    log(
        shard,
        f"Launching vLLM prefill on :8000 (model={SERVED_MODEL_NAME}, run_id={spec['run_id']}, "
        f"ns={spec['kv_namespace']}, engine_id={spec['engine_id']})...",
    )
    t_boot = time.time()
    srv_proc = subprocess.Popen(["bash", "/tmp/in_pod_serve.sh", "prefill"])
    srv_log = Path("/dev/shm/pd_prefill.log")
    seen_lines = 0
    progress_keys = (
        "[PD-LAUNCH]",
        "Loading weights took",
        "Built EP mesh",
        "Fused EP MoE armed",
        "MXFP4 fused EP W4A8",
        "Memory statistics",
        "KV cache size",
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

    models_payload = None
    while True:
        if srv_proc.poll() is not None:
            tail = srv_log.read_text(errors="replace")[-4000:] if srv_log.exists() else ""
            raise RuntimeError(f"prefill-{shard} exited early (rc={srv_proc.returncode}):\n{tail}")
        if srv_log.exists():
            lines = srv_log.read_text(errors="replace").splitlines()
            for ln in lines[seen_lines:]:
                if any(k in ln for k in progress_keys):
                    log(shard, ln[:320])
                if any(fk in ln for fk in fail_keys):
                    tail = "\n".join(lines[-80:])
                    (out_dir / f"prefill_{shard}_boot_fail.log").write_text(tail)
                    raise RuntimeError(f"prefill-{shard} boot failure:\n{ln}\n{tail}")
            seen_lines = len(lines)
        ok_models, models_payload = check_models_ready(8000, SERVED_MODEL_NAME)
        if ok_models:
            boot_s = round(time.time() - t_boot, 2)
            log(shard, f"PREFILL {shard.upper()} READY on :8000/v1/models ({SERVED_MODEL_NAME}) in {boot_s}s!")
            break
        if time.time() - t_boot > 2400:
            raise TimeoutError(f"prefill-{shard} boot exceeded 2400s")
        time.sleep(6)

    # Wait until in_pod_serve.sh has cleaned up the 225 GB safetensor shards from /dev/shm (<=15s)
    for _ in range(6):
        if srv_log.exists() and "deleted checkpoint shards" in srv_log.read_text(errors="replace"):
            break
        time.sleep(3)

    log(shard, "Running direct local /v1/completions smoke check (max_tokens=16, temperature=0.0, no kv_transfer_params)...")
    smoke_res = run_local_smoke_completion(8000)
    log(shard, f"Local smoke completion OK: {smoke_res}")

    archive_info = None
    if shard == "h0":
        try:
            archive_info = archive_clean_phs0_prefill_cache(shard)
        except Exception as exc:
            log(shard, f"WARNING: Clean cache archive upload encountered error: {exc}")
            archive_info = {"error": str(exc)}

    status_doc = {
        "shard": shard,
        "pod_name": f"llmd-prefill-poola-{shard}",
        "host": my_host,
        "ip": my_ip,
        "port": 8000,
        "served_model_name": SERVED_MODEL_NAME,
        "run_id": spec["run_id"],
        "kv_namespace": spec["kv_namespace"],
        "raiden_engine_id": spec["engine_id"],
        "http_timeout_keep_alive_s": 120,
        "pythonhashseed": 0,
        "boot_s": boot_s,
        "models_response": models_payload,
        "local_smoke_completion": smoke_res,
        "clean_cache_archive": archive_info,
        "ready_ts": time.time(),
    }
    (out_dir / f"ready_prefill_{shard}.json").write_text(json.dumps(status_doc, indent=2))
    shutil.copy2(srv_log, out_dir / f"pd_prefill_{shard}_boot.log")
    log(shard, f"Wrote {out_dir / f'ready_prefill_{shard}.json'} — ENTERING WARM STEADY-STATE SUPERVISOR LOOP!")

    stop_requested = False

    def _on_sig(signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    last_log_sync = time.time()
    try:
        while not stop_requested:
            if srv_proc.poll() is not None:
                log(shard, f"ERROR: vLLM prefill-{shard} exited unexpectedly with rc={srv_proc.returncode}")
                shutil.copy2(srv_log, out_dir / f"pd_prefill_{shard}_crash.log")
                return 1
            if time.time() - last_log_sync >= 60.0 and srv_log.exists():
                try:
                    shutil.copy2(srv_log, out_dir / f"pd_prefill_{shard}.log")
                except Exception:
                    pass
                last_log_sync = time.time()
            time.sleep(5)
    finally:
        log(shard, "Shutting down prefill supervisor and syncing final logs...")
        if srv_log.exists():
            try:
                shutil.copy2(srv_log, out_dir / f"pd_prefill_{shard}.log")
            except Exception:
                pass
        srv_proc.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
