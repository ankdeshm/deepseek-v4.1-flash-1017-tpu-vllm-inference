#!/usr/bin/env python3
"""Step 2 Task B Probe on ank-tpu-v7x-8-pool-b (059c + cz6l):
1. Inspects GCE metadata (agent-worker-number, worker-network-endpoints, tpu-env) on both hosts.
2. Unpacks stack_27be32f509754e88.tar and copies tpu_connector.py, tpu_platform.py,
   tpu_rank_binding.py, tpu_worker.py, ep_mesh.py for multi-host DP=16 Raiden inspection.
3. Runs a live 2-host 16-TensorCore (2x2x2, world_size=16) torch_tpu ICI all_reduce + all_to_all
   smoke check using real GCE metadata + unix:///var/run/tpu-plugin/vbar.sock + prepare_mp_multihost_dp_env
   slicebuilder ordering (NO mock_metadata.py, NO metadata_hook.so, NO single-host overrides).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

GCS_BUCKET = "ank-usc1-bucket-314837540096"
STACK_PREFIX = "dsv41-stack"


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def get_meta(rel_path: str) -> str:
    url = f"http://169.254.169.254/computeMetadata/v1/{rel_path}"
    req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"<ERR:{e}>"


def vm_access_token() -> str:
    return json.loads(get_meta("instance/service-accounts/default/token"))["access_token"]


def gcs_download_file(bucket: str, obj_name: str, dst: Path) -> None:
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


def get_primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tag", required=True, choices=["poolb-059c", "poolb-cz6l"])
    ap.add_argument("--out-dir", default="/models/dsv41-v7x-llmd-20261106/step2_probe")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect GCE Metadata, vbar.sock, /dev/shm remount, and port availability
    subprocess.run(["bash", "-lc", "mount -o remount,size=850G /dev/shm || true"], check=False)
    df_shm = subprocess.check_output(["df", "-h", "/dev/shm"], text=True).strip()
    vbar_exists = Path("/var/run/tpu-plugin/vbar.sock").is_socket()
    vfio_devs = sorted(str(p) for p in Path("/dev/vfio").glob("*")) if Path("/dev/vfio").exists() else []
    listen_ports = sorted({
        int(ln.split()[1].split(":")[1], 16)
        for p in ("/proc/net/tcp", "/proc/net/tcp6")
        if Path(p).exists()
        for ln in Path(p).read_text().splitlines()[1:]
        if len(ln.split()) > 3 and ln.split()[3] == "0A"
    })

    agent_worker_num = get_meta("instance/attributes/agent-worker-number")
    worker_endpoints = get_meta("instance/attributes/worker-network-endpoints")
    tpu_env_raw = get_meta("instance/attributes/tpu-env")
    env_tpu = {
        k: v
        for k, v in sorted(os.environ.items())
        if any(p in k for p in ("TPU", "LWS", "VBAR", "PJRT", "CLOUD", "GOOGLE", "KUBERNETES_"))
    }

    meta_info = {
        "host_tag": args.host_tag,
        "hostname": socket.gethostname(),
        "ip": get_primary_ip(),
        "agent_worker_number": agent_worker_num,
        "worker_network_endpoints": worker_endpoints,
        "tpu_env": tpu_env_raw,
        "vbar_sock_exists": vbar_exists,
        "vfio_devs": vfio_devs,
        "df_shm_after_remount": df_shm,
        "pod_env_filtered": env_tpu,
        "listening_tcp_ports": listen_ports,
    }
    (out_dir / f"meta_lws_{args.host_tag}.json").write_text(json.dumps(meta_info, indent=2))
    log(
        args.host_tag,
        f"hostname={meta_info['hostname']} ip={meta_info['ip']} "
        f"agent-worker-number={agent_worker_num} endpoints={worker_endpoints} vbar={vbar_exists} ports={listen_ports}",
    )

    # 2. Unpack stack.tar & install wheels (without 510GB checkpoint download)
    log(args.host_tag, "Downloading and extracting stack_27be32f509754e88.tar + installing wheels...")
    gcs_download_file(GCS_BUCKET, f"{STACK_PREFIX}/stack_27be32f509754e88.tar", Path("/tmp/stack.tar"))
    install_sh = r"""
set -euo pipefail
mkdir -p /dev/shm/tmp
export TMPDIR=/dev/shm/tmp
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar && rm -f /tmp/stack.tar
git config --global --add safe.directory '*'
for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
pip install --no-cache-dir --no-deps "libtpu==0.0.48" portpicker==1.6.0 compressed-tensors==0.15.0 sympy >/dev/null
pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
if ls /workspace/whls_pd/*.whl >/dev/null 2>&1; then
  pip install --no-cache-dir --no-index --no-deps --force-reinstall /workspace/whls_pd/*.whl >/dev/null
fi
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
  [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf /workspace/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
done
for d in /usr/local/lib/python3.12/*-packages/vllm; do
  [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm/vllm)" ] && cp -rf /workspace/vllm/vllm/* "$d/" || true
done
"""
    subprocess.run(["bash", "-lc", install_sh], check=True)
    log(args.host_tag, "Stack wheels installed.")

    # On poolb-059c, export key source files for Task B.2 inspection
    if args.host_tag == "poolb-059c":
        src_dump = out_dir / "src_dump"
        src_dump.mkdir(parents=True, exist_ok=True)
        vtt = Path("/workspace/vllm-torchtpu/src/vllm_torchtpu")
        for rel in [
            "distributed/kv_transfer/tpu_connector.py",
            "platforms/tpu_platform.py",
            "worker/tpu_rank_binding.py",
            "worker/tpu_worker.py",
            "distributed/ep_mesh.py",
            "distributed/tpu_mp_multihost.py",
            "envs.py",
        ]:
            p = vtt / rel
            if p.exists():
                shutil.copy2(p, src_dump / p.name)
        # Also search for any other files under distributed/kv_transfer/
        kv_dir = vtt / "distributed/kv_transfer"
        if kv_dir.exists():
            for kp in sorted(kv_dir.rglob("*.py")):
                shutil.copy2(kp, src_dump / f"kv__{kp.name}")
        log(args.host_tag, f"Exported {len(list(src_dump.iterdir()))} source files to {src_dump}")

    # 3. Coordinate both hosts and run 16-TensorCore (2x2x2) torch_tpu ICI smoke test!
    ready_file = out_dir / f"ready_{args.host_tag}.json"
    ready_file.write_text(
        json.dumps(
            {
                "host_tag": args.host_tag,
                "ip": meta_info["ip"],
                "agent_worker_number": int(agent_worker_num),
            }
        )
    )
    t_wait = time.time()
    while not ((out_dir / "ready_poolb-059c.json").exists() and (out_dir / "ready_poolb-cz6l.json").exists()):
        if time.time() - t_wait > 180:
            raise TimeoutError("Timed out waiting for peer host in step2_probe")
        time.sleep(1)

    r0 = json.loads((out_dir / "ready_poolb-059c.json").read_text())
    r1 = json.loads((out_dir / "ready_poolb-cz6l.json").read_text())
    peers = sorted([r0, r1], key=lambda x: x["agent_worker_number"])
    worker0_ip = peers[0]["ip"]
    worker1_ip = peers[1]["ip"]
    my_node_rank = int(agent_worker_num)

    # Test script exercising 16-rank (8 procs/host x 2 hosts) torch_tpu ICI all_reduce + all_to_all
    smoke16_py = r"""
import os, sys, time, json, traceback
try:
    import torch
    import torch.distributed as dist
    import torch_tpu

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    t0 = time.time()

    # Mirror vllm_torchtpu's tpu_worker.py / tpu_rank_binding.py per-worker setup:
    n_dev = torch.tpu.device_count()
    dev_idx = 0 if n_dev == 1 else local_rank
    torch.tpu.set_device(dev_idx)
    dev = torch.device(f"tpu:{dev_idx}")
    dist.init_process_group(backend="tpu", init_method="env://")

    # 1) 16-way ICI all_reduce
    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device=dev) * (rank + 1)
    dist.all_reduce(x)
    torch.tpu.synchronize()
    ar_val = float(x[0, 0].cpu().item())
    ar_exp = float(sum(range(1, world + 1)))  # 136.0 for world=16
    assert ar_val == ar_exp, f"rank {rank}: all_reduce got {ar_val}, expected {ar_exp}"

    # 2) 16-way ICI all_to_all_single (simulating EP=16 dispatch/combine!)
    send_t = torch.arange(world, dtype=torch.float32, device=dev) + (rank * 100.0)
    recv_t = torch.zeros(world, dtype=torch.float32, device=dev)
    dist.all_to_all_single(recv_t, send_t)
    torch.tpu.synchronize()
    recv_cpu = recv_t.cpu().tolist()
    exp_cpu = [src * 100.0 + rank for src in range(world)]
    assert recv_cpu == exp_cpu, f"rank {rank}: all_to_all got {recv_cpu}, expected {exp_cpu}"

    elapsed = round(time.time() - t0, 3)
    if rank in (0, 8):
        print(f"ICI16_PASS_RANK_{rank} " + json.dumps({
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world,
            "device_count_visible": n_dev,
            "all_reduce_sum": ar_val,
            "all_to_all_ok": True,
            "elapsed_s": elapsed,
        }), flush=True)
    dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    Path("/tmp/smoke16.py").write_text(smoke16_py)

    # Build TORCH_TPU_SLICEBUILDER_ADDRESSES & TORCH_TPU_TOPOLOGY via vllm_torchtpu's own helper!
    topo_out = subprocess.check_output(
        [
            "python3",
            "-c",
            "import sys; sys.path.insert(0, '/workspace/vllm-torchtpu/src'); "
            "from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology; "
            "from vllm_torchtpu import envs; "
            "print(json.dumps({'topology': get_tpu_multihost_topology(16), 'base_port': envs.TORCH_TPU_BASE_PORT}))"
            if False
            else "import sys, json; sys.path.insert(0, '/workspace/vllm-torchtpu/src'); "
            "from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology; "
            "from vllm_torchtpu import envs; "
            "print(json.dumps({'topology': get_tpu_multihost_topology(16), 'base_port': envs.TORCH_TPU_BASE_PORT}))",
        ],
        text=True,
    ).strip()
    topo_cfg = json.loads(topo_out.splitlines()[-1])
    base_port = int(topo_cfg["base_port"])
    topology = topo_cfg["topology"]
    sb_addrs = ",".join(
        f"{ip}:{base_port + slot}" for ip in (worker0_ip, worker1_ip) for slot in range(8)
    )
    log(args.host_tag, f"Running 2-host 16-TensorCore ICI smoke check: node_rank={my_node_rank} topology={topology} sb={sb_addrs}")

    run_env = os.environ.copy()
    # Clean any single-host overrides; keep real GCE metadata + vbar.sock!
    for k in [
        "TPU_SKIP_MDS_QUERY",
        "TPU_HOST_BOUNDS",
        "HOST_BOUNDS",
        "LD_PRELOAD",
        "GCE_METADATA_HOST",
        "GCE_METADATA_IP",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "TPU_VISIBLE_CHIPS",
    ]:
        run_env.pop(k, None)

    if vbar_exists:
        run_env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
    run_env["PJRT_DEVICE"] = "TPU"
    run_env["TORCH_TPU_TOPOLOGY"] = topology
    run_env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addrs
    run_env["PYTHONPATH"] = "/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"

    t_smoke = time.time()
    proc = subprocess.run(
        [
            "torchrun",
            "--nnodes=2",
            "--nproc_per_node=8",
            f"--node_rank={my_node_rank}",
            f"--master_addr={worker0_ip}",
            "--master_port=29650",
            "/tmp/smoke16.py",
        ],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    smoke_wall = round(time.time() - t_smoke, 2)
    combined = proc.stdout + "\n" + proc.stderr
    passed = proc.returncode == 0 and (f"ICI16_PASS_RANK_{my_node_rank * 8}" in combined)
    res = {
        "host_tag": args.host_tag,
        "node_rank": my_node_rank,
        "worker0_ip": worker0_ip,
        "worker1_ip": worker1_ip,
        "topology": topology,
        "sb_addrs": sb_addrs,
        "returncode": proc.returncode,
        "passed": passed,
        "wall_s": smoke_wall,
        "output_tail": combined[-4000:],
    }
    (out_dir / f"ici16_smoke_{args.host_tag}.json").write_text(json.dumps(res, indent=2))
    log(args.host_tag, f"16-TensorCore ICI smoke passed={passed} rc={proc.returncode} in {smoke_wall}s!")
    if not passed:
        log(args.host_tag, f"FAILURE OUTPUT:\n{combined[-3000:]}")
    log(args.host_tag, "PROBE_COMPLETE")
    time.sleep(600)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
