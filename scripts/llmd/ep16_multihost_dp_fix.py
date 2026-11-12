#!/usr/bin/env python3
"""Idempotent fix for vllm-torchtpu multi-host DP (DP=16, dp_local=8, EP=16) bootstrap.

Root causes fixed:
1. vllm_torchtpu/worker/tpu_rank_binding.py:
   In `get_tpu_worker_binding` (MoE EP + DP > 1 branch), `local_world_size` was set
   to `local_rank_offset + global_world` (= 16) instead of the per-host chip count
   `local_rank_offset + world_size * dp_size_local` (= 8), unlike `_slice_binding_from_env`
   (used by RayDistributedExecutorV2) and the non-EP branch right below it.
2. vllm_torchtpu/distributed/tpu_mp_multihost.py:
   `prepare_mp_multihost_dp_env` omitted `TPU_NUM_HOSTS`, `NNODES`, and `NODE_RANK`
   (which `prepare_mp_multihost_env` and `RayDistributedExecutorV2` set).
3. vllm_torchtpu/worker/tpu_worker.py:
   In `mp` multi-host DP (`WORLD_SIZE > 8`), 16 independent `DPEngineCoreProc` processes
   import vLLM/Torch staggered across several seconds before entering `init_device()`.
   Adding a pre-`init_distributed_environment` `TCPStore` barrier aligns all 16 workers
   to enter `PjrtBackend::GetClient()` simultaneously (<5ms skew), matching Ray V2.
"""

import glob
import os
import pathlib
import shutil

SRC_ROOT = pathlib.Path("/workspace/vllm-torchtpu/src/vllm_torchtpu")


def patch_rank_binding(path: pathlib.Path) -> bool:
    txt = path.read_text()
    old = """        dp_rank_local = parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_size_local = parallel_config.data_parallel_size_local or dp_size
            dp_rank_local = dp_rank % dp_size_local
        native_local_rank = dp_rank_local * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=global_world,
            )
        else:
            tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + global_world,"""

    new = """        dp_size_local_eff = parallel_config.data_parallel_size_local or dp_size
        dp_rank_local = parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = dp_rank % dp_size_local_eff
        native_local_rank = dp_rank_local * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        local_chip_count = world_size * dp_size_local_eff
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=local_chip_count,
            )
        else:
            tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + local_chip_count,"""

    if "local_world_size=local_rank_offset + local_chip_count," in txt and "dp_size_local_eff =" in txt:
        return False
    if old not in txt:
        raise RuntimeError(f"Expected snippet not found in {path}")
    path.write_text(txt.replace(old, new, 1))
    return True


def patch_mp_multihost(path: pathlib.Path) -> bool:
    txt = path.read_text()
    old = """    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DP_DONE_SENTINEL] = str(dp_size)"""

    new = """    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TPU_NUM_HOSTS"] = str(num_hosts)
    os.environ["NNODES"] = str(num_hosts)
    os.environ["NODE_RANK"] = str(node_index)
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DP_DONE_SENTINEL] = str(dp_size)"""

    if 'os.environ["NODE_RANK"] = str(node_index)' in txt:
        return False
    if old not in txt:
        raise RuntimeError(f"Expected snippet not found in {path}")
    path.write_text(txt.replace(old, new, 1))
    return True


def patch_tpu_worker_barrier(path: pathlib.Path) -> bool:
    txt = path.read_text()
    sentinel = "_TORCH_TPU_PRE_INIT_STORES"
    if sentinel in txt:
        return False

    old = """        dist_backend = current_platform.get_worker_distributed_backend(dist_world_size)

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment("""

    new = """        dist_backend = current_platform.get_worker_distributed_backend(dist_world_size)

        _g_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        _b_port = int(os.environ.get("TORCH_TPU_PRE_INIT_BARRIER_PORT", "0") or "0")
        if _g_world > 8 and _b_port > 0 and os.environ.get("MASTER_ADDR"):
            from datetime import timedelta as _td
            _g_rank = int(os.environ["RANK"])
            _b_store = torch.distributed.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=_b_port,
                world_size=_g_world,
                is_master=(_g_rank == 0),
                timeout=_td(seconds=300),
            )
            globals().setdefault("_TORCH_TPU_PRE_INIT_STORES", []).append(_b_store)
            _b_store.set(f"pre_init_ready_{_g_rank}", "1")
            _b_store.wait([f"pre_init_ready_{i}" for i in range(_g_world)], _td(seconds=300))
            logger.info(
                "TPU multi-host pre-GetClient barrier passed | rank=%d/%d port=%d",
                _g_rank,
                _g_world,
                _b_port,
            )

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment("""

    if old not in txt:
        raise RuntimeError(f"Expected snippet not found in {path}")
    path.write_text(txt.replace(old, new, 1))
    return True


def main() -> None:
    rb = SRC_ROOT / "worker" / "tpu_rank_binding.py"
    mh = SRC_ROOT / "distributed" / "tpu_mp_multihost.py"
    tw = SRC_ROOT / "worker" / "tpu_worker.py"
    c1 = patch_rank_binding(rb)
    c2 = patch_mp_multihost(mh)
    c3 = patch_tpu_worker_barrier(tw)
    for d in glob.glob("/usr/local/lib/python3.12/*-packages/vllm_torchtpu"):
        if os.path.isdir(d) and os.path.realpath(d) != os.path.realpath(str(SRC_ROOT)):
            shutil.copyfile(str(rb), os.path.join(d, "worker", "tpu_rank_binding.py"))
            shutil.copyfile(str(mh), os.path.join(d, "distributed", "tpu_mp_multihost.py"))
            shutil.copyfile(str(tw), os.path.join(d, "worker", "tpu_worker.py"))
    for pyc_dir in glob.glob("/workspace/vllm-torchtpu/**/__pycache__", recursive=True) + glob.glob(
        "/usr/local/lib/python3.12/*-packages/vllm_torchtpu/**/__pycache__", recursive=True
    ):
        shutil.rmtree(pyc_dir, ignore_errors=True)
    print(
        f"[EP16-FIX] Applied multi-host DP rank/env/barrier fix "
        f"(rank_binding={c1}, mp_multihost={c2}, worker_barrier={c3})",
        flush=True,
    )


if __name__ == "__main__":
    main()
