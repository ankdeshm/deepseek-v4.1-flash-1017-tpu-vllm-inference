# DeepSeek-V4.1-Flash-1017 Corrected Reproduction Manifest (Cloud TPU v7x)

## 1. Purpose and Summary of Configuration Fixes

This document provides a complete, standalone reproduction specification for serving **`deepseek-ai/DeepSeek-V4.1-Flash-1017`** (mixed-precision `FP8` attention / `MXFP4` routed experts, `FP8` KV cache) with Prefill/Decode (`P/D`) disaggregation over Raiden Stage 3 (`TPURaidenConnector`) on Cloud TPU v7x (`Ironwood`, `192 GiB` HBM per physical chip, `2` TensorCores per physical chip).

### Root Cause Fixed from the Original Shipped Decode Script
In the original artifact pack, the reference decode launch script listed `--max-num-seqs 16 --max-num-batched-tokens 96` (`COMPILE_SIZES="[16, 40, 48, 64, 96]"`), whereas the actual server log (`0909_decode.log`) had been launched with `--max-num-seqs 64 --max-num-batched-tokens 384`. Re-running `(16, 96)` with `DSpark` (`K=5`) enabled (`--speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'`) caused draft-token acceptance to collapse to `0%` (`mean_accept_len_tau = 1.0000`, `2,185 tok/s` across `8` chips) due to the `96`-token scheduler budget (`16 × (1 + 5) = 96`) interacting with `DSpark` multi-token verification padding.

### Corrected Decode Configuration (`A1-b` / `C1` Standard)
All disaggregated decode instances (`1P1D`, `1P2D`, `1P3D`, `1P4D`) use:
- **Scheduler caps**: `--max-num-seqs 48` and `--max-num-batched-tokens 288` (`48 × (1 + 5) = 288`)
- **Explicit XLA compile bucket ladder**: `--compilation-config '{"compile_sizes": [16, 40, 48, 64, 96, 192, 288]}'` (including the mandatory `288` bucket matching `MAX_NUM_BATCHED_TOKENS=288`)
- **Speculative decoding**: `--speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'` (`VLLM_SPEC_DECODE_TOKENS=5`, `VLLM_DSV41_DSPARK_SHARDED_CTX_KV=1`)
- **Asynchronous scheduling**: `--async-scheduling` (overlaps Python host scheduling and detokenization with TPU execution)
- **Per-host parallelism**: `4` physical TPU v7x chips (`2x2x1` single-host topology = `8` TensorCores) configured as `DP=8, TP=1, EP=8` (`--data-parallel-size 8 --data-parallel-size-local 8 --tensor-parallel-size 1 --enable-expert-parallel`)

---

## 2. Software Stack Provenance

| Component | Version / Commit SHA | Notes |
|---|---|---|
| Base Container Image | `vllm/vllm-tpu:nightly` | Python `3.12` runtime |
| `vllm` | `afbcce3152df0faf818fb37565c47a8740646cba` | Installed editable (`-e`) with `VLLM_TARGET_DEVICE=empty` |
| `vllm-torchtpu` | `1adb0d81fdd60aae25a78e514878d71274c14435` | Provides `torchtpu`, `torchtpu_layers`, `TPURaidenConnector`, `toy_proxy_server.py` |
| `libtpu` | `0.0.48` | Installed with `--no-deps` |
| Model Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `48` `safetensors` shards (`~475.25 GiB`), `FP8` non-expert + `MXFP4` MoE |

---

## 3. Hardware Topology Mapping (`1P1D`, `1P2D`, `1P3D`, `1P4D`)

Each Prefill (`P`) or Decode (`D`) instance occupies **1 TPU v7x host** (`4` physical v7x chips, `8` TensorCores, `2x2x1` topology, `768 GiB` aggregate HBM per host):

| Ratio | Prefill Instances (`P`) | Decode Instances (`D`) | Total Physical Chips | Recommended GKE Placement |
|---|---|---|---|---|
| `1P1D` | `1 × 4-chip` (`host0`) | `1 × 4-chip` (`host1`) | `8` chips | Single `2x2x2` (`8`-chip) nodepool (`2` hosts) |
| `1P2D` | `1 × 4-chip` (`host0`) | `2 × 4-chip` (`hosts 1–2`) | `12` chips | `3` hosts inside one `2x2x4` (`16`-chip) nodepool (zero cross-pool hop) |
| `1P3D` | `1 × 4-chip` (`host0`) | `3 × 4-chip` (`hosts 1–3`) | `16` chips | All `4` hosts of one `2x2x4` (`16`-chip) nodepool (zero cross-pool hop) |
| `1P4D` | `1 × 4-chip` (`host0`) | `4 × 4-chip` (`hosts 1–3` + `1` host on `8`-chip pool) | `20` chips | `2x2x4` (`16`-chip) nodepool + `1` host (`4` chips) on `2x2x2` nodepool |

> **Important Accounting Rule**: Always normalize cluster output throughput (`tok/s/chip`) by **total physical chips** across both Prefill and Decode (`8`, `12`, `16`, or `20` chips), never by decode-only chips.

---

## 4. Common Runtime & Kernel Environment Variables

Apply the following environment block on **both** Prefill and Decode hosts before invoking `vllm serve`:

```bash
ulimit -c 0
ulimit -n 65536 || true
mount -o remount,size=850G /dev/shm || true

export PJRT_DEVICE=TPU
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1 TOPOLOGY=2x2x1
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_VISIBLE_DEVICES=0,1,2,3
export TPU_RUNTIME_METRICS_PORTS="8431,8432,8433,8434,8435,8436,8437,8438"

# Runtime & compilation caches
export CACHE_ROOT="/dev/shm/pd_cache_${ROLE}"
mkdir -p "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/xla" "${CACHE_ROOT}/inductor" \
         "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/plan_dump"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export VLLM_XLA_CACHE_PATH="${CACHE_ROOT}/xla"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_ROOT}/xla"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export TMPDIR="${CACHE_ROOT}/tmp"

export OMP_NUM_THREADS=6 GOMP_SPINCOUNT=1000
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export VLLM_LOG_POST_WARMUP_HBM=1
export USE_PHASED_PROFILER=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 SPEC_WARMUP=0 VLLM_SPEC_WARMUP=0
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

# DeepSeek-V4.1-Flash MoE & Attention kernel flags
export TPU_ROPE_CACHE_TRUNCATE=1 TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_COLLECTION_CHUNK_SIZE=8192 VLLM_TPU_FAST_TOKEN_SUBSTITUTION=0
export USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v2
export MOE_FUSED_EP_ENABLE_W4A8=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1
export MOE_REQUANTIZE_BLOCK_SIZE=512 TPU_MOE_DENSE_EXPERT_THRESHOLD=0
export VLLM_TPU_MOE_FP8_DISPATCH=1 VLLM_DSV41_MOE_EP_V2_ALL39=0
export VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS=16384 TPU_MOE_SKIP_PADDED_TOKENS=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_DSV41_FAST_GROUPED_ATTN=1 VLLM_DSV41_REPL_SHARED_EXPERTS=1
export VLLM_DSV41_TP_ATTN=0 VLLM_DSV41_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_TP_ATTN_LOCAL_TOPK=1 VLLM_DSV41_TP_INDEXER=1
export VLLM_DSV41_TP_INDEXER_MIN_TOKENS=0 VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS=0
export VLLM_DSV41_FUSED_SEAM_QUANT=1 VLLM_DSV41_FUSED_Q_ROPE=1
export VLLM_DSV41_DSPARK_SHARDED_CTX_KV=1 TPU_DEEPSEEK_V41_ENGRAM_DEBUG_VERIFY=0
export VLLM_SPLIT_TOKENIZE_MIN_CHARS=65536

# Disable SparseCore collective offload on v7x for deterministic MoE / MLA execution
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true --xla_tpu_enable_sparse_core_compute_offload=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_offloading_gather_to_sparsecore=false --xla_tpu_offload_all_supported_gathers_to_sparsecore=false --xla_tpu_enable_offloading_scatter_to_sparsecore=false --xla_tpu_enable_offloading_sort_to_sparsecore=false --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_offloading_reshape_to_sparsecore=false --xla_tpu_enable_offloading_reduce_to_sparsecore=false --xla_tpu_num_sparse_cores_for_gather_offloading=0 --xla_tpu_num_sparse_cores_for_scatter_offloading=0 --xla_tpu_num_sparse_cores_for_sort_offloading=0 --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false --xla_tpu_enable_sparse_core_async_all_to_all=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"

# Raiden Stage 3 Host-DRAM / TCP KV Transfer
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_DSV4_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_TRANSFER_PARALLELISM=1
export TPU_ENABLE_D2H_TRANSFER=1
export TPU_HMA_D2H_COPY_IMPL=device_put
export TPU_HMA_MIN_DONE_RECVING_BATCH=1
export TPU_HMA_MAX_DONE_RECVING_BATCH=1
export TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1
export TPU_HMA_MAX_INFLIGHT_PULLS=8
export TPU_MAX_HOST_KV_BUFFER_SIZE=64
export TPU_P2P_WAIT_PULL_TIMEOUT=300
export TPU_KV_SHM_POOL_GB=8
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_RAIDEN_PLAN_DUMP_DIR="${CACHE_ROOT}/plan_dump"
```

---

## 5. Exact Server Launch Commands

### 5.1 Prefill Instance (`1P` — `4` chips, `DP=8 / TP=1 / EP=8`, `(64, 8192)`, Prefix Caching Enabled)

```bash
export ROLE="prefill"
export ADVERTISE_HOST="<PREFILL_HOST_INTERNAL_IP>"
export VLLM_DSV41_CED=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29499
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:29500,localhost:29501,localhost:29502,localhost:29503,localhost:29504,localhost:29505,localhost:29506,localhost:29507"
export TORCH_TPU_DP_MASTER_PORT=29530
export TPU_RAIDEN_ADVERTISE_HOST="${ADVERTISE_HOST}"
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_KV_TRANSFER_PORT=9100
export TPU_KV_TRANSFER_NAMESPACE="prefill_dsv41_p0"
export TPU_RAIDEN_JOB_NAME="prefill"
export TPU_RAIDEN_ENGINE_ID="prefill-engine-dsv41-p0"

vllm serve /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port 8400 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --api-server-count 8 \
  --renderer-num-workers 4 \
  --prefill-schedule-interval 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-prompt-tokens-details \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --no-async-scheduling \
  --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":14579}' \
  --compilation-config '{"compile_sizes": [512, 2048, 8192]}'
```

### 5.2 Corrected Decode Instance (`D_i` — `4` chips each, `(48, 288)`, Compile Bucket `288`, `DSpark K=5`, `--async-scheduling`)

Run on each decode host `i in {0, 1, 2, 3}` (assigning unique `DECODE_IDX` per host):

```bash
export ROLE="decode"
export DECODE_IDX="<0|1|2|3>"
export ADVERTISE_HOST="<DECODE_HOST_INTERNAL_IP>"
export VLLM_DSV41_CED=0
export VLLM_SPEC_DECODE_TOKENS=5
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29509
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:29510,localhost:29511,localhost:29512,localhost:29513,localhost:29514,localhost:29515,localhost:29516,localhost:29517"
export TORCH_TPU_DP_MASTER_PORT=29520
export TPU_RAIDEN_ADVERTISE_HOST="${ADVERTISE_HOST}"
export TPU_RAIDEN_RESHARD_PORT_BASE=28000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
export TPU_KV_TRANSFER_PORT=9200
export TPU_KV_TRANSFER_NAMESPACE="decode_dsv41_d${DECODE_IDX}"
export TPU_RAIDEN_JOB_NAME="decode"
export TPU_RAIDEN_ENGINE_ID="decode-engine-dsv41-d${DECODE_IDX}"

vllm serve /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port 9400 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --api-server-count 8 \
  --renderer-num-workers 4 \
  --prefill-schedule-interval 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-seqs 48 \
  --max-num-batched-tokens 288 \
  --gpu-memory-utilization 0.85 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-prompt-tokens-details \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --async-scheduling \
  --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":14579}' \
  --compilation-config '{"compile_sizes": [16, 40, 48, 64, 96, 192, 288]}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'
```

### 5.3 Disaggregated Proxy Router (`toy_proxy_server.py` — Least-In-Flight DP Routing)

```bash
# Example for 1P2D (pass 1, 2, 3, or 4 decoder IPs/ports for 1P1D, 1P2D, 1P3D, or 1P4D):
python3 -u /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts "${PREFILL_HOST_IP}" \
  --prefiller-ports 8400 \
  --prefiller-dp-size 8 \
  --decoder-hosts "${DECODE_HOST_0_IP}" "${DECODE_HOST_1_IP}" \
  --decoder-ports 9400 9400 \
  --decoder-dp-size 8
```

---

## 6. Post-Startup Verification Checklist (Mandatory Before Benchmarking)

1. **Endpoint Health**: Confirm `HTTP 200` on `http://<PREFILL_IP>:8400/health`, every `http://<DECODE_IP>:9400/health`, and `http://<PROXY_IP>:8000/healthcheck`.
2. **Decode Flag Audit**: Inspect running `vllm serve` arguments on every decode node and verify:
   - `--max-num-seqs 48 --max-num-batched-tokens 288`
   - `--compilation-config {"compile_sizes": [16, 40, 48, 64, 96, 192, 288]}`
   - `--speculative-config {"method": "dspark", "num_speculative_tokens": 5}`
   - `--async-scheduling`
3. **`DSpark` Acceptance Sanity Check**: Scrape `http://<DECODE_IP>:9400/metrics` (`vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_drafts_total`) during warmup or first sweep point and confirm `mean_accept_len_tau = 1 + accepted / drafts >= 3.0` (`~3.1–3.2` on the empirical `50:20:12:9:9` tool-calling workload; `tau == 1.00` indicates broken speculative verification).
4. **Benchmark Evaluation Gates**:
   - **`GATE-COMPLIANT`**: `P90_low` per-stream decode speed (`10th` percentile across all completed requests) `>= 50.0 tok/s/stream` **AND** `TTFT P95` (across all completed requests) `<= 3.0 s`.
   - **`MAX-THROUGHPUT`**: Highest steady-state window output throughput (`window_output_tok_per_s` and `window_output_tok_per_s / total_physical_chips`).

See [`manifests/deepseek_v41_repro_manifest.yaml`](manifests/deepseek_v41_repro_manifest.yaml) for the sanitized Kubernetes Pod template.
