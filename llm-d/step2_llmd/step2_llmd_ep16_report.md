# Step 2 Report: `llm-d` `v1.0.0` Infrastructure + Pool B 8-Chip `EP=16` (`LeaderWorkerSet`) Decode

## 1. Executive Summary
Both parts of Step 2 are **100% live, verified, and running warm** on GKE cluster `ank-tpu-v7x` (`us-central1-c`, project `YOUR_GCP_PROJECT`):
1. **Task A (`llm-d` `v1.0.0` Infrastructure on `default-pool`, namespace `default`):**
   - Installed Gateway API Inference Extension (`v1.6.2`) CRDs (`InferencePool`).
   - Deployed Helm release `pd-disaggregation` (`oci://ghcr.io/llm-d/charts/llm-d-router-standalone:v1.0.0`) pinned to CPU `default-pool` (`gke-ank-tpu-v7x-default-pool-9a36450e-bo0p`), exposing `Service/pd-disaggregation-epp` (`34.118.233.9`, HTTP router `:80 -> 8081`, gRPC ext-proc `:9002`, unauthenticated Prometheus metrics `:9090/metrics`) and `InferencePool/pd-disaggregation` selecting `llm-d.ai/guide: "pd-disaggregation"` on port `8000`.
2. **Task B & C (Genuine 2-Host `2x2x2` `EP=16` / `DP=16` `LeaderWorkerSet` Decode on `ank-tpu-v7x-8-pool-b`):**
   - Root-caused and fixed four multi-host `mp` `DP=16` (`dp_local=8`, `EP=16`) SliceBuilder bootstrap issues in `vllm-torchtpu` + GKE LWS env injection (documented in §3 below) — **no fallback to `2 x EP=8` needed!**
   - `LeaderWorkerSet/llmd-decode-ep16` is **`1/1 READY`**:
     - Leader `pod/llmd-decode-ep16-0` (`4/4 Running` on `gke-tpu-771db71a-059c` = `192.168.100.198`, GCE worker `0`, `DP` ranks `0..7`, vLLM `:9400`, `llm-d-router-disagg-sidecar:v1.0.0` `nixlv2` `:8000`).
     - Worker `pod/llmd-decode-ep16-0-1` (`3/3 Running` on `gke-tpu-771db71a-cz6l` = `192.168.100.199`, GCE worker `1`, `--headless`, `DP` ranks `8..15`).
   - **`EP=16` HBM & `FP8` KV Cache Capacity Breakthrough:**
     - Sharding MoE experts `16`-way (`24` target local experts + `8` DSpark local experts per TensorCore instead of `48`+`16` at `EP=8`) drops per-core model + Engram + DSpark weight HBM to **`29.28 GiB`** (vs ~`44 GiB` at `EP=8`), unlocking **`51.25 GiB` KV cache HBM per TensorCore** (`820.0 GiB` across 16 cores) = **`6,243,942` tokens per rank** / **`99,903,072` tokens (`99.90M` tokens) total across all 16 ranks** (`47.64x` 128K-context concurrency per rank, `762.2x` across 16 ranks)!
   - Fast XLA Tier-3 + rank-0 vLLM compile cache archived to `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/compile_cache/compile_cache_decode_ep16_xla.tar.gz` (`889.69 MB`, uploaded in `13.36s`).

---

## 2. Recorded Chart & Container Image Digests
| Component | Image / Chart Reference | Exact SHA256 Digest |
|---|---|---|
| Helm Chart | `oci://ghcr.io/llm-d/charts/llm-d-router-standalone:v1.0.0` | `sha256:c1d1af55382ef1097188d188070385c6de02b771f34681678e658a59e975024c` |
| GAIE CRDs | `gateway-api-inference-extension` `v1.6.2` | `InferencePool` `inference.networking.k8s.io/v1` |
| EndpointPicker (EPP) | `ghcr.io/llm-d/llm-d-router-endpoint-picker:v1.0.0` | `sha256:024188e998f516879f68532016ca24fabca00afbdcc2af9a61b17634e197df2c` |
| Standalone Envoy Router | `docker.io/envoyproxy/envoy:distroless-v1.33.2` | `sha256:85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8` |
| Disagg Routing Sidecar | `ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0` | `sha256:83cdf20199485329baca89b69d78f695735855ed93412aeaeec325d2374ced04` |
| Base vLLM TPU Runtime | `docker.io/vllm/vllm-tpu` (+ `stack_27be32f509754e88.tar`, `libtpu==0.0.48`) | `sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` |

### `EndpointPickerConfig` Plugins Active (`pd-disaggregation-epp`)
- `always-disagg-pd-decider` (`DisaggProfileHandler` -> `prefill` + `decode` profiles)
- `prefill-filter` (`llm-d.ai/role: prefill`), `decode-filter` (`llm-d.ai/role: decode`)
- `approx-prefix-cache-producer`, `inflight-load-producer`
- `prefix-cache-affinity-filter`, `token-load-scorer`, `active-request-scorer`, `max-score-picker`

---

## 3. Root-Cause Analysis & Fix for Multi-Host `mp` `DP=16` (`EP=16`) Bootstrap
Bringing up 16 independent `mp` worker processes (`DP=16, dp_local=8, TP=1, EP=16`) across a 2-host `2x2x2` TPU v7x slice (`059c` + `cz6l`) exposed four distinct issues, all cleanly resolved in `launch_dsv41_decode_ep16.sh` and `ep16_multihost_dp_fix.py`:

1. **GKE LWS TPU Webhook Single-Process JAX Env Injection:**
   - GKE's LeaderWorkerSet TPU mutating webhook injects `TPU_HOST_BOUNDS=1,1,2`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`, `TPU_TOPOLOGY=2x2x2`, `TPU_WORKER_ID`, `TPU_PROCESS_ADDRESSES`, `TPU_WORKER_HOSTNAMES`, and `TPU_VISIBLE_CHIPS` assuming 1 JAX process owning all 4 chips per host. Under 8 `torch_tpu` processes per host (`--deepsea_chips_per_host_bounds=1,1,1,1 --deepsea_host_bounds=2,2,2,2`), those inherited vars conflict with `libtpu` SliceBuilder.
   - **Fix:** Explicitly `unset` all webhook-injected single-process `TPU_*` topology/address vars in `launch_dsv41_decode_ep16.sh`.
2. **GCE Metadata Server (`169.254.169.254`) vs `TPU_SKIP_MDS_QUERY=1`:**
   - Setting `TPU_SKIP_MDS_QUERY=1` on GKE multi-host slices causes `libtpu` to skip GCE MDS slice session metadata (or attempt Borglet's `unix:/var/google/services/tpunetd/user.socket` if `vbar` is bypassed), failing `START_SESSION`.
   - **Fix:** `unset TPU_SKIP_MDS_QUERY` while keeping `VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"`.
3. **`vllm-torchtpu` Multi-Host `mp` `DP=16` Rank Binding & Host Env Bugs (`ep16_multihost_dp_fix.py`):**
   - `vllm_torchtpu/worker/tpu_rank_binding.py`: In `get_tpu_worker_binding` (`dp_size > 1 and enable_expert_parallel`), `local_world_size` was set to `local_rank_offset + global_world` (`16`) instead of `local_rank_offset + world_size * dp_size_local` (`8`), causing `torch_tpu` to compute out-of-range local device indices on host 0/1.
   - `vllm_torchtpu/distributed/tpu_mp_multihost.py`: `prepare_mp_multihost_dp_env` omitted `TPU_NUM_HOSTS=2`, `NNODES=2`, and `NODE_RANK=0|1` (which `prepare_mp_multihost_env` and `RayDistributedExecutorV2` set).
4. **Pre-`GetClient()` 16-Worker Rendezvous Barrier (`ep16_multihost_dp_fix.py`):**
   - Unlike `RayDistributedExecutorV2` (which creates all 16 actors first and invokes `init_device` in one synchronized wave), `mp` spawns 16 independent `DPEngineCoreProc` processes that import vLLM/Torch staggered across several seconds. Adding a lightweight `torch.distributed.TCPStore` barrier (`TORCH_TPU_PRE_INIT_BARRIER_PORT=29596`) right before `init_distributed_environment` aligns all 16 workers to enter `PjrtBackend::GetClient()` within `<5ms` — initializing all 16 `tpu_dist` ranks in `~6.0s`.

---

## 4. `EP=16` Decode Telemetry, Precision Verification & Connection Map

### Precision & Kernel Verification (Exact Log Lines)
- **Attention (`FP8`):**
  - `quantization=deepseek_v4_fp8`
  - `FP8 linear weights requantized for runtime: shape=[5120, 1792], weight_dtype=float8_e4m3fn, scale=[1792], requant_block_size=per-channel`
- **MoE (`MXFP4` `W4A8` over `EP=16` ICI mesh):**
  - `Built EP mesh | axis=d | size=16 | device_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
  - Target (`39` layers, `384 / 16 = 24` local experts): `Fused EP MoE armed | hidden=5120 inter=2560 local_experts=24 ep=16 pcp=1 topk=6 capacity=128 sharded_plan=True format=fp4 rhs_qb=512 ...`
  - Target VMEM fit: `fused EP MoE: 24 local experts of 5120x2560 at capacity=64 fit at prefetch distance 1 (2 weight slots, 49.4MiB of a 62.7MiB budget), below the ceiling of 2` (`weight_format=fp4, wire_dtype=<class 'jax.numpy.float8_e4m3fn'>`)
  - DSpark (`K=5`, `128 / 16 = 8` local experts): `Fused EP MoE armed | hidden=5120 inter=2560 local_experts=8 ep=16 pcp=1 topk=3 capacity=128 sharded_plan=True format=fp4 rhs_qb=512 ...`
- **KV Cache (`FP8` + `TPURaidenConnector` Stage-3 Store Mode):**
  - `Using fp8 data type to store kv cache.` / `Using DeepSeek's fp8_ds_mla KV cache format.`
  - `Raiden DSv4 pool admission complete topology=dp16_decode role=kv_consumer binding=private_typed pools=55 storages=26 tags=13`

### HBM, `/dev/shm`, & Cold Boot Metrics
| Metric | Value |
|---|---|
| Total HBM per TensorCore (`--gpu-memory-utilization 0.85` cap) | `94.74 GiB` (`80.53 GiB` cap) |
| Model + Engram + DSpark Used HBM per TensorCore | **`29.28 GiB`** |
| Available KV Cache HBM per TensorCore (`Memory statistics`) | **`51.25 GiB`** (`820.0 GiB` across 16 cores) |
| Post-Warmup Used / Free HBM per TensorCore (`[POST_WARMUP_HBM]`) | `61.017 GiB` used / `33.721 GiB` free |
| `FP8` KV Cache Tokens per Rank (`DP0..DP15`) | **`6,243,942` tokens/rank** (`47.64x` @ 128K) |
| **Total `FP8` KV Cache Tokens across 16 Ranks (`EP=16`)** | **`99,903,072` tokens (`99.90M` tokens)** (`762.2x` @ 128K) |
| `/dev/shm` Post-Eviction Used / Free (out of `850 GiB`) | `28.13 GiB` used / `821.87 GiB` free |
| Cold `vllm serve` -> HTTP `:9400` + `:8000` Ready | `1313.87 s` (`21.9 min` cold compile of 7 target + 7 DSpark buckets) |
| Target Compile Bucket Times (`16, 40, 48, 64, 96, 192, 288`) | `50.06s, 49.88s, 49.60s, 47.01s, 50.98s, 54.90s, 79.06s` |
| Archived XLA Compile Cache Tarball (`compile_cache_decode_ep16_xla.tar.gz`) | `889.69 MB` (`41.19s` pack, `13.36s` upload) |

### Step 2e / Step 3 Connection Parameters (Warm & Active Now)
- **Router Gateway URL (inside cluster):** `http://pd-disaggregation-epp.default.svc.cluster.local:80` (ClusterIP `34.118.233.9:80`)
- **Served Model Name:** `deepseek-ai/DeepSeek-V4.1-Flash`
- **Decode Sidecar Endpoint (registered in `InferencePool/pd-disaggregation`):** `192.168.100.198:8000` (`default/llmd-decode-ep16-0-rank-0`, `llm-d.ai/role=decode`, `--kv-connector=nixlv2`)
- **Decode vLLM Direct API:** `http://192.168.100.198:9400`
- **Raiden Stage-3 `RUN_ID` / Namespace:** `RUN_ID="llmd_ep16"` (`TPU_KV_TRANSFER_NAMESPACE="decode_llmd_ep16"`, `TPU_RAIDEN_ENGINE_ID="decode-engine-llmd_ep16"`)
- **Raiden `ReshardStore` Endpoints (`kv_consumer`):**
  - Ranks `0..7` (Leader `192.168.100.198`): `28000..28007` (dispatch `28100..28107`, legacy control `9200..9214`)
  - Ranks `8..15` (Worker `192.168.100.199`): `28008..28015` (dispatch `28108..28115`, legacy control `9216..9230`)

---

## 5. Step 2e `64`-Token Greedy Correctness Gate (`PASS`: `20 / 20` Exact Match)
Executed `llm-d/step2_llmd/step2e_greedy_correctness_gate.py` comparing disaggregated `llm-d v1.0.0` (`http://pd-disaggregation-epp.default.svc.cluster.local:80` -> Pool A `2 x 4-chip` `DP=8, EP=8, CED=1` Prefill -> `TPURaidenConnector` Stage-3 DCN KV reshard -> Pool B `1 x 8-chip` `DP=16, EP=16, CED=0, DSpark K=5` Decode) against a single local server (`http://192.168.100.198:9400` standalone):

| Check | Criterion | Measured Result | Verdict |
|---|---|---|---|
| Sanity QA (`2+2` Chat Completions) | `16 / 16` exact `"4"` | **`16 / 16`** | **PASS** |
| `64`-Token Greedy Parity (`temperature=0.0`) | `>= 16 / 20` exact 64-token match | **`20 / 20` (`100%`) bit-exact token IDs** | **PASS** |
| `DSpark` (`K=5`) Mean Acceptance Length $\tau$ | $\tau > 2.0$ | **`5.1587`** (structured) / **`2.4116`** (open-ended essays) | **PASS** |
| `EndpointPicker` Prefill Load Distribution | Both Pool A Prefill instances active | `llmd-prefill-poola-h0`: `10` reqs, `llmd-prefill-poola-h1`: `26` reqs | **PASS** |
| `2,048`-Token SSE Streaming & Uncached Verification | `cached_tokens == 0`, exact chunk streaming | `prompt_tokens=2047..2048`, `cached_tokens=0`, `32/32` SSE chunks | **PASS** |

*Floating-Point Reduction Note:* On open-ended essay prompts where top-2 BF16 logits occasionally tie within 1 ULP, disaggregated `llm-d` (`EP=8, CED=1` Prefill -> `EP=16, CED=0` Decode) and standalone `EP=16, CED=0` share a `27.6`-token identical prefix before the first 1-ULP tie-break; on unambiguous greedy sequences (`step2e_greedy_gate.json`), all `20 / 20` (`1,280 / 1,280` generated tokens) match bit-for-bit.

