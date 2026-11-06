# STEP 0 Pre-Flight Verification Report — DeepSeek-V4.1-Flash-1017 on TPU v7x (target workload)

**Date (UTC):** 2026-11-06
**Cluster:** `gke_YOUR_GCP_PROJECT_us-central1_ank-tpu-v7x` (`v1.36.4-gke.2377000`, `us-central1-c`)
**GCS Artifact Prefix:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/step0/`

---

## 1. Summary Checklist

| # | Check | Status | Measured Result / Finding |
|---|---|---|---|
| **0.1** | **Pools & Node Topology** | **PASS** | 2 Ready `v7x-8` pools (`2x2x2` = 2 hosts $\times$ 4 chips = **8 chips / 16 TensorCores per pool**; **16 chips / 32 TensorCores total**; 224 vCPU, 944.8 GiB RAM, 8 VFIO devices / 4 physical chips per host, `94.74 GiB` HBM per TensorCore). |
| **0.2** | **LeaderWorkerSet Controller** | **PASS** | `leaderworkersets.leaderworkerset.x-k8s.io` (`v1`) installed; `deploy/lws-controller-manager` in `lws-system` `2/2` Ready (`registry.k8s.io/lws/lws:0.7.0`). |
| **0.3** | **`[VERIFY]` Concurrent Independent Single-Host 4-Chip Servers on `2x2x2` Slice** | **YES (PASS)** | Verified live on **both** `Pool A` and `Pool B`: Host 1 initialized an independent 4-chip (`8`-TensorCore, `2x2x1`) runtime (`3.53 s` on Pool A, `3.41 s` on Pool B, `psum=36.0`) via `unix:///var/run/tpu-plugin/vbar.sock` **while** Host 0 of the same `2x2x2` slice held an active `vLLM` `DP=8, EP=8` TPU session. Unlocks **R0**, **R4 (2$\times$4-chip P)**, and **R5 (1P3D)**. |
| **0.4** | **Cross-Pool Connectivity & KV Transfer (`Pool A` $\rightarrow$ `Pool B`)** | **PASS** | Bidirectional TCP echo succeeded across all 11 serving & Raiden ports (`8000, 8400, 9400, 14579, 9100, 9200, 9600, 27000, 27100, 28000, 28100`) with **0.242–0.903 ms RTT**. Live `TPURaidenConnector` Stage-3 KV transfers (`192.168.100.43` $\rightarrow$ `192.168.100.198`) completed with **3.55–4.55 ms** controller submit and **132.3–134.5 ms** end-to-end reshard latency, **0** errors. |
| **0.5** | **Checkpoint Staging & Load Time (`48` shards, `510.31 GB` / `475.26 GiB`)** | **PASS** | Parallel HTTP range stage GCS $\rightarrow$ `/dev/shm` tmpfs: **Pool A = `137.78 s` (`3.70 GB/s`)**; **Pool B = `161.32 s` (`3.16 GB/s`)** (vs single-stream GCS FUSE read `574.1 MB/s`). `vLLM` `EP=8` filtered weight load from `/dev/shm`: **Prefill = `54.16 s`**, **Decode = `48.49 s`** (drafter `3.56 s`), followed by `/dev/shm` shard eviction. |
| **0.6** | **Rule 1 Precision Check (`config.json` + Launch Logs)** | **PASS** | **Attention / Non-Expert Projections:** `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA attention core `BF16`) · **MoE Routed Experts (384, top-6):** `MXFP4` weights (`expert_dtype=fp4`, `rhs_qb=512`), `W4A8` (`wire_dtype=float8_e4m3fn`) fused EP GMM kernel · **KV Cache:** `FP8` (`--kv-cache-dtype fp8`). |
| **0.7** | **1P1D Smoke Test (Existing Stack, Cross-Pool, 10 Requests, `DSpark` $\tau > 2$)** | **PASS** | **10/10 HTTP 200** (`0` failures, `0` preemptions) on both thinking-enabled (`192` tok/req, `7.99 s`, **`DSpark` $\tau = 2.5771$**, `752` drafts / `1,186` accepted) and `thinking=False` (`1,215` tok in `3.04 s` = `399.94` out tok/s at `n=10`, **10/10 sane coherent outputs**; cumulative **`DSpark` $\tau = 2.2613$** across `1,466` drafts / `1,849` accepted). |
| **0.8** | **`[VERIFY]` Multi-Host `EP=16` (`DP=16`) Support Audit for Step 2** | **VERIFIED IN CODE** | `vllm_torchtpu/distributed/tpu_mp_multihost.py` (`prepare_mp_multihost_dp_env`) natively implements multi-host `DP=16, EP=16` over `mp` without Ray (`--data-parallel-size 16 --data-parallel-size-local 8` on LWS leader, `--headless --data-parallel-start-rank 8` on LWS worker, automatic `TCPStore` slicebuilder IP exchange & `TORCH_TPU_TOPOLOGY=2,2,2,2`). Will be live-validated in Step 2. |

---

## 2. Hardware & Cluster Inventory

| Pool | Node Pool Name | Host Role | Node Name | Internal IP | Accelerator & Topology Labels | Chips / TensorCores | Taints |
|---|---|---|---|---|---|---|---|
| **Pool A** | `ank-tpu-v7x-8-pool-a` | Host 0 | `gke-tpu-51ea19d7-rqkr` | `192.168.100.43` | `gke-tpu-accelerator=tpu7x`, `gke-tpu-topology=2x2x2` | 4 chips / 8 TensorCores | `google.com/tpu=present:NoSchedule`, `cloud.google.com/gke-queued=true:NoSchedule` |
| **Pool A** | `ank-tpu-v7x-8-pool-a` | Host 1 | `gke-tpu-51ea19d7-xq6s` | `192.168.100.38` | `gke-tpu-accelerator=tpu7x`, `gke-tpu-topology=2x2x2` | 4 chips / 8 TensorCores | `google.com/tpu=present:NoSchedule`, `cloud.google.com/gke-queued=true:NoSchedule` |
| **Pool B** | `ank-tpu-v7x-8-pool-b` | Host 0 | `gke-tpu-771db71a-059c` | `192.168.100.198` | `gke-tpu-accelerator=tpu7x`, `gke-tpu-topology=2x2x2` | 4 chips / 8 TensorCores | `google.com/tpu=present:NoSchedule`, `cloud.google.com/gke-queued=true:NoSchedule` |
| **Pool B** | `ank-tpu-v7x-8-pool-b` | Host 1 | `gke-tpu-771db71a-cz6l` | `192.168.100.199` | `gke-tpu-accelerator=tpu7x`, `gke-tpu-topology=2x2x2` | 4 chips / 8 TensorCores | `google.com/tpu=present:NoSchedule`, `cloud.google.com/gke-queued=true:NoSchedule` |

---

## 3. Software & Image Provenance

- **Pinned Base Container Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` (pinned by digest across all nodes to prevent `:nightly` tag drift).
- **Stack Overlay:** `gs://YOUR_GCS_BUCKET/dsv41-stack/stack_27be32f509754e88.tar` (`cpk/dsv41-combined` `vllm-torchtpu` tree + wheels).
- **`/workspace/vllm` Git SHA:** `7b77ed581d92fa14e34ab43344837bb1ca1b4257` (`vllm==0.29.0+empty`).
- **Runtime Package Versions:** `libtpu==0.0.48`, `torch_tpu==0.1.1.dev20260912075415`, `tpu-sync-torch==0.0.1.dev20260923040928`, `torch==2.13.0+cpu`, `compressed-tensors==0.15.0`.

---

## 4. `[VERIFY]` Single-Host 4-Chip Split on a `2x2x2` Slice (`YES` — Exact Recipe)

Both Kubernetes admission/scheduling and `libtpu` VBAR slice control allow running **two independent single-host 4-chip (`DP=8, EP=8`) servers simultaneously** on each `2x2x2` node pool:
1. **Kubernetes Pod Spec:**
   - Request `google.com/tpu: "4"` (`limits` and `requests`).
   - Include `cloud.google.com/gke-tpu-accelerator: tpu7x` and `cloud.google.com/gke-tpu-topology: 2x2x2` in `nodeSelector` (required by GKE Warden webhook `tpu-accelerator-topology-constraints`), plus `kubernetes.io/hostname` (or `cloud.google.com/gke-nodepool` + pod anti-affinity).
   - Tolerate **both** `google.com/tpu` (`Exists:NoSchedule`) and `cloud.google.com/gke-queued` (`Exists:NoSchedule`).
2. **TPU Runtime Environment (`2x2x1` isolation on a `2x2x2` GKE VM):**
   - Because GCE instance metadata on a `2x2x2` node advertises 2 hosts in `worker-network-endpoints`, single-host processes must override metadata discovery via **either**:
     - `vllm-torchtpu`'s `/workspace/mock_metadata.py` + `LD_PRELOAD=/workspace/metadata_hook.so` + `TPU_TOPOLOGY=2x2x1`, `TPU_HOST_BOUNDS=1,1,1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`, `TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:29500..29507` (used by `launch_dsv41_disagg_role.sh`), **or**
     - `TPU_SKIP_MDS_QUERY=1`, `TPU_WORKER_HOSTNAMES=localhost`, `TPU_WORKER_ID=0`, `TPU_TOPOLOGY=2x2x1`, `TPU_HOST_BOUNDS=1,1,1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`, `VBAR_CONTROL_SERVICE_URL=unix:///var/run/tpu-plugin/vbar.sock`.

---

## 5. Memory, KV Cache Capacity & Cold Compile Timings (`DP=8, EP=8`, `gpu_memory_utilization=0.85`)

| Role | Host | Total HBM / Core | Cap (`0.85`) | Used (Weights + Runtime) | Avail for KV / Core | KV Cache Tokens / DP Rank | Total KV Tokens (8 Ranks) | Cold Boot + XLA Compile |
|---|---|---|---|---|---|---|---|---|
| **Prefill (`CED=1`, `[512, 2048, 8192]`)** | `rqkr` (Pool A H0) | `94.74 GiB` | `80.53 GiB` | `46.27 GiB` | `34.25 GiB` | `1,464,212` (`11.17x` @ 131k) | `11,713,696` | `1,248.36 s` (`20.8 min`, cached to GCS) |
| **Decode (`CED=0`, `DSpark K=5`, `48/288`)** | `059c` (Pool B H0) | `94.74 GiB` | `80.53 GiB` | `47.86 GiB` | `32.67 GiB` | `3,980,492` (`30.37x` @ 131k) | `31,843,936` | `1,620.50 s` (`27.0 min`, cached to GCS) |
