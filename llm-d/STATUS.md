# DeepSeek-V4.1-Flash-1017 TPU `v7x` (`llm-d` + `EP=16` vs `EP=8`) Autonomous Benchmark — Live Status

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **Hardware:** `ank-tpu-v7x-8-pool-a` (`2x2x2` = `8` physical chips / `16` TensorCores) + `ank-tpu-v7x-8-pool-b` (`2x2x2` = `8` physical chips / `16` TensorCores) = **`16` physical `tpu7x` chips**
- **Engine Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` (`vllm-torchtpu` `ebe4298eb`, `vllm` `77a1dc7f9`)
- **Verified Precision (Rule 1):** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`)
- **GCS Artifact Root:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/` (mirrored status at `gs://YOUR_GCS_BUCKET/atlas/v7x16-llmd-20261106/STATUS.md`)
- **GitHub Subfolder:** `llm-d/` in `ankdeshm/deepseek-v4.1-flash-1017-tpu-vllm-inference`

---

## Timeline & Step Log

| Timestamp (UTC) | Step | Status | Summary / Measured Result | Next Action |
|---|---|---|---|---|
| `2026-11-06T22:50Z` | **STEP 0: Pre-Flight** | **PASS (7/7)** | Both `2x2x2` pools Ready (`16` chips); `LWS v0.7.0` Ready; `[VERIFY]` independent 4-chip single-host split on `2x2x2` = **YES** (`3.41–3.53 s`); cross-pool `TPURaidenConnector` Stage-3 KV reshard = `132.3–134.5 ms` (`0` errors); `FP8` attn + `MXFP4` MoE + `FP8` KV verified; `1P1D` smoke test `10/10` HTTP 200, `DSpark tau = 2.58` (thinking) / `2.26` (`thinking=False`); warm compile caches (`7.58 GiB`) archived to GCS. | Launch Step 1 (`R0`) on Pool A + Step 2 (`llm-d` + Pool B `EP=16` Decode) in parallel. |
| `2026-11-06T23:06Z` | **STEP 1: `R0` Reproducibility Anchor** | **IN PROGRESS** | `r0-prefill-poola-h0` (`rqkr`) + `r0-decode-poola-h1` (`xq6s`, `48/288`, `DSpark K=5`, `--async-scheduling`) running `3/3` on Pool A; restoring warm compile caches and executing `n=128` (`40` trunks, `~95%` hit, `300 s` steady window, `PASS = 6,700–8,200` out tok/s). | Record `R0` metrics, switch Pool A to warm-cached `llm-d` Prefill, run Step 2e `64`-token greedy correctness gate. |
| `2026-11-06T23:14Z` | **STEP 2: `llm-d` Bring-Up + Pool B `EP=16` LWS Decode** | **IN PROGRESS** | `InferencePool` (`pd-disaggregation`) + `EndpointPicker` (`pd-disaggregation-epp`, `llm-d v1.0.0`, `2/2 Running`) live on `default-pool`; Pool B `2x2x2` `EP=16` `LeaderWorkerSet` (`059c` + `cz6l`) booting with `llm-d-router-disagg-sidecar:v1.0.0` + `TPURaidenConnector`. | Complete `EP=16` Decode boot, run `64`-token greedy correctness gate (`>=16` prompts), then run `R1a -> R1b -> R1 Snapshot -> R2 -> R3 -> R4 -> R5 -> R6 -> Final Report`. |
