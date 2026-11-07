# DeepSeek-V4.1-Flash-1017 TPU `v7x` (`llm-d` + `EP=16` vs `EP=8`) Autonomous Benchmark — Live Status

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **Hardware:** `ank-tpu-v7x-8-pool-a` (`2x2x2` = `8` physical chips / `16` TensorCores) + `ank-tpu-v7x-8-pool-b` (`2x2x2` = `8` physical chips / `16` TensorCores) = **`16` physical `tpu7x` chips**
- **Engine Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` (`vllm-torchtpu` `ebe4298eb`, `vllm` `77a1dc7f9`, `libtpu==0.0.48`, `PYTHONHASHSEED=0`)
- **Verified Precision (Rule 1):** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`)
- **GCS Artifact Root:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/` (mirrored status at `gs://YOUR_GCS_BUCKET/atlas/v7x16-llmd-20261106/STATUS.md`)
- **GitHub Subfolder:** `llm-d/` in `ankdeshm/deepseek-v4.1-flash-1017-tpu-vllm-inference`
- **Last Updated (UTC):** `2026-11-07T01:17:00Z`

---

## Timeline & Step Log

| Timestamp (UTC) | Step | Status | Summary / Measured Result | Next Action |
|---|---|---|---|---|
| `2026-11-06T22:50Z` | **STEP 0: Pre-Flight** | **PASS (7/7)** | Both `2x2x2` pools Ready (`16` chips); `LWS v0.7.0` Ready; `[VERIFY]` independent 4-chip single-host split on `2x2x2` = **YES** (`3.41–3.53 s`); cross-pool `TPURaidenConnector` Stage-3 KV reshard = `132.3–134.5 ms` (`0` errors); `FP8` attn + `MXFP4` MoE + `FP8` KV verified; `1P1D` smoke test `10/10` HTTP 200, `DSpark tau = 2.58` (thinking) / `2.26` (`thinking=False`); warm compile caches (`7.58 GiB`) archived to GCS. | Launch Step 1 (`R0`) on Pool A + Step 2 (`llm-d` + Pool B `EP=16` Decode) in parallel. |
| `2026-11-07T00:10Z` | **STEP 1: `R0` Reproducibility Anchor (`n=128`, `300 s`)** | **PASS (`7,423.72` out tok/s)** | **PASS GATE (`6,700–8,200` out tok/s, `tau > 2.0`):** Measured **`7,423.72` out tok/s** (`+0.20%` vs prior `A1-b` `7,408.59`; `-0.30%` vs `7,446` anchor), **`1,855.93` out tok/s per decode chip (`/4`)**, **`927.97` out tok/s per total chip (`/8`)**, **`283,403.02` total (`in+out`) tok/s**, **`DSpark tau = 2.8755`**, per-user `Med / P90_low / P95_low = 67.83 / 54.65 / 47.43 tok/s` (in-window), `TTFT P50 / P95 / P99 = 1.205 / 1.559 / 8.731 s` (in-window) & `1.205 / 1.556 / 8.726 s` (all reqs), realized cache hit **`95.47%`**, `0` preemptions, **`0 / 4,631` failures**. | Deploy `2 x 4-chip` `llm-d` Prefill on Pool A (`llmd-prefill-poola-h{0,1}`). |
| `2026-11-07T01:15Z` | **STEP 2: `llm-d` v1.0.0 Path A + Pool A `2x4c` Prefill + Pool B `1x8c` `EP=16` LWS Decode + Correctness Gate** | **PASS (`20/20` Greedy Gate)** | **Path A (`llm-d v1.0.0` + GAIE `v1.6.2` `InferencePool` + `EndpointPicker` + `llm-d-router-disagg-sidecar:v1.0.0` `nixlv2` -> `TPURaidenConnector` Stage-3):** Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `1 x 8-chip` `LeaderWorkerSet` `DP=16, EP=16` Decode (`llmd-decode-ep16`, **`29.28 GiB` weights/core, `51.25 GiB` KV HBM/core = `99.90M` FP8 KV tokens across 8 chips**). **Step 2e Correctness Gate PASS:** `16/16` QA sanity + **`20/20` (`100%`) bit-exact 64-token greedy parity** vs single local server, `DSpark tau = 5.16` (structured) / `2.41` (open-ended), active EPP routing across `h0` & `h1`. | Run Step 3 `R1a` (`S1 = 2048:1024`, `EP=16` Decode, `n = 32..1024`, `300 s`/level). |
| `2026-11-07T01:17Z` | **STEP 3 `R1a`: `S1 = 2048:1024` (`P`=`2x4c` Pool A, `D`=`1x8c` `EP=16` Pool B)** | **IN PROGRESS** | Running concurrency ladder `32, 64, 128, 256, 384, 512, 768, 1024` (`>= 300 s` steady window per level, random uncached prompts `0%` prefix hit, `ignore_eos=True`, caps `48/288` per DP rank, `DSpark K=5`). | Save & publish metrics after each concurrency level, then run `R1b` (`2 x 4c` `EP=8` Decode). |


---

## Step 1 (`R0`) Complete Rule 4 Metrics Summary

- **Precision Verified:** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`).
- **Workload:** Empirical `50:20:12:9:9` ISL distribution (`mean ISL = 17,916.9` tok, `mean OSL = 488.8` tok), `40` pinned prefix trunks, `95%` target cache hit (`95.47%` realized), natural stop (`--no-ignore-eos`).

| Run | Topology & Hosts | EP per Decode Instance | Concurrency (`n`) | Steady Window | Total Out tok/s | Out tok/s / Decode Chip (`/4`) | Out tok/s / Total Chip (`/8`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | `DSpark` $\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Preemptions (`P / D`) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---:|
| **`R0` (Measured)** | `1P1D` Pool A (`P`=`rqkr` `4c`, `D`=`xq6s` `4c`) | `EP=8` (`DP=8`) | `128` | `300.0 s` | **`7,423.72`** | **`1,855.93`** | **`927.97`** | `283,403.02` | `67.83 / 54.65 / 47.43` \| `67.74 / 54.44 / 47.37` | `1.205 / 1.556 / 8.726` | `1.205 / 1.559 / 8.731` | **`2.8755`** | `13.61 / 14.0 / 16.0` (`108.91` total) | `D`: `9.02% / 13.57%` · `P`: `0.82% / 7.46%` | `0 / 0` | **`0 / 4,631`** |
| **Prior `A1-b` Ref** | `1P1D` (`4c` P + `4c` D) | `EP=8` (`DP=8`) | `128` | `180.0 s` | `7,408.59` | `1,852.15` | `926.07` | `287,099.52` | `66.64 / 54.32 / 47.78` \| `66.62 / 53.91 / 47.74` | `1.205 / 1.613 / 3.491` | `1.201 / — / 3.499` | `2.8699` | — | — | `0 / 0` | `0 / 2,828` |
