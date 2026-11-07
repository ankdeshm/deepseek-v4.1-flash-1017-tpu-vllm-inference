# DeepSeek-V4.1-Flash-1017 TPU `v7x` (`llm-d` + `EP=16` vs `EP=8`) Autonomous Benchmark — Live Status

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **Hardware:** `ank-tpu-v7x-8-pool-a` (`2x2x2` = `8` physical chips / `16` TensorCores) + `ank-tpu-v7x-8-pool-b` (`2x2x2` = `8` physical chips / `16` TensorCores) = **`16` physical `tpu7x` chips**
- **Engine Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` (`vllm-torchtpu` `ebe4298eb`, `vllm` `77a1dc7f9`, `libtpu==0.0.48`, `PYTHONHASHSEED=0`)
- **Verified Precision (Rule 1):** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`)
- **GCS Artifact Root:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/` (mirrored status at `gs://YOUR_GCS_BUCKET/atlas/v7x16-llmd-20261106/STATUS.md`)
- **GitHub Subfolder:** `llm-d/` in `ankdeshm/deepseek-v4.1-flash-1017-tpu-vllm-inference`
- **Last Updated (UTC):** `2026-11-07T01:43:51Z`

---

## Timeline & Step Log

| Timestamp (UTC) | Step | Status | Summary / Measured Result | Next Action |
|---|---|---|---|---|
| `2026-11-06T22:50Z` | **STEP 0: Pre-Flight** | **PASS (7/7)** | Both `2x2x2` pools Ready (`16` chips); `LWS v0.7.0` Ready; `[VERIFY]` independent 4-chip single-host split on `2x2x2` = **YES** (`3.41–3.53 s`); cross-pool `TPURaidenConnector` Stage-3 KV reshard = `132.3–134.5 ms` (`0` errors); `FP8` attn + `MXFP4` MoE + `FP8` KV verified; `1P1D` smoke test `10/10` HTTP 200, `DSpark tau = 2.58` (thinking) / `2.26` (`thinking=False`); warm compile caches (`7.58 GiB`) archived to GCS. | Launch Step 1 (`R0`) on Pool A + Step 2 (`llm-d` + Pool B `EP=16` Decode) in parallel. |
| `2026-11-07T00:10Z` | **STEP 1: `R0` Reproducibility Anchor (`n=128`, `300 s`)** | **PASS (`7,423.72` out tok/s)** | **PASS GATE (`6,700–8,200` out tok/s, `tau > 2.0`):** Measured **`7,423.72` out tok/s** (`+0.20%` vs prior `A1-b` `7,408.59`; `-0.30%` vs `7,446` anchor), **`1,855.93` out tok/s per decode chip (`/4`)**, **`927.97` out tok/s per total chip (`/8`)**, **`283,403.02` total (`in+out`) tok/s**, **`DSpark tau = 2.8755`**, per-user `Med / P90_low / P95_low = 67.83 / 54.65 / 47.43 tok/s` (in-window), `TTFT P50 / P95 / P99 = 1.205 / 1.559 / 8.731 s` (in-window) & `1.205 / 1.556 / 8.726 s` (all reqs), realized cache hit **`95.47%`**, `0` preemptions, **`0 / 4,631` failures**. | Deploy `2 x 4-chip` `llm-d` Prefill on Pool A (`llmd-prefill-poola-h{0,1}`). |
| `2026-11-07T01:15Z` | **STEP 2: `llm-d` v1.0.0 Path A + Pool A `2x4c` Prefill + Pool B `1x8c` `EP=16` LWS Decode + Correctness Gate** | **PASS (`20/20` Greedy Gate)** | **Path A (`llm-d v1.0.0` + GAIE `v1.6.2` `InferencePool` + `EndpointPicker` + `llm-d-router-disagg-sidecar:v1.0.0` `nixlv2` -> `TPURaidenConnector` Stage-3):** Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `1 x 8-chip` `LeaderWorkerSet` `DP=16, EP=16` Decode (`llmd-decode-ep16`, **`29.28 GiB` weights/core, `51.25 GiB` KV HBM/core = `99.90M` FP8 KV tokens across 8 chips**). **Step 2e Correctness Gate PASS:** `16/16` QA sanity + **`20/20` (`100%`) bit-exact 64-token greedy parity** vs single local server, `DSpark tau = 5.16` (structured) / `2.41` (open-ended), active EPP routing across `h0` & `h1`. | Run Step 3 `R1a` (`S1 = 2048:1024`, `EP=16` Decode, `n = 32..1024`, `300 s`/level). |
| `2026-11-07T01:43Z` | **STEP 3 `R1a`: `S1 = 2048:1024` (`P`=`2x4c` Pool A, `D`=`1x8c` `EP=16` Pool B)** | **IN PROGRESS (`c=32, 64, 128` done)** | Completed `c = 32, 64, 128`; latest `c=128` = **`11,800.75` out tok/s** (`1,475.09`/dec-chip, `tau=4.1321`, `P90_low=87.57` tok/s, `0` err). Continuing ladder... | Continue `R1a` concurrency escalation until a stop rule triggers. |


---

## Step 1 (`R0`) Complete Rule 4 Metrics Summary

- **Precision Verified:** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`).
- **Workload:** Empirical `50:20:12:9:9` ISL distribution (`mean ISL = 17,916.9` tok, `mean OSL = 488.8` tok), `40` pinned prefix trunks, `95%` target cache hit (`95.47%` realized), natural stop (`--no-ignore-eos`).

| Run | Topology & Hosts | EP per Decode Instance | Concurrency (`n`) | Steady Window | Total Out tok/s | Out tok/s / Decode Chip (`/4`) | Out tok/s / Total Chip (`/8`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | `DSpark` $\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Preemptions (`P / D`) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---:|
| **`R0` (Measured)** | `1P1D` Pool A (`P`=`rqkr` `4c`, `D`=`xq6s` `4c`) | `EP=8` (`DP=8`) | `128` | `300.0 s` | **`7,423.72`** | **`1,855.93`** | **`927.97`** | `283,403.02` | `67.83 / 54.65 / 47.43` \| `67.74 / 54.44 / 47.37` | `1.205 / 1.556 / 8.726` | `1.205 / 1.559 / 8.731` | **`2.8755`** | `13.61 / 14.0 / 16.0` (`108.91` total) | `D`: `9.02% / 13.57%` · `P`: `0.82% / 7.46%` | `0 / 0` | **`0 / 4,631`** |
| **Prior `A1-b` Ref** | `1P1D` (`4c` P + `4c` D) | `EP=8` (`DP=8`) | `128` | `180.0 s` | `7,408.59` | `1,852.15` | `926.07` | `287,099.52` | `66.64 / 54.32 / 47.78` \| `66.62 / 53.91 / 47.74` | `1.205 / 1.613 / 3.491` | `1.201 / — / 3.499` | `2.8699` | — | — | `0 / 0` | `0 / 2,828` |

<!-- R1A_RESULTS_START -->
---

## Step 3 (`R1a`) Results — `S1 = 2048 : 1024` (`P` = Pool A `2 x 4c` `EP=8`, `D` = Pool B `1 x 8c` `EP=16` LWS)

- **Topology:** `llm-d` `v1.0.0` (`InferencePool/pd-disaggregation` + `EndpointPicker` + standalone Envoy router) -> Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `1 x 8-chip` `LeaderWorkerSet` `DP=16, EP=16` Decode (`llmd-decode-ep16`, `2` hosts x `4` `tpu7x` chips).
- **Caps & Speculative Decoding:** `max_num_seqs=48`, `max_num_batched_tokens=288` per DP rank (`compile_sizes=[16, 40, 48, 64, 96, 192, 288]`), `DSpark K=5` (` speculative_token_tree='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]'`).
- **Workload (`S1`):** Calibrated exact `ISL = 2,048` tokens (`word_count = 2048` random single-token words per request, `0.00%` realized prefix cache hit, `/reset_prefix_cache` before every level), forced `OSL = 1,024` tokens (`min_tokens=1024, max_tokens=1024, ignore_eos=True, temperature=0.6`), `300.0 s` closed-loop steady window + clean drain per level.

| Level | Window | Steady Out tok/s (Gain) | Out tok/s / Dec Chip (`/8`) | Out tok/s / Tot Chip (`/16`) | Server Gen tok/s (`60-290s`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Prefill Split (`h0 / h1`) | Preempt (`P / D`) | Failures |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---|---|---:|---:|
| **`c=32`** | `300 s` | **`3,541.90`** *(base)* | **`442.74`** | **`221.37`** | `3,687.4` | `10,737.21` | `120.04 / 96.98 / 87.49` \| `120.12 / 97.01 / 87.52` | `0.391 / 0.894 / 5.056` | `0.391 / 0.898 / 5.056` | `8.33 / 11.43 / 14.69` | **`3.8725`** | `1.92 / 2.0 / 4.0` (`30.7` tot) | `D`: `0.268% / 0.643%` · `P`: `0.003% / 0.118%` | `44.5% / 55.5%` | `0 / 0` | **`0 / 1054`** |
| **`c=64`** | `300 s` | **`7,102.50`** (`+100.53%`) | **`887.81`** | **`443.91`** | `7,416.8` | `21,520.42` | `121.17 / 100.21 / 89.55` \| `121.27 / 100.43 / 89.91` | `0.421 / 1.097 / 3.764` | `0.422 / 1.119 / 3.765` | `8.25 / 11.12 / 13.96` | **`3.9901`** | `3.80 / 4.0 / 6.0` (`60.8` tot) | `D`: `0.529% / 1.194%` · `P`: `0.007% / 0.471%` | `48.6% / 51.4%` | `0 / 0` | **`0 / 2114`** |
| **`c=128`** | `300 s` | **`11,800.75`** (`+66.15%`) | **`1,475.09`** | **`737.55`** | `13,597.4` | `35,776.00` | `111.60 / 86.63 / 55.44` \| `111.81 / 87.57 / 58.93` | `0.501 / 1.270 / 6.416` | `0.501 / 1.300 / 6.433` | `8.94 / 16.97 / 39.94` | **`4.1321`** | `7.61 / 8.0 / 10.0` (`121.8` tot) | `D`: `1.063% / 1.509%` · `P`: `0.019% / 0.236%` | `50.1% / 49.9%` | `0 / 0` | **`0 / 3526`** |
<!-- R1A_RESULTS_END -->
