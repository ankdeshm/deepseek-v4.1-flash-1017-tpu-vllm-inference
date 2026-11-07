# DeepSeek-V4.1-Flash-1017 TPU `v7x` (`llm-d` + `EP=16` vs `EP=8`) Autonomous Benchmark — Live Status

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **Hardware:** `ank-tpu-v7x-8-pool-a` (`2x2x2` = `8` physical chips / `16` TensorCores) + `ank-tpu-v7x-8-pool-b` (`2x2x2` = `8` physical chips / `16` TensorCores) = **`16` physical `tpu7x` chips**
- **Engine Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` (`vllm-torchtpu` `ebe4298eb`, `vllm` `77a1dc7f9`, `libtpu==0.0.48`, `PYTHONHASHSEED=0`)
- **Verified Precision (Rule 1):** Attention/non-expert `FP8` (`float8_e4m3fn`, `scale_fmt=ue8m0`, MLA core `BF16`), MoE routed experts `MXFP4` (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM), KV cache `FP8` (`--kv-cache-dtype fp8`)
- **GCS Artifact Root:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/` (mirrored status at `gs://YOUR_GCS_BUCKET/atlas/v7x16-llmd-20261106/STATUS.md`)
- **GitHub Subfolder:** `llm-d/` in `ankdeshm/deepseek-v4.1-flash-1017-tpu-vllm-inference`
- **Last Updated (UTC):** `2026-11-07T03:11:57Z`

---

## Timeline & Step Log

| Timestamp (UTC) | Step | Status | Summary / Measured Result | Next Action |
|---|---|---|---|---|
| `2026-11-06T22:50Z` | **STEP 0: Pre-Flight** | **PASS (7/7)** | Both `2x2x2` pools Ready (`16` chips); `LWS v0.7.0` Ready; `[VERIFY]` independent 4-chip single-host split on `2x2x2` = **YES** (`3.41–3.53 s`); cross-pool `TPURaidenConnector` Stage-3 KV reshard = `132.3–134.5 ms` (`0` errors); `FP8` attn + `MXFP4` MoE + `FP8` KV verified; `1P1D` smoke test `10/10` HTTP 200, `DSpark tau = 2.58` (thinking) / `2.26` (`thinking=False`); warm compile caches (`7.58 GiB`) archived to GCS. | Launch Step 1 (`R0`) on Pool A + Step 2 (`llm-d` + Pool B `EP=16` Decode) in parallel. |
| `2026-11-07T00:10Z` | **STEP 1: `R0` Reproducibility Anchor (`n=128`, `300 s`)** | **PASS (`7,423.72` out tok/s)** | **PASS GATE (`6,700–8,200` out tok/s, `tau > 2.0`):** Measured **`7,423.72` out tok/s** (`+0.20%` vs prior `A1-b` `7,408.59`; `-0.30%` vs `7,446` anchor), **`1,855.93` out tok/s per decode chip (`/4`)**, **`927.97` out tok/s per total chip (`/8`)**, **`283,403.02` total (`in+out`) tok/s**, **`DSpark tau = 2.8755`**, per-user `Med / P90_low / P95_low = 67.83 / 54.65 / 47.43 tok/s` (in-window), `TTFT P50 / P95 / P99 = 1.205 / 1.559 / 8.731 s` (in-window) & `1.205 / 1.556 / 8.726 s` (all reqs), realized cache hit **`95.47%`**, `0` preemptions, **`0 / 4,631` failures**. | Deploy `2 x 4-chip` `llm-d` Prefill on Pool A (`llmd-prefill-poola-h{0,1}`). |
| `2026-11-07T01:15Z` | **STEP 2: `llm-d` v1.0.0 Path A + Pool A `2x4c` Prefill + Pool B `1x8c` `EP=16` LWS Decode + Correctness Gate** | **PASS (`20/20` Greedy Gate)** | **Path A (`llm-d v1.0.0` + GAIE `v1.6.2` `InferencePool` + `EndpointPicker` + `llm-d-router-disagg-sidecar:v1.0.0` `nixlv2` -> `TPURaidenConnector` Stage-3):** Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `1 x 8-chip` `LeaderWorkerSet` `DP=16, EP=16` Decode (`llmd-decode-ep16`, **`29.28 GiB` weights/core, `51.25 GiB` KV HBM/core = `99.90M` FP8 KV tokens across 8 chips**). **Step 2e Correctness Gate PASS:** `16/16` QA sanity + **`20/20` (`100%`) bit-exact 64-token greedy parity** vs single local server, `DSpark tau = 5.16` (structured) / `2.41` (open-ended), active EPP routing across `h0` & `h1`. | Run Step 3 `R1a` (`S1 = 2048:1024`, `EP=16` Decode, `n = 32..1024`, `300 s`/level). |
| `2026-11-07T02:19Z` | **STEP 3 `R1a`: `S1 = 2048:1024` (`P`=`2x4c` Pool A, `D`=`1x8c` `EP=16` Pool B)** | **COMPLETE (Peak `40,907.05` out tok/s @ `c=1024`)** | Completed full ladder `c = 32, 64, 128, 256, 384, 512, 768, 1024` (`300.0 s` window/level, `0.00%` cache hit). Peak **`40,907.05` out tok/s** (**`5,113.38`/dec-chip**, **`2,556.69`/tot-chip**) at `c=1024` (and **`37,549.04` out tok/s = `4,693.63`/dec-chip** at `1.319 s` P50 TTFT @ `c=768`), `DSpark tau = 3.87..4.19`, `0` preemptions, **`0 / 48,345` failures**. Completed full concurrency ladder `32..1024` (`max_num_seqs=48` x `16` DP ranks = `768` decode slot cap pinned at `99.5%` fill at `c=1024`, queuing excess `256` requests into `6.20 s` P50 TTFT). | Proceed to Step 3 `R1b` (`2 x 4c` `EP=8` Decode on Pool B). |
| `2026-11-07T03:11Z` | **STEP 3 `R1b`: `S1 = 2048:1024` (`P`=`2x4c` Pool A, `D`=`2x4c` `EP=8` Pool B)** | **IN PROGRESS (`c=32` done)** | Completed `c = 32`; latest `c=32` = **`4,149.10` out tok/s** (`518.64`/dec-chip, `tau=3.8499`, `P90_low=116.05` tok/s, `0` err). Continuing ladder... | Continue `R1b` concurrency escalation (`32..1024`). |


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
| **`c=256`** | `300 s` | **`12,943.13`** (`+9.68%`) | **`1,617.89`** | **`808.95`** | `14,510.8` | `39,635.39` | `57.78 / 42.30 / 34.71` \| `57.72 / 42.81 / 35.73` | `0.724 / 2.156 / 22.620` | `0.724 / 2.505 / 22.625` | `17.33 / 27.99 / 45.34` | **`4.1331`** | `15.39 / 16.0 / 18.0` (`246.2` tot) | `D`: `2.147% / 2.611%` · `P`: `0.038% / 0.707%` | `47.9% / 52.1%` | `0 / 0` | **`0 / 3935`** |
| **`c=384`** | `300 s` | **`22,051.02`** (`+70.37%`) | **`2,756.38`** | **`1,378.19`** | `22,744.8` | `67,407.39` | `61.71 / 53.18 / 49.22` \| `61.35 / 51.31 / 45.84` | `0.742 / 2.070 / 6.453` | `0.743 / 2.106 / 6.462` | `16.30 / 21.82 / 25.30` | **`4.1700`** | `22.89 / 23.0 / 26.0` (`366.2` tot) | `D`: `3.185% / 3.674%` · `P`: `0.073% / 0.707%` | `56.4% / 43.6%` | `0 / 0` | **`0 / 6660`** |
| **`c=512`** | `300 s` | **`23,138.05`** (`+4.93%`) | **`2,892.26`** | **`1,446.13`** | `24,818.5` | `71,081.73` | `49.78 / 39.13 / 29.93` \| `49.62 / 39.07 / 30.56` | `0.855 / 2.353 / 5.266` | `0.906 / 2.476 / 5.299` | `20.15 / 32.72 / 37.40` | **`4.1729`** | `30.74 / 31.0 / 34.0` (`491.9` tot) | `D`: `4.274% / 4.868%` · `P`: `0.087% / 1.041%` | `49.3% / 50.7%` | `0 / 0` | **`0 / 7040`** |
| **`c=768`** | `300 s` | **`37,549.04`** (`+62.28%`) | **`4,693.63`** | **`2,346.81`** | `39,621.1` | `115,065.84` | `54.46 / 46.02 / 40.10` \| `54.21 / 44.99 / 39.16` | `1.319 / 3.809 / 12.374` | `1.326 / 4.066 / 12.400` | `18.45 / 25.53 / 29.47` | **`4.1850`** | `44.84 / 45.0 / 48.0` (`717.5` tot) | `D`: `6.258% / 7.059%` · `P`: `0.314% / 1.041%` | `49.7% / 50.3%` | `0 / 0` | **`0 / 11401`** |
| **`c=1024`** | `300 s` | **`40,907.05`** (`+8.94%`) | **`5,113.38`** | **`2,556.69`** | `41,492.1` | `125,230.04` | `54.57 / 48.67 / 46.75` \| `54.43 / 48.55 / 46.48` | `6.196 / 10.480 / 20.871` | `6.173 / 10.668 / 20.937` | `18.37 / 21.52 / 24.80` | **`4.1595`** | `47.76 / 48.0 / 48.0` (`764.2` tot) | `D`: `7.121% / 8.214%` · `P`: `1.090% / 2.003%` | `50.8% / 49.2%` | `0 / 0` | **`0 / 12615`** |

- **Ladder Completion:** Completed full concurrency ladder `32..1024` (`max_num_seqs=48` x `16` DP ranks = `768` decode slot cap pinned at `99.5%` fill at `c=1024`, queuing excess `256` requests into `6.20 s` P50 TTFT).
- **Peak Raw Throughput (`c=1024`):** **`40,907.05` out tok/s** (**`5,113.38` out tok/s/decode-chip**, **`2,556.69` out tok/s/total-chip**, `125,230.04` total `in+out` tok/s), `DSpark tau = 4.1595`, per-user `Med / P90_low = 54.43 / 48.55` tok/s, `TTFT P50 / P95 = 6.196 / 10.480` s, `0` preemptions, `0` failures.
- **Zero-Queue Full-Bucket Sweet Spot (`c=768`):** **`37,549.04` out tok/s** (**`4,693.63` out tok/s/decode-chip**, **`2,346.81` out tok/s/total-chip**), `Med / P90_low = 54.21 / 44.99` tok/s, **`TTFT P50 / P95 = 1.319 / 3.809` s**, `DSpark tau = 4.1850` (`4.2141` in `60-290s`), `0` preemptions, `0` failures.
<!-- R1A_RESULTS_END -->

<!-- R1B_RESULTS_START -->
---

## Step 3 (`R1b`) Results — `S1 = 2048 : 1024` (`P` = Pool A `2 x 4c` `EP=8`, `D` = Pool B `2 x 4c` `EP=8`)

- **Topology:** `llm-d` `v1.0.0` (`InferencePool/pd-disaggregation` + `EndpointPicker` + standalone Envoy router) -> Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `2 x 4-chip` independent single-host `DP=8, EP=8` Decode (`llmd-decode-poolb-h0` on `059c` + `llmd-decode-poolb-h1` on `cz6l`, `2` instances x `8` DP ranks = `16` total DP ranks).
- **Caps & Speculative Decoding:** `max_num_seqs=48`, `max_num_batched_tokens=288` per DP rank (`48 x 16 = 768` global decode slots; `compile_sizes=[16, 40, 48, 64, 96, 192, 288]`), `DSpark K=5` (`[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]`).
- **Workload (`S1`):** Calibrated exact `ISL = 2,048` tokens (`word_count = 2048` random single-token words per request, `0.00%` realized prefix cache hit, `/reset_prefix_cache` on all `4` vLLM servers before every level), forced `OSL = 1,024` tokens (`min_tokens=1024, max_tokens=1024, ignore_eos=True, temperature=0.6`), `300.0 s` closed-loop steady window + clean drain per level.

| Level | Window | Steady Out tok/s (Gain) | Out tok/s / Dec Chip (`/8`) | Out tok/s / Tot Chip (`/16`) | Server Gen tok/s (`60-290s`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Pod Split (`P h0/h1` · `D b0/b1`) | Preempt (`P / D`) | Failures |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---|---|---:|---:|
| **`c=32`** | `300 s` | **`4,149.10`** *(base)* | **`518.64`** | **`259.32`** | `4,307.8` | `12,559.55` | `141.92 / 116.69 / 100.21` \| `141.86 / 116.05 / 98.99` | `0.345 / 0.547 / 4.155` | `0.345 / 0.549 / 4.155` | `7.05 / 10.10 / 13.18` | **`3.8499`** | `1.93 / 2.0 / 3.0` (`30.9` tot) | `D`: `0.422% / 1.194%` · `P`: `0.002% / 0.118%` | `P`:`45.7/54.3%` · `D`:`49.2/50.8%` | `0 / 0` | **`0 / 1233`** |
<!-- R1B_RESULTS_END -->
