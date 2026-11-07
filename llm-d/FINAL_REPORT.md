# DeepSeek-V4.1-Flash-1017 on TPU `v7x` with `llm-d` — Final Report (`EP=16` vs `EP=8` Decode, P/D Disaggregation, Cache-Aware Routing)

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`) · **Date:** `2026-11-06/07` · **Artifacts:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/` (status mirror `gs://YOUR_GCS_BUCKET/atlas/v7x16-llmd-20261106/`)
- **Live log of every step and level:** [`STATUS.md`](STATUS.md) · per-run summaries: [`R1_INTERIM_SNAPSHOT.md`](R1_INTERIM_SNAPSHOT.md), [`step3_runs/r2_summary.md`](step3_runs/r2_summary.md), [`step3_runs/r3_decode_ceiling/r3_summary.md`](step3_runs/r3_decode_ceiling/r3_summary.md)
- All numbers below are measured. Interpolated values are labeled `[interpolated between c=X and c=Y]`; nothing is extrapolated.

---

## 1. Setup

| Item | Value |
|---|---|
| Hardware | `ank-tpu-v7x-8-pool-a` (`2x2x2`: hosts `rqkr`, `xq6s`) + `ank-tpu-v7x-8-pool-b` (`2x2x2`: hosts `059c`, `cz6l`) = **`16` physical `tpu7x` chips** (`1` chip = `2` TensorCores). All per-chip numbers are per physical chip. |
| Serving path | **Path A — `llm-d` `v1.0.0`** (no fallback to Path B was needed) |
| `llm-d` components | Helm `oci://ghcr.io/llm-d/charts/llm-d-router-standalone:v1.0.0` (`sha256:c1d1af55382e…024c`); GAIE `v1.6.2` CRDs (`InferencePool` `inference.networking.k8s.io/v1`); EndpointPicker `ghcr.io/llm-d/llm-d-router-endpoint-picker:v1.0.0` (`sha256:024188e998f5…df2c`); standalone Envoy `envoyproxy/envoy:distroless-v1.33.2` (`sha256:85500e28ed08…e4e8`); disagg routing sidecar `ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0` (`sha256:83cdf2019948…ed04`, `--kv-connector=nixlv2`). EPP + router pinned to CPU `default-pool`. |
| KV connector | `TPURaidenConnector` (`vllm_torchtpu.distributed.kv_transfer.tpu_connector`, Stage-3 store reshard, `kv_producer` on prefill / `kv_consumer` on decode), driven by the sidecar's `nixlv2` P/D protocol |
| LWS | `LeaderWorkerSet` `v0.7.0` — used for the `1 x 8-chip` `EP=16` decode (`llmd-decode-ep16`, leader `059c` = DP ranks `0..7`, worker `cz6l` = DP ranks `8..15`); `EP=8` instances are single-host pods |
| Engine | `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` + `stack_27be32f509754e88.tar` (`vllm-torchtpu` `ebe4298eb`, `vllm` `77a1dc7f9`, `libtpu==0.0.48`), `PYTHONHASHSEED=0` |
| **Precision (every table)** | Attention / non-expert **`FP8`** (`float8_e4m3fn`, `ue8m0` scales; MLA core `BF16`) · routed MoE experts **`MXFP4`** (`384` routed + `1` shared, `top-6`, `W4A8` fused EP GMM) · KV cache **`FP8`** (`--kv-cache-dtype fp8`) |
| Decode flags (all runs, from live `/proc/<pid>/cmdline`) | `vllm serve … --port 9400 --api-server-count 8 --data-parallel-size 8 --data-parallel-size-local 8 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 48 --max-num-batched-tokens 288 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --async-scheduling --kv-transfer-config {TPURaidenConnector, kv_consumer} --compilation-config {"compile_sizes":[16,40,48,64,96,192,288]} --speculative-config {"method":"dspark","num_speculative_tokens":5}` (`EP=16` LWS: same with `--data-parallel-size 16`, worker `--headless --data-parallel-start-rank 8`), `CED=0` |
| Prefill flags (from live cmdline) | `vllm serve … --port 8000 --api-server-count 8 --data-parallel-size 8 --data-parallel-size-local 8 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --prefix-cache-retention-interval 1024 --no-async-scheduling --kv-transfer-config {TPURaidenConnector, kv_producer} --compilation-config {"compile_sizes":[512,2048,8192]}`, `CED_PREFILL=1`, no DSpark |
| Topologies | `R1a`: P = Pool A `2 x 4c` `EP=8`, D = Pool B `1 x 8c` `EP=16` (LWS) · `R1b`/`R2`/`R3`/`R4`/`R6`: P = Pool A `2 x 4c`, D = Pool B `2 x 4c` `EP=8` (`2P2D`) · `R5`: `1P3D` (P = Pool A `h0`; D = Pool A `h1` + Pool B `2 x 4c`) |

---

## 2. Headline Table (`R1`–`R3`, Rule 4 columns)

Fixed shapes use random prompts, no prefix cache, `max_tokens` fixed and `ignore_eos=True`; `R3` uses `~99.3%` cached `2,048`-token prefixes so prefill never limits decode. Per-user and TTFT are over ALL requests (incl. drain). Full per-level tables: `STATUS.md`, `r2_summary.md`, `r3_summary.md`.

| Run | Shape (`ISL:OSL`) | Topology | EP / Decode Inst. | `c` | Out tok/s (Total) | /Decode Chip | /Total Chip (`/16`) | Per-User `Med / P90_low` | TTFT `P50 / P95` (s) | `τ` | Failures |
|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| `R1a` | `2048:1024` | `2P` + `1x8c` D (`/8`) | `EP=16` | `768` | `37,549.04` | `4,693.63` | `2,346.81` | `54.21 / 44.99` | `1.319 / 3.809` | `4.19` | `0 / 11,401` |
| `R1a` | `2048:1024` | `2P` + `1x8c` D (`/8`) | `EP=16` | `1024` | `40,907.05` | `5,113.38` | `2,556.69` | `54.43 / 48.55` | `6.196 / 10.480` | `4.16` | `0 / 12,615` |
| `R1b` | `2048:1024` | `2P2D` (`/8`) | `EP=8` | `768` | **`42,719.11`** | **`5,339.89`** | **`2,669.94`** | `61.33 / 53.65` | `1.284 / 2.389` | `4.18` | `0 / 12,933` |
| `R1b` | `2048:1024` | `2P2D` (`/8`) | `EP=8` | `1024` | **`46,822.96`** | **`5,852.87`** | **`2,926.44`** | `62.21 / 55.96` | `5.537 / 8.559` | `4.14` | `0 / 14,361` |
| `R2-S2` | `1024:1024` | `2P2D` (`/8`) | `EP=8` | `768` | `42,715.03` | `5,339.38` | `2,669.69` | `59.48 / 51.45` | `0.687 / 1.849` | `3.96` | `0 / 12,937` |
| `R2-S2` | `1024:1024` | `2P2D` (`/8`) | `EP=8` | `1024` | `44,166.04` | `5,520.76` | `2,760.38` | `59.12 / 51.61` | `5.828 / 9.353` | `3.94` | `0 / 13,572` |
| `R2-S3` | `8192:1024` | `2P2D` (`/8`) | `EP=8` | `512` | `26,229.45` | `3,278.68` | `1,639.34` | `75.50 / 66.69` | `5.876 / 8.704` | `3.98` | `0 / 8,020` |
| `R2-S3` | `8192:1024` | `2P2D` (`/8`) | `EP=8` | `768` (stop) | `26,185.10` | `3,273.14` | `1,636.57` | `75.60 / 66.59` | `15.677 / 18.039` | `4.00` | `0 / 8,259` |
| `R3` | `2048:1024` cached | `2P2D` (`/8`) | `EP=8` | `768` | `42,866.27` | `5,358.28` | `2,679.14` | `59.11 / 52.82` | `0.546 / 1.441` | `4.15` | `0 / 12,948` |
| `R3` | `2048:1024` cached | `2P2D` (`/8`) | `EP=8` | `1024` (stop) | `44,100.66` | `5,512.58` | `2,756.29` | `59.49 / 53.02` | `5.585 / 11.998` | `4.14` | `0 / 13,576` |

- **Limiting factors:** `S1`/`S2` saturate the decode slot cap (`48` seqs x `16` DP ranks = `768` slots; `99.5%` filled at `c=1024`, excess requests queue → TTFT jumps). `S3` is bound by Pool A prefill (`~215k` uncached prompt tok/s on `8` prefill chips; decode only `~47%` full). `R3` max sequences that fit = `768` live (slot cap), batch `47.8` per DP rank, KV `12.0%` steady / `14.6%` peak-rank — KV HBM is not the limit. `tpu-info` HBM-bandwidth utilization was not readable while PJRT held the devices (snapshots in `r3_decode_ceiling/tpu_info_*.txt`).

---

## 3. `EP=16` vs `EP=8` (`R1`) — Plain-Language Result

A larger decode EP domain did **not** raise output per decode chip on this hardware. Two independent `4`-chip `EP=8` decode instances beat one `8`-chip `EP=16` instance at **every** concurrency level (`+13.3%` to `+46.3%` output tok/s on the same `8` decode chips). At matched per-user speed of `~61 tok/s/user`, `EP=8` delivered `42,719` out tok/s (`5,340` per decode chip, `c=768`) versus `22,051` (`2,756` per decode chip, `c=384`) for `EP=16` — `1.94x`. At maximum throughput (`c=1024`) `EP=8` reached `5,853` per decode chip versus `5,113` (`+14.5%`). The measured causes: `EP=16` spans two hosts, so every MoE dispatch/combine crosses the inter-host ICI hop (TPOT `18.4 ms` vs `16.1 ms` at the same batch), and all `16` ranks share one step barrier, so whenever any rank crosses an XLA bucket boundary (`96→192` tokens at `c=256`, `192→288` at `c=512`) all ranks slow down; two `8`-rank instances step independently. `EP=16` does give more KV room (`51.25 GiB` KV HBM per TensorCore, `99.9M` FP8 tokens vs `63.7M` for `2 x EP=8`), which only matters for workloads needing more than `~64M` live KV tokens per `8` decode chips.

---

## 4. Throughput vs Interactivity (per shape, `2P2D` `EP=8`)

`Out tok/s / Decode Chip (/8)` · `/ Total Chip (/16)` at target per-user median speeds (all-requests median). Interpolation only between two bracketing measured levels of the same shape.

| Target | `S2 = 1024:1024` | `S1 = 2048:1024` (`R1b`) | `S3 = 8192:1024` |
|---|---|---|---|
| `~50 tok/s/user` | N/A — lowest measured `59.12` tok/s/user (`c=1024`: `5,520.8` / `2,760.4`) | N/A — lowest measured `61.33` tok/s/user (`c=768`: `5,339.9` / `2,669.9`) | N/A — lowest measured `74.34` tok/s/user (`c=384`: `3,023.8` / `1,511.9`) |
| `~65 tok/s/user` | `3,586.3` / `1,793.2` `[interpolated between c=384 and c=512]` | `4,984.9` / `2,492.5` `[interpolated between c=512 and c=768]` | N/A — lowest measured `74.34` tok/s/user |
| `~100 tok/s/user` | `1,911.1` / `955.5` `[interpolated between c=128 and c=256]` | `2,135.0` / `1,067.5` `[interpolated between c=128 and c=256]` | `2,803.0` / `1,401.5` `[interpolated between c=256 and c=384]` |
| `~150 tok/s/user` | N/A — highest measured `137.13` (`c=32`: `507.6` / `253.8`) | N/A — highest measured `141.86` (`c=32`: `518.6` / `259.3`) | N/A — highest measured `137.72` (`c=32`: `463.6` / `231.8`) |

---

## 4b. Ratio Check (`R5`, `1P3D`, `S1 = 2048:1024`)

`P` = Pool A `h0` (`4c`, `EP=8`); `D` = Pool A `h1` (`4c`, `EP=8`) + Pool B `2 x 4c` (`EP=8`) → `12` decode chips / `16` total. Same caps per DP rank (`48 / 288`, `DSpark K=5`), `0%` prefix hit, `ignore_eos=True`, `300 s`/level. EndpointPicker split decode requests `34.6 / 32.4 / 33.0%` across the three decode instances.

| `c` | Out tok/s | /Decode Chip (`/12`) | /Total Chip (`/16`) | Per-User `Med / P90_low` | TTFT `P50 / P95` (s) | `τ` | Decode batch/rank / Prefill running | `R1b` (`2P2D`) /Total Chip | Δ vs `R1b` | Failures |
|---:|---:|---:|---:|---|---|---:|---|---:|---:|---:|
| `32` | **`4,200.53`** | `350.04` | **`262.53`** | `146.25 / 116.87` | `0.417 / 0.980` | `3.70` | `1.25` / `0.00` | `259.32` | `+1.2%` | `0 / 1249` |
| `64` | **`7,773.56`** | `647.80` | **`485.85`** | `135.48 / 118.00` | `0.868 / 1.333` | `3.92` | `2.42` / `0.00` | `502.77` | `-3.4%` | `0 / 2309` |
| `128` | **`14,951.41`** | `1,245.95` | **`934.46`** | `133.71 / 116.04` | `0.979 / 1.823` | `4.05` | `4.78` / `0.06` | `847.66` | `+10.2%` | `0 / 4454` |
| `256` | **`22,459.46`** | `1,871.62` | **`1,403.72`** | `102.95 / 88.66` | `1.214 / 2.112` | `4.11` | `9.46` / `0.53` | `1,183.19` | `+18.6%` | `0 / 6722` |
| `384` | **`32,370.40`** | `2,697.53` | **`2,023.15`** | `97.21 / 81.50` | `1.234 / 2.001` | `4.14` | `14.39` / `0.76` | `1,638.34` | `+23.5%` | `0 / 9689` |
| `512` | **`33,960.54`** | `2,830.05` | **`2,122.53`** | `73.79 / 65.81` | `1.300 / 2.257` | `4.14` | `19.51` / `1.31` | `2,020.68` | `+5.0%` | `0 / 10225` |
| `768` | **`49,512.35`** | `4,126.03` | **`3,094.52`** | `78.98 / 68.60` | `1.993 / 4.143` | `4.15` | `28.03` / `12.20` | `2,669.94` | `+15.9%` | `0 / 14935` |
| `1024` (stop) | **`51,134.93`** | `4,261.24` | **`3,195.93`** | `76.53 / 58.54` | `6.984 / 8.271` | `4.17` | `29.65` / `15.33` | `2,926.44` | `+9.2%` | `0 / 15580` |

- **Result:** moving one host from prefill to decode raised output per **total** chip at every level from `c=128` up (`+5.0%` at `c=512` to `+23.5%` at `c=384`; `+1.2%` / `-3.4%` at `c=32` / `64`), with peak **`51,134.93` out tok/s** (`3,195.93` per total chip, `4,261.24` per decode chip) at `c=1024` versus `46,822.96` (`2,926.44` per total chip) for `2P2D` (`+9.2%`). At `c=768` it delivers `49,512.35` out tok/s at `78.98` tok/s/user median (TTFT P50 `1.99 s`) versus `42,719.11` at `61.33` tok/s/user for `2P2D`. Output per **decode** chip is lower (`4,126` vs `5,340` at `c=768`) because the decode slot cap rises to `48 x 24 = 1,152` and the extra decode chips run at lower batch (`28–30` of `48` slots per rank at `c≥768`).
- **Limiting factor:** from `c=768` the single `4`-chip prefill instance saturates (`~101–105k` uncached prompt tok/s, prefill running `12.2 → 15.3` sequences, TTFT P50 `2.0 → 7.0 s`), so the ladder stopped at `c=1024` on the `<5%` gain rule. For `S1` on `16` chips, `1P3D` gives more output per total chip than `2P2D` up to that prefill ceiling; for prefill-heavy `S3` (`8:1`) the `R2` data shows prefill is already the bottleneck with two prefill instances, so `2P2D` (or more prefill) is the better ratio there.

---

## 5. Comparison Block (best per decode chip at `S1` vs LMSYS GB200 NVL72)

- **Matching reference:** our MoE experts run in **`MXFP4`** with **`FP8`** attention, so the matching public number is **LMSYS `13,386` out tok/s per decode GPU (FP8 attention + NVFP4 MoE)**; `9,087` (BF16 attention + FP8 MoE) and `7,583` (FP8, earlier post) are listed for context only.
- **Ours (measured):** `5,852.87` out tok/s per decode chip at `S1`, `c=1024` (`62.21` tok/s/user median, TTFT P50 `5.54 s` with queueing) and `5,339.89` per decode chip at `c=768` (`61.33` tok/s/user, TTFT P50 `1.28 s`). That is `43.7%` / `39.9%` of `13,386` per accelerator.
- **Differences that prevent a like-for-like reading:** decode `EP=8` on `4` chips (one host) vs `EP=48` on `48` GPUs (NVL72 NVLink domain); model DeepSeek-V4.1-Flash-1017 (`384` routed experts, `top-6`, sparse CSA attention, DSpark `K=5` speculative decoding) vs DeepSeek V3/R1; `MXFP4` W4A8 vs `NVFP4`; ISL `2,048` vs `2,000`; our decode slot cap (`48` seqs/rank) limits batch per rank; the per-user speed behind the LMSYS figure is not part of our reference, so no matched-interactivity comparison is claimed.

---

## 6. `llm-d` Routing Result (`R4`)

<!-- FR_R4_START -->
- **Workload:** `requests_mocked.jsonl.xz` (`50:20:12:9:9` ISL mix), `256` distinct trunks (`128` per prefill instance), `80%` target hit, natural stop (`ignore_eos=False`), `600 s` per level, no cache reset between `n = 64 -> 128 -> 256`; topology `2P2D` (`EP=8` decode). TTFT over ALL requests incl. drain.

| Routing condition | `n` | Realized hit (API) | TTFT `P50 / P95 / P99` (s) | Out tok/s | Out tok/s / Total Chip (`/16`) | Per-User `Med / P90_low` | Measured mean ISL / OSL | `τ` | Failures |
|---|---:|---:|---|---:|---:|---|---|---:|---:|
| (a) cache-blind (`token-load`) | `64` | **`1.60%`** | `2.322 / 7.028 / 9.472` | `4,503.93` | **`281.50`** | `95.39 / 81.53` | `18,780.4 / 747.7` | `3.171` | `0 / 3650` |
| (a) cache-blind (`token-load`) | `128` | **`2.09%`** | `2.921 / 8.468 / 10.917` | `8,189.51` | **`511.84`** | `91.56 / 78.51` | `18,782.7 / 746.4` | `3.142` | `0 / 6672` |
| (a) cache-blind (`token-load`) | `256` | **`2.71%`** | `10.835 / 19.371 / 23.484` | `9,762.40` | **`610.15`** | `89.34 / 76.08` | `18,778.6 / 742.1` | `3.152` | `0 / 8093` |
| (b) `llm-d` prefix-aware | `64` | **`2.86%`** | `2.354 / 7.044 / 9.781` | `4,461.01` | **`278.81`** | `94.17 / 79.59` | `18,770.2 / 729.4` | `3.162` | `0 / 3714` |
| (b) `llm-d` prefix-aware | `128` | **`2.05%`** | `3.115 / 8.950 / 10.785` | `7,967.55` | **`497.97`** | `90.28 / 75.96` | `18,769.8 / 718.0` | `3.138` | `0 / 6749` |
| (b) `llm-d` prefix-aware | `256` | **`3.23%`** | `10.898 / 18.740 / 22.569` | `9,642.22` | **`602.64`** | `87.64 / 73.28` | `18,778.0 / 728.1` | `3.155` | `0 / 8164` |
| (c) prefix-aware + DP-rank pin | `64` | **`45.69%`** | `1.825 / 7.047 / 9.107` | `4,584.96` | **`286.56`** | `91.91 / 77.91` | `18,805.5 / 708.6` | `3.188` | `0 / 3912` |
| (c) prefix-aware + DP-rank pin | `128` | **`68.08%`** | `1.778 / 30.884 / 41.607` | `7,230.94` | **`451.93`** | `82.88 / 65.21` | `18,811.5 / 707.8` | `3.160` | `0 / 6218` |
| (c) prefix-aware + DP-rank pin | `256` | **`62.95%`** | `1.833 / 167.782 / 176.501` | `5,940.05` | **`371.25`** | `86.48 / 57.79` | `18,805.3 / 686.3` | `3.125` | `0 / 5412` |
| (d) cache-blind + DP-rank pin | `64` | **`29.04%`** | `1.919 / 7.006 / 8.809` | `4,457.83` | **`278.61`** | `93.46 / 79.80` | `18,822.7 / 699.4` | `3.152` | `0 / 3886` |
| (d) cache-blind + DP-rank pin | `128` | **`48.29%`** | `1.824 / 44.134 / 49.222` | `5,722.09` | **`357.63`** | `90.92 / 77.35` | `18,821.8 / 697.4` | `3.103` | `0 / 5019` |
| (d) cache-blind + DP-rank pin | `256` | **`49.90%`** | `1.878 / 99.133 / 129.498` | `6,768.88` | **`423.05`** | `86.81 / 62.73` | `18,815.0 / 713.3` | `3.118` | `0 / 5934` |

- **Result as deployed ((a) vs (b)):** realized hit stays at `1.6–3.2%` with either routing policy, so prefix-aware routing gave no measurable gain on this deployment. Cause: each prefill instance is one `DP=8` vLLM server behind a single endpoint. The EndpointPicker chooses the pod; vLLM's internal DP load balancer then scatters a trunk's turns across `8` independent rank-local prefix caches.
- **With DP-rank affinity ((c) vs (d)):** pinning each trunk to a rank with `X-data-parallel-rank` (verified end-to-end through `EPP → disagg sidecar → prefill vLLM`; the same mechanism `toy_proxy_server.py` used for `R0` and the prior runs) lifts realized hit to `45.7 / 68.1 / 63.0%` with prefix-aware pod routing versus `29.0 / 48.3 / 49.9%` cache-blind (**+16.7, +19.8, +13.1 points**). TTFT P50 drops to `~1.8–1.9 s` (vs `2.3–10.9 s` unpinned), but static `trunk % 8` rank pinning overloads some ranks: TTFT P95 rises to `31–168 s` and output per total chip falls at `n = 128 / 256`. `R6` therefore uses a load-weighted (balanced) trunk-to-rank map.
- **Takeaway:** `llm-d` prefix-aware routing adds about `13–20` points of realized cache hit here, but only after the engine exposes DP-rank-level affinity. Exposing each prefill DP rank as its own `InferencePool` endpoint (or adding a DP-aware picker) is the deployment change needed to get this without client-side pinning or load imbalance.
<!-- FR_R4_END -->

---

## 7. Reproducibility (`R0` vs prior `7,446`)

`R0` (`1P1D`, `n=128`, agentic `~18K` mean ISL, `95%` target hit, `300 s`): **`7,423.72` out tok/s** (`1,855.93` per decode chip, `927.97` per total chip), `DSpark τ = 2.8755`, per-user `67.83 / 54.65` (median / P90_low), TTFT P50 / P95 `1.205 / 1.559 s`, realized hit `95.47%`, `0 / 4,631` failures — **`-0.30%`** vs the `7,446` anchor (`+0.20%` vs the prior `A1-b` run `7,408.59`). Pass gate (`6,700–8,200`, `τ > 2`) met.

<!-- FR_R6_START -->
**`R6` (agentic continuity, `2P2D`, `~95%` hit, prefix-aware routing + balanced DP-rank affinity, `300 s`/level):**

| `n` | Target Hit | Realized Hit (API) | Out tok/s | /Dec Chip (`/8`) | /Tot Chip (`/16`) | Total (`in+out`) tok/s | TTFT `P50 / P95 / P99` (s, ALL) | Per-User `Med / P90_low / P95_low` | Mean ISL / OSL | `DSpark` $\tau$ | Failures |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| `128` | `95%` | **`95.32%`** | **`8,529.90`** | `1,066.24` | `533.12` | `291,481.29` | `0.965 / 1.452 / 1.604` | `70.56 / 59.19 / 56.10` | `17,945.6 / 548.5` | `2.9835` | `0 / 4744` |
| `256` | `95%` | **`95.31%`** | **`10,753.45`** | `1,344.18` | `672.09` | `379,982.30` | `1.154 / 1.597 / 2.575` | `43.78 / 37.66 / 35.75` | `17,951.6 / 535.9` | `2.9492` | `0 / 6195` |

- **Continuity vs prior results (same model, agentic `~18K` input, `~95%` hit):** prior `2P2D n=256` = `14,542` out tok/s (`1,818` per decode chip, `~68` tok/s/user); this run `2P2D n=256` = `10,753.45` (`1,344.18` per decode chip, `43.78` tok/s/user median) = `-26.1%`. Prior `1P1D n=128` = `7,446` (`1,862` per decode chip, 4 decode chips); this run `2P2D n=128` (8 decode chips) = `8,529.90`. Realized hit matches the target (`95.3%`) and TTFT P95 stays `<1.6 s`. Likely contributor to the lower per-chip output (not isolated): the same `X-data-parallel-rank` header also pins the *decode* rank per trunk, whereas the prior `toy_proxy_server.py` balanced decode ranks by least in-flight; the `~80%` continuity point is the `R4` table.
<!-- FR_R6_END -->

---

## 8. Correctness Gate

**PASS.** Through `llm-d` (`EPP → Pool A 2 x 4c EP=8 prefill (CED=1) → TPURaidenConnector Stage-3 reshard → Pool B decode`), `20 / 20` prompts produced **bit-exact `64`-token greedy output** versus a single local server, plus `16 / 16` QA sanity prompts; `DSpark τ = 5.16` (structured) / `2.41` (open-ended). Script and raw results: `step2_llmd/step2e_greedy_correctness_gate.py`, `step2_llmd/step2e_greedy_gate.json`. (A prior run had failed this gate; this run used `PYTHONHASHSEED=0` on all roles.)

---

## 9. Failures, Fallbacks, `[VERIFY]` Outcomes, Runs Not Completed, Next Steps

- **`[VERIFY]` outcomes:** independent 4-chip single-host split on `2x2x2` = **YES** (`3.41–3.53 s` bring-up); `vllm-torchtpu` `EP=16` multi-host DSV4.1 decode = **YES** after four multi-host `mp` DP bootstrap fixes (GKE LWS TPU env injection, `TPU_SKIP_MDS_QUERY`, rank binding, pre-init barrier; see `step2_llmd/step2_llmd_ep16_report.md`); `llm-d` accepts a custom `--kv-transfer-config` → `TPURaidenConnector` used; `X-data-parallel-rank` propagates `EPP → sidecar → prefill vLLM` = **YES** (probe: rank-matched `8,192–9,216` cached tokens, mismatched `0`).
- **Fallbacks:** none needed (Path A throughout; `EP=16` started without the `2 x EP=8` fallback; `EP=8` was then chosen on measurements).
- **Incidents handled during the run:** `generate_r2_summary.py` / `generate_r3_summary.py` crashed on a renamed tau key after `R2-S3` (stopping the pipeline before `R3`); fixed and resumed with no data loss. `R4` (a)/(b) exposed intra-pod DP-rank cache fragmentation; added DP-pinned conditions (c)/(d).
<!-- FR_R5_START -->
- **Runs completed:** `R0`, Step 2 gate, `R1a`, `R1b`, `R2` (`S2`, `S3`), `R3`, `R4` (4 conditions), `R5` (`1P3D`), `R6` (`~95%`). Runs not completed: none of the spec'd runs; cap variants (`64/384`, `96/576`) were not tried because KV HBM stayed `<15%` and the decode step, not KV, bounded throughput.
<!-- FR_R5_END -->
- **Next steps:** expose each prefill DP rank as its own `InferencePool` endpoint (or a DP-aware picker) so `llm-d` prefix-aware routing works without client-side rank pinning; raise `max_num_seqs` / caps (`64/384`, `96/576`) on `EP=8` decode since KV HBM stays `<15%` used at the slot cap; a single larger slice for `EP=32+` decode, with per-host step barriers decoupled.
