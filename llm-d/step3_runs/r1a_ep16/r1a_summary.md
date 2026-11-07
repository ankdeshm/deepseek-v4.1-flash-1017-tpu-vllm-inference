# Step 3 (`R1a`) Summary Report — `S1 = 2048 : 1024` (`P` = Pool A `2 x 4c` `EP=8`, `D` = Pool B `1 x 8c` `EP=16` LWS)

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **GCS Artifacts:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/step3_runs/r1a_ep16/`
- **Completed At (UTC):** `2026-11-07T02:00:29Z`

---

## 1. Topology & Configuration

- **Routing Layer (`llm-d` `v1.0.0` Path A):** Standalone Envoy router (`Service/pd-disaggregation-epp:80`) + `EndpointPicker` (`InferencePool/pd-disaggregation`, `disagg-profile-handler` + `prefill-filter` / `decode-filter` + `load-aware-scorer` + `prefix-cache-scorer`) + `llm-d-router-disagg-sidecar:v1.0.0` (`nixlv2` -> `TPURaidenConnector` Stage-3 cross-pool KV transfer).
- **Prefill Pool (`8` physical `tpu7x` chips on Pool A):** `2 x 4-chip` single-host pods (`llmd-prefill-poola-h0` `192.168.100.43:8000`, `llmd-prefill-poola-h1` `192.168.100.38:8000`), each `DP=8, TP=1, EP=8`, `--kv-cache-dtype fp8`, `kv_producer`.
- **Decode Pool (`8` physical `tpu7x` chips on Pool B):** `1 x 8-chip` `LeaderWorkerSet/llmd-decode-ep16` (`llmd-decode-ep16-0` `192.168.100.198` + `llmd-decode-ep16-0-1` `192.168.100.199`), `DP=16, dp_local=8, TP=1, EP=16`, `--kv-cache-dtype fp8`, `kv_consumer`, `29.28 GiB` weights/TensorCore, `51.25 GiB` available KV HBM/TensorCore (`99,903,072` FP8 KV tokens across `16` ranks).
- **Scheduler Caps & Speculative Decoding:** `max_num_seqs = 48`, `max_num_batched_tokens = 288` per DP rank (`compile_sizes = [16, 40, 48, 64, 96, 192, 288]`), `DSpark` speculative decoding `K=5` (`[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]`).
- **Workload (`S1 = 2048 : 1024`, `0.00%` Prefix Cache Hit):**
  - Auto-calibrated `word_count = 2048` single-token words -> exact `usage.prompt_tokens = 2048.0` on every request, unique per-request deterministic RNG seeds (`0.0000%` realized prefix cache hit across all `24,329` measured requests), `POST /reset_prefix_cache` verified (`HTTP 200`) on all 3 vLLM servers before every concurrency level.
  - Forced exact `OSL = 1024.0` tokens (`max_tokens=1024, min_tokens=1024, ignore_eos=True, temperature=0.6`), `15.0 s` warmup at `c=16` (`66/66` OK), then `300.0 s` exact steady-state window + full in-flight drain per level.

---

## 2. Complete Rule 4 Metrics Table (`c = 32 .. 512`)

| Concurrency (`c`) | Steady Window | Steady Out tok/s (Gain) | Out tok/s / Decode Chip (`/8`) | Out tok/s / Total Chip (`/16`) | Server Gen tok/s (`60-290s`) | Steady In tok/s | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\tau$ (`Full` / `60-290s`) | Decode Batch / DP Rank (`mean / med / max`, `30-295s`) | KV Usage % `Steady / Peak-Rank` | Prefill Split (`h0 / h1`) | Preemptions (`P / D`) | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|---:|
| **`32`** | `300.0 s` | **`3,541.90`** *(base)* | **`442.74`** | **`221.37`** | `3,687.40` | `7,195.31` | `10,737.21` | `120.04 / 96.98 / 87.49` \| `120.12 / 97.01 / 87.52` | `0.391 / 0.894 / 5.056` | `0.391 / 0.898 / 5.056` | `8.33 / 11.43 / 14.69` | **`3.8725`** / `3.8693` | `1.92 / 2.0 / 4.0` (`30.70` tot) | `D`: `0.268% / 0.643%` · `P`: `0.003% / 0.118%` | `44.5% / 55.5%` | `0 / 0` | **`0 / 1,054`** |
| **`64`** | `300.0 s` | **`7,102.50`** (`+100.53%`) | **`887.81`** | **`443.91`** | `7,416.76` | `14,417.92` | `21,520.42` | `121.17 / 100.21 / 89.55` \| `121.27 / 100.43 / 89.91` | `0.421 / 1.097 / 3.764` | `0.422 / 1.120 / 3.765` | `8.25 / 11.12 / 13.96` | **`3.9901`** / `3.9872` | `3.80 / 4.0 / 6.0` (`60.83` tot) | `D`: `0.529% / 1.194%` · `P`: `0.007% / 0.471%` | `48.6% / 51.4%` | `0 / 0` | **`0 / 2,114`** |
| **`128`** | `300.0 s` | **`11,800.75`** (`+66.15%`) | **`1,475.09`** | **`737.55`** | `13,597.36` | `23,975.25` | `35,776.00` | `111.60 / 86.63 / 55.44` \| `111.81 / 87.57 / 58.93` | `0.502 / 1.270 / 6.416` | `0.501 / 1.300 / 6.433` | `8.94 / 16.97 / 39.94` | **`4.1321`** / `4.1346` | `7.61 / 8.0 / 10.0` (`121.81` tot) | `D`: `1.063% / 1.509%` · `P`: `0.019% / 0.236%` | `50.1% / 49.9%` | `0 / 0` | **`0 / 3,526`** |
| **`256`** | `300.0 s` | **`12,943.13`** (`+9.68%`) | **`1,617.89`** | **`808.95`** | `14,510.79` | `26,692.27` | `39,635.39` | `57.78 / 42.30 / 34.71` \| `57.72 / 42.81 / 35.73` | `0.724 / 2.156 / 22.620` | `0.724 / 2.505 / 22.625` | `17.33 / 27.99 / 45.34` | **`4.1331`** / `4.1523` | `15.39 / 16.0 / 18.0` (`246.21` tot) | `D`: `2.147% / 2.611%` · `P`: `0.038% / 0.707%` | `47.9% / 52.1%` | `0 / 0` | **`0 / 3,935`** |
| **`384`** | `300.0 s` | **`22,051.02`** (`+70.37%`) | **`2,756.38`** | **`1,378.19`** | `22,744.85` | `45,356.37` | `67,407.39` | `61.71 / 53.18 / 49.22` \| `61.35 / 51.31 / 45.84` | `0.742 / 2.070 / 6.453` | `0.743 / 2.106 / 6.462` | `16.30 / 21.82 / 25.30` | **`4.1700`** / `4.1672` | `22.89 / 23.0 / 26.0` (`366.19` tot) | `D`: `3.185% / 3.674%` · `P`: `0.073% / 0.707%` | `56.4% / 43.6%` | `0 / 0` | **`0 / 6,660`** |
| **`512`** *(STOP)* | `300.0 s` | **`23,138.05`** (`+4.93%`) | **`2,892.26`** | **`1,446.13`** | `24,818.51` | `47,943.68` | `71,081.73` | `49.78 / 39.13 / 29.93` \| `49.62 / 39.07 / 30.56` | `0.855 / 2.353 / 5.266` | `0.906 / 2.476 / 5.299` | `20.15 / 32.72 / 37.40` | **`4.1729`** / `4.1631` | `30.74 / 31.0 / 34.0` (`491.89` tot) | `D`: `4.274% / 4.868%` · `P`: `0.087% / 1.041%` | `49.3% / 50.7%` | `0 / 0` | **`0 / 7,040`** |

- **Stop Rule Triggered after `c=512`:** **Stop Rule 1** — `steady_out_tok_s` gain was **`+4.93%`** (`< +5.0%` threshold) vs `c=384` (`22,051.02 -> 23,138.05` out tok/s). All other stop rules remained healthy (`P90_low = 39.07 tok/s >= 20.0`, `0` preemptions, `0 / 24,329` total failures across all 6 levels, `DSpark tau = 4.1729 >= 1.5`, `0` pod restarts).

---

## 3. Limiting-Factor & Bottleneck Analysis

1. **Peak Throughput & Sweet-Spot Operating Points:**
   - **Maximum Throughput Point (`c=512`):** **`23,138.05` out tok/s** (`2,892.26` out tok/s per decode chip, `1,446.13` out tok/s per total chip; `24,818.51` server gen tok/s in `60-290s`; `71,081.73` total `in+out` tok/s), with `Med = 49.62` tok/s, `P90_low = 39.07` tok/s, and `TTFT P50 / P95 = 0.855 / 2.353 s`.
   - **High-Interactivity Sweet Spot (`c=384`):** **`22,051.02` out tok/s** (`95.3%` of `c=512` throughput; **`2,756.38` out tok/s per decode chip**) while maintaining **`Med = 61.35` tok/s, `P90_low = 51.31` tok/s (`>= 50` tok/s), and `TTFT P95 = 2.070 s` (`<= 3.0 s`)** simultaneously!
   - **Ultra-Low-Latency Point (`c=64..128`):** **`120–121` tok/s median per user** (`8.25–8.94 ms` TPOT, `0.39–0.50 s` `P50` TTFT) delivering `7,102.50` (`c=64`) to `11,800.75` (`c=128`) out tok/s.

2. **Why `c=256` Exhibited a Temporary Knee (`+9.68%`) Followed by a `+70.37%` Surge at `c=384`:**
   - With `DSpark K=5`, each active decode sequence contributes `1 + 5 = 6` query tokens per target forward step against `compile_sizes = [16, 40, 48, 64, 96, 192, 288]`.
   - At `c=256` (`256 / 16 = 16.0` seqs/DP rank -> `16 * 6 = 96` batched tokens), the 16 DP ranks straddle the exact `96 -> 192` XLA bucket boundary (`mean=15.39, med=16.0, max=18.0`). Because all 16 DP ranks execute synchronous `EP=16` MoE `all_to_all` collectives across the 2 hosts (`2x2x2` ICI), whenever any single DP rank holds `17–18` sequences (`102–108` tokens), that rank executes the **`192`-token compiled XLA graph** at only `~53%` fill and stalls the other 15 ranks at the collective barrier (`TPOT P50` jumps `8.94 -> 17.33 ms`).
   - At `c=384` (`22.89 / 23.0 / 26.0` seqs/DP rank -> `138–156` tokens), **all 16 DP ranks operate uniformly inside the same `192`-token XLA bucket** (`~72–81%` fill, zero cross-bucket straggler skew). Step time stays virtually unchanged (`~68.0 ms` vs `~71.6 ms` at `c=256`), so median per-user speed actually **improves** (`57.72 -> 61.35` tok/s) and total output throughput surges **`+70.37%`** (`12,943.13 -> 22,051.02` out tok/s)!

3. **Why Throughput Saturated at `c=512` (`+4.93%` Gain -> Stop Rule 1):**
   - **Decode Compute + Synchronous 16-Rank `EP=16` `192 -> 288` Bucket Step Scaling (PRIMARY BOTTLENECK):** At `c=512`, `32` seqs/DP rank = `192` tokens (exact ceiling of the `192` bucket). With `per_dp_rank` `med=31.0, max=34.0`, ranks hitting `33–34` sequences (`198–204` tokens) step into the **`288`-token XLA bucket**, pushing synchronous 16-rank step latency from `~68.0 ms` (`16.30 ms * 4.170`) to `~84.1 ms` (`20.15 ms * 4.173`, `+23.7%` step latency) and yielding diminishing returns (`+4.93%` net cluster output gain).
   - **Prefill Pool Is NOT the Bottleneck:** Across `2 x 4-chip` `EP=8` Prefill pods (`h0` and `h1`), `47,943.68` uncached prompt tok/s (`~24k` tok/s/pod) were served with `prefill_running_mean <= 0.02`, `0` waiting queue, `49.3% / 50.7%` EPP load split, and sub-second `P50` TTFT (`0.855 s` at `c=512`).
   - **Raiden Cross-Pool KV Transfer Is NOT the Bottleneck:** `TPURaidenConnector` Stage-3 reshard (`P` `EP=8` -> `D` `EP=16`) sustained `23.5` req/s (`48k` prompt tok/s) with `P50 TTFT = 0.855 s` (`P95 = 2.353 s`) and `0` transfer errors across `24,329` requests.
   - **KV HBM Is Massively Under-Utilized (`< 4.9%`):** `EP=16` leaves `99.90M` FP8 KV tokens across the 8 Decode chips, so even `512` concurrent `2048:1024` sequences consumed only **`4.27%` steady / `4.87%` peak-rank** KV HBM (`0` preemptions). `R1b` (`2 x 4-chip` `EP=8` single-host Decode instances) will test whether eliminating cross-host ICI hops in MoE `all_to_all` and shrinking synchronous barrier width (`8` ranks vs `16` ranks) yields higher decode tok/s/chip at `S1` despite lower total KV capacity.
