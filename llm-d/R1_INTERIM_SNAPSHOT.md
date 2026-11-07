# `R1` Interim Snapshot: `EP=16` (`1 x 8c` LWS) vs `EP=8` (`2 x 4c`) Decode on TPU `v7x`

- **Workload & Stack:** `DeepSeek-V4.1-Flash-1017` (`FP8` attn + `MXFP4 W4A8` MoE + `FP8` KV + `DSpark K=5`), `llm-d v1.0.0` Path A (`2 x 4c EP=8` Prefill on Pool A + `8` Decode chips on Pool B = `16` `tpu7x` chips), `S1 = 2048 : 1024` (`0.00%` prefix hit, `ignore_eos=True`, `300.0 s` steady window/level, `max_num_seqs=48`, `MNBT=288`, **`0 / 106,641` total request failures**).

## 1. Head-to-Head Throughput & Latency Across Full Concurrency Ladder (`c = 32 .. 1024`)

| `c` | `R1a` (`1x8c EP=16`) Out tok/s (`/8` Dec \| `/16` Tot) | `R1a` User `Med / P90_low` (tok/s) & `TTFT P50/P95` (s) | `R1b` (`2x4c EP=8`) Out tok/s (`/8` Dec \| `/16` Tot) | `R1b` User `Med / P90_low` (tok/s) & `TTFT P50/P95` (s) | `R1b` vs `R1a` Out tok/s |
|---:|---:|---|---:|---|---:|
| **`32`** | `3,541.90` (`442.74` \| `221.37`) | `120.12 / 97.01` · `0.391 / 0.894s` (`τ=3.87`) | **`4,149.10`** (`518.64` \| `259.32`) | **`141.86 / 116.05`** · `0.345 / 0.547s` (`τ=3.85`) | **`+17.14%`** |
| **`64`** | `7,102.50` (`887.81` \| `443.91`) | `121.27 / 100.43` · `0.421 / 1.097s` (`τ=3.99`) | **`8,044.30`** (`1,005.54` \| `502.77`) | **`134.39 / 117.59`** · `0.390 / 0.869s` (`τ=4.01`) | **`+13.26%`** |
| **`128`** | `11,800.75` (`1,475.09` \| `737.55`) | `111.81 / 87.57` · `0.502 / 1.270s` (`τ=4.13`) | **`13,562.52`** (`1,695.32` \| `847.66`) | **`124.10 / 101.57`** · `0.546 / 1.340s` (`τ=4.11`) | **`+14.93%`** |
| **`256`** | `12,943.13` (`1,617.89` \| `808.95`) | `57.72 / 42.81` · `0.724 / 2.156s` (`τ=4.13`) | **`18,931.05`** (`2,366.38` \| `1,183.19`) | **`87.32 / 66.31`** · `1.088 / 2.025s` (`τ=4.13`) | **`+46.26%`** |
| **`384`** | `22,051.02` (`2,756.38` \| `1,378.19`) | `61.35 / 51.31` · `0.742 / 2.070s` (`τ=4.17`) | **`26,213.37`** (`3,276.67` \| `1,638.34`) | **`74.91 / 64.50`** · `1.153 / 2.034s` (`τ=4.14`) | **`+18.88%`** |
| **`512`** | `23,138.05` (`2,892.26` \| `1,446.13`) | `49.62 / 39.07` · `0.855 / 2.353s` (`τ=4.17`) | **`32,330.91`** (`4,041.36` \| `2,020.68`) | **`74.75 / 53.42`** · `1.238 / 2.207s` (`τ=4.17`) | **`+39.73%`** |
| **`768`** | `37,549.04` (`4,693.63` \| `2,346.81`) | `54.21 / 44.99` · `1.319 / 3.809s` (`τ=4.19`) | **`42,719.11`** (`5,339.89` \| `2,669.94`) | **`61.33 / 53.65`** · `1.284 / 2.389s` (`τ=4.18`) | **`+13.77%`** |
| **`1024`** | `40,907.05` (`5,113.38` \| `2,556.69`) | `54.43 / 48.55` · `6.196 / 10.480s` (`τ=4.16`) | **`46,822.96`** (`5,852.87` \| `2,926.44`) | **`62.21 / 55.96`** · `5.537 / 8.559s` (`τ=4.14`) | **`+14.46%`** |

## 2. Key Operating Regimes & Limiting-Factor Physics

- **Interactive Operating Point (`~60–65 tok/s/user`):**
  - **`R1b` (`2 x 4c EP=8`) at `c=768`:** **`42,719.11` out tok/s** (**`5,339.89` / decode chip**, **`2,669.94` / total chip**), `61.33` tok/s median (`53.65` `P90_low`, `16.30 ms` P50 TPOT), `1.284 s` P50 / `2.389 s` P95 TTFT (`50.0% / 50.0%` `b0/b1` split).
  - **`R1a` (`1 x 8c EP=16`) at `c=384`:** `22,051.02` out tok/s (`2,756.38` / decode chip, `1,378.19` / total chip), `61.35` tok/s median (`51.31` `P90_low`), `0.742 s` P50 / `2.070 s` P95 TTFT (at `c=768`, `EP=16` drops to `54.21` median / `44.99` `P90_low`). **`EP=8` delivers `+93.73%` (`1.94x`) higher output throughput at `~61 tok/s/user`!**
- **Max-Throughput Point (`c=768` Unqueued vs `c=1024` Saturated):**
  - At `c=768` (`~44.7` active seqs/DP rank = `94%` of `288`-token XLA bucket, zero admission queue), `EP=8` hits **`42,719.11` out tok/s** (`+13.77%` vs `EP=16` `37,549.04`). At `c=1024`, both topologies pin the `max_num_seqs = 48` (`16 x 48 = 768` global slot) cap at **`99.5%` fill** (`47.77` seqs/rank), queueing the excess `256` requests (`5.54 s` vs `6.20 s` P50 TTFT) while `EP=8` reaches **`46,822.96` out tok/s (`5,852.87` / decode chip, `143,079` total `in+out` tok/s, `+14.46%` vs `EP=16`)** and still sustains `62.21` median / `55.96` `P90_low` tok/s per active stream.
- **Exact Physical Bottlenecks (`ICI Collectives`, `XLA Buckets`, `KV HBM`, `Slot Cap`):**
  1. **Single-Host 1D ICI vs Multi-Host 2D ICI `all_to_all`:** `EP=8` keeps MoE dispatch/combine inside a single 4-chip tray (`059c` or `cz6l`), cutting step TPOT by `12–15%` at identical batch fill (`16.08 ms` vs `18.37 ms` at `c=1024`).
  2. **XLA Bucket-Straddle Decoupling (`[16, 40, 48, 64, 96, 192, 288]`):** On `1 x EP=16`, *any* rank crossing `16` seqs (`96->192` at `c=256`) or `32` seqs (`192->288` at `c=512`) stalls all `16` ranks (`+9.7%` and `+4.9%` knees). Two independent `8`-rank `EP=8` pods decouple step barriers, surging **`+46.26%` at `c=256`** and **`+39.73%` at `c=512`**.
  3. **KV HBM Capacity vs Slot Cap:** `EP=16` frees **`51.25 GiB` KV HBM/TensorCore (`99.90M` FP8 KV tokens)** vs `EP=8`'s **`32.67 GiB` KV HBM/TensorCore post-DSpark (`37.13 GiB` pre-DSpark, `63.69M` FP8 KV tokens)** (`48` vs `24` local experts/core). Even at `c=1024`, `EP=8` uses only **`11.15%` steady / `12.56%` peak KV HBM** (`0` preemptions) — throughput is bounded by the `max_num_seqs=48` (`288`-token) compute/collective step time, NOT KV HBM!

## 3. Empirical Verdict for `R2`, `R3`, `R4`, and `R5`

- **Selected Decode Winner: `R1b` (`2 x 4-chip` Single-Host `DP=8, EP=8` Decode).** It wins **8 out of 8** concurrency levels (`+13.3%` to `+46.3%`), achieves **`5,852.87` out tok/s/decode-chip** (`5,339.89` unqueued at `61.33` tok/s/user), retains `87.4%` free KV HBM (`63.69M` tokens total), and provides clean 4-chip modularity for `R2` (multi-shape `S2`/`S3`), `R3` (decode ceiling), `R4` (prefix-aware routing), and `R5` (`1P:3D` P:D ratio scaling). Reserve `EP=16` solely for ultra-long-context workloads needing `>63.7M` live KV tokens per 8 decode chips.
