# DeepSeek-V4.1-Flash-1017 on TPU v7x: 1P1D (8 chips) and 2P2D (16 chips) Results

**Run ID:** `20261031_1p1d_2p2d`
**Updated (UTC):** 2026-10-31 02:45:45
**Model:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MXFP4 MoE experts)
**Serving:** Prefill/decode (P/D) disaggregation. Context encoder (`CED=1`) on prefill, DSpark speculative decoding (K=5) on decode, `TPURaidenConnector` Stage 3 KV transfer.
**Hardware:** TPU v7x `2x2x4` slice = 4 hosts × `2x2x1` (4 chips each) = 16 chips / 32 TensorCores
**Cold boot time:** 1,590.2 s

### Node layout

| Host | Role | Pair |
|:---|:---|:---|
| 0 | Prefill | A |
| 1 | Decode | A |
| 2 | Prefill | B |
| 3 | Decode | B |

- **1P1D** uses pair A only (8 chips).
- **2P2D** uses pairs A and B behind one router (16 chips).

### Definitions

- **Win Out tok/s:** output tokens produced inside the steady-state measurement window ÷ window length.
- **Drain Out tok/s:** all output tokens ÷ total wall-clock time, including the drain of in-flight requests.
- **Total In tok/s:** all prompt tokens processed per second, **including cached prefix tokens**. Uncached prefill is roughly Total In × (1 − cache hit).
- **Med TPS/stream:** median per-user decode speed (tok/s).
- **Per chip:** output tok/s ÷ total chips (prefill + decode). "Per decode chip" divides by decode chips only.

---

## 1. Correctness and micro-latency gates (1P1D, 8 chips, `pd_gates.py`)

| Check | Result |
|:---|:---|
| **Hard gates** (2+2 ×16 and 64-token greedy equality, P/D vs. local) | **FAIL** |
| USA 50-states probe (P/D / local) | 50 / 50 |

> ⚠️ The hard gates did not pass: P/D output did not exactly match local single-server output on at least one check. The summary does not record which sub-check failed. Treat throughput numbers as performance data, not validated correctness.

### Streaming probes

| Probe | Requests | Concurrency | Prompt : output (tokens) | Output tok/s | Input tok/s | Mean TPOT (ms) | TTFT p50 / mean / max (s) | Steady-state interior output tok/s | Mean accept length (per-position acceptance) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 8K:128 | 16 | 8 | 8,192 : 128 | 342.48 | 21,918.87 | 13.13 | 0.67 / 0.88 / 1.39 | 521.61 | 2.08 (0.59 / 0.24 / 0.15 / 0.06 / 0.05) |
| 1K:512 | 64 | 64 | 1,024 : 512 | 2,378.51 | 4,757.01 | 16.10 | 2.91 / 2.47 / 3.24 | 5,801.96 | 2.69 (0.63 / 0.42 / 0.28 / 0.20 / 0.15) |

*The harness also outputs "throughput at accept length 3.5 / 4.25 / 5.0" fields. Those are what-if projections, not measurements, so they are left out here.*

---

## 2. 1P1D (8 chips): empirical ISL sweep, ~80% prefix cache hit

Input-length mix (50:20:12:9:9) across buckets 1–8K / 8–16K / 16–32K / 32–64K / 64K+. Outputs stop naturally.

| Sessions (n) | Trunks | Mean ISL | Cache hit | Mean OSL | Win Out tok/s | Drain Out tok/s | Out tok/s per chip | Total In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | 40 | 20,297 | 79.9% | 782.9 | 117.5 | 118.0 | 14.7 | 3,213.8 | 126.58 | 7.90 | 0.50 | 19 / 19 |
| 8 | 40 | 18,692 | 80.0% | 600.9 | 771.2 | 723.3 | 96.4 | 25,078.0 | 119.50 | 8.37 | 0.91 | 161 / 161 |
| 16 | 40 | 18,718 | 80.0% | 774.4 | 1,522.4 | 1,230.0 | 190.3 | 39,921.5 | 110.20 | 9.07 | 1.09 | 258 / 258 |
| 32 | 40 | 18,745 | 80.0% | 668.8 | 2,511.0 | 2,258.1 | 313.9 | 72,133.2 | 94.28 | 10.61 | 1.44 | 469 / 469 |
| 64 | 64 | 18,793 | 80.0% | 645.2 | 4,554.7 | 3,953.6 | 569.3 | 137,764.7 | 90.09 | 11.10 | 1.69 | 890 / 890 |
| 128 | 128 | 18,796 | 75.2% | 670.0 | 6,848.4 | 5,418.3 | 856.1 | 205,743.3 | 69.19 | 14.45 | 1.85 | 1,339 / 1,339 |

## 3. 1P1D (8 chips): empirical ISL sweep, ~95% prefix cache hit

| Sessions (n) | Trunks | Mean ISL | Cache hit | Mean OSL | Win Out tok/s | Drain Out tok/s | Out tok/s per chip | Total In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 8 | 40 | 17,986 | 95.5% | 482.1 | 809.6 | 783.7 | 101.2 | 30,728.0 | 111.79 | 8.95 | 0.32 | 155 / 155 |
| 32 | 40 | 17,986 | 95.5% | 480.5 | 2,592.8 | 1,962.8 | 324.1 | 100,084.7 | 89.60 | 11.16 | 0.85 | 507 / 507 |
| 64 | 64 | 17,905 | 95.5% | 477.0 | 4,797.8 | 3,480.1 | 599.7 | 186,863.4 | 86.17 | 11.61 | 1.04 | 950 / 950 |
| 128 | 128 | 17,934 | 94.4% | 484.3 | 7,446.2 | 6,082.5 | 930.8 | 287,198.9 | 69.10 | 14.47 | 1.27 | 1,462 / 1,462 |

## 4. 2P2D (16 chips): empirical ISL sweep, ~80% prefix cache hit

| Sessions (n) | Trunks | Mean ISL | Cache hit | Mean OSL | Win Out tok/s | Drain Out tok/s | Out tok/s per chip | Total In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 16 | 80 | 18,895 | 80.0% | 675.3 | 1,674.2 | 1,579.6 | 104.6 | 47,829.6 | 120.31 | 8.31 | 0.89 | 308 / 308 |
| 32 | 80 | 18,770 | 80.0% | 667.2 | 3,039.9 | 2,724.0 | 190.0 | 86,874.6 | 111.90 | 8.94 | 1.11 | 561 / 561 |
| 64 | 80 | 18,771 | 80.0% | 653.6 | 4,905.3 | 4,259.7 | 306.6 | 146,333.2 | 94.50 | 10.58 | 1.46 | 948 / 948 |
| 128 | 128 | 18,789 | 80.0% | 672.1 | 9,396.6 | 7,727.0 | 587.3 | 270,202.5 | 90.81 | 11.01 | 1.68 | 1,747 / 1,747 |
| 256 | 256 | 18,812 | 76.8% | 691.3 | 14,389.6 | 11,036.7 | 899.4 | 410,231.0 | 71.15 | 14.05 | 1.85 | 2,675 / 2,675 |

## 5. 2P2D (16 chips): empirical ISL sweep, ~95% prefix cache hit

| Sessions (n) | Trunks | Mean ISL | Cache hit | Mean OSL | Win Out tok/s | Drain Out tok/s | Out tok/s per chip | Total In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 16 | 80 | 17,912 | 95.5% | 482.9 | 1,641.0 | 1,536.9 | 102.6 | 62,276.0 | 114.89 | 8.70 | 0.33 | 316 / 316 |
| 64 | 80 | 17,950 | 95.5% | 477.1 | 5,059.1 | 4,531.8 | 316.2 | 195,217.4 | 89.54 | 11.17 | 0.75 | 995 / 995 |
| 128 | 128 | 17,929 | 95.5% | 501.7 | 9,527.9 | 7,835.8 | 595.5 | 352,107.2 | 85.45 | 11.70 | 1.04 | 1,797 / 1,797 |
| 256 | 256 | 17,905 | 94.7% | 487.8 | 14,541.5 | 10,887.3 | 908.8 | 563,251.7 | 67.55 | 14.80 | 1.29 | 2,874 / 2,874 |

### Scaling, 1P1D → 2P2D (agentic sweep, same load per pair)

| Cache target | 1P1D (n=128) | 2P2D (n=256) | Scaling | Per-chip output (1P1D → 2P2D) |
|:---|:---|:---|:---|:---|
| ~95% | 7,446.2 tok/s | 14,541.5 tok/s | 1.95× | 930.8 → 908.8 |
| ~80% | 6,848.4 tok/s (75.2% hit) | 14,389.6 tok/s (76.8% hit) | 2.10× | 856.1 → 899.4 |

*2P2D runs two independent 1P1D pairs behind one router. It adds capacity, but per-chip efficiency stays about the same.*

---

## 6. Fixed-shape probes (InferenceX-style, 1P1D vs. 2P2D)

**Total tok/s** = (input + output) tokens per second, computed as output tok/s × (ISL + OSL) ÷ OSL.

| Topology | Chips | Shape (ISL : OSL) | Concurrency | Output tok/s | Out tok/s per chip | Total tok/s | Med stream tok/s | Med TPOT (ms) | Med TTFT (ms) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1P1D | 8 | 1024 : 512 | 1 | 136.55 | 17.1 | 409.66 | 144.93 | 6.94 | 250.85 | 4 / 4 |
| 1P1D | 8 | 1024 : 512 | 8 | 898.42 | 112.3 | 2,695.25 | 136.97 | 7.30 | 471.72 | 16 / 16 |
| 1P1D | 8 | 4096 : 1024 | 8 | 874.48 | 109.3 | 4,372.38 | 166.16 | 6.02 | 843.16 | 16 / 16 |
| 1P1D | 8 | 4096 : 1024 | 64 | 5,224.30 | 653.0 | 26,121.51 | 139.37 | 7.18 | 2,102.42 | 64 / 64 |
| 2P2D | 16 | 1024 : 512 | 16 | 1,721.48 | 107.6 | 5,164.45 | 139.86 | 7.15 | 420.25 | 32 / 32 |
| 2P2D | 16 | 4096 : 1024 | 16 | 2,076.33 | 129.8 | 10,381.64 | 164.92 | 6.06 | 1,213.43 | 32 / 32 |
| 2P2D | 16 | 4096 : 1024 | 128 | 8,251.62 | 515.7 | 41,258.09 | 136.93 | 7.30 | 2,324.55 | 128 / 128 |

**Scaling on fixed shapes (1P1D → 2P2D at matched load per pair):** 1024:512, c=8 → 16: **1.92×**. 4096:1024, c=64 → 128: **1.58×**.

> ⚠️ Fixed-shape probes completed only 4–128 requests each (as few as 4–16 at low concurrency), so treat low-concurrency rows as indicative.

---

## Notes and limitations

- **Correctness gate failed** (Section 1). Investigate before treating these results as validated.
- **Speculative-decoding acceptance was not recorded in the sweep tables.** The two probes show mean accept lengths of 2.08 and 2.69.
- **Prefix cache is synthetic:** conversations reuse pre-seeded shared "trunks." Above ~128 distinct prefixes per prefill node, the realized hit rate drops (Sections 2 and 4, n=128/256).
- **Per-user speed falls as concurrency rises.** Highest-throughput rows run at ~68–71 tok/s per user.
- Private IP addresses and GKE node hostnames from the raw run log have been removed.
