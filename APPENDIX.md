# Appendix: Complete Disaggregated Configurations, Concurrency Sweeps, and Routing Ablations (`DeepSeek-V4.1-Flash-1017` on TPU v7x)

This document preserves all detailed `vllm serve` runtime configurations, `EP=8` vs. `EP=16` decode ablations, `1P3D` vs. `2P2D` concurrency ladders, `llm-d` prefix-aware routing experiments, scheduler-cap fixes, and external comparison caveats. For headline results and quickstart commands, see [`README.md`](./README.md).

---

## 1. Key Engineering Fixes & Serving Configuration

### A. Bit-Exact `P/D` Greedy Parity (`PYTHONHASHSEED=0`)
Initial `1P1D` / `2P2D` runs ([`results/initial_1p1d_2p2d/`](./results/initial_1p1d_2p2d/)) failed strict 64-token greedy equality between disaggregated `P/D` and a single local instance. Setting `PYTHONHASHSEED=0` consistently across prefill and decode workers resolved hash-dependent block/metadata ordering; subsequent `llm-d` runs passed **20 / 20 bit-exact 64-token greedy prompts** and **16 / 16 QA checks** (`DSpark τ = 5.16` structured / `2.41` open-ended; [`results/llmd_disaggregated/greedy_correctness_gate.json`](./results/llmd_disaggregated/greedy_correctness_gate.json)).

### B. `DSpark` (`K=5`) Scheduler-Cap Fix (`16/96` Collapse vs. `48/288`)
Running `DSpark` (`num_speculative_tokens = 5`) with `--max-num-seqs 16 --max-num-batched-tokens 96` caused draft-token acceptance to collapse (`τ = 1.000`, `2,185 tok/s` across 8 chips; [`results/agentic_xpyd_scaling/1p1d_cap16_96_dspark_collapse.json`](./results/agentic_xpyd_scaling/1p1d_cap16_96_dspark_collapse.json)) because `16 × (1 + 5) = 96` left zero headroom for multi-token verification padding. Raising decode caps to `--max-num-seqs 48 --max-num-batched-tokens 288` with explicit XLA buckets `[16, 40, 48, 64, 96, 192, 288]` restored `τ ≈ 2.9–4.2` and `>3.4×` throughput.

### C. Canonical Prefill & Decode Flags (Per 4-Chip `v7x` Host = `8` TensorCores)
- **Decode (`DP=8, EP=8`):**
  ```bash
  PYTHONHASHSEED=0 CED=0 vllm serve /dev/shm/DeepSeek-V4.1-Flash \
    --port 9400 --api-server-count 8 \
    --data-parallel-size 8 --data-parallel-size-local 8 --tensor-parallel-size 1 \
    --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 \
    --max-num-seqs 48 --max-num-batched-tokens 288 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --async-scheduling \
    --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_consumer"}' \
    --compilation-config '{"compile_sizes":[16,40,48,64,96,192,288]}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
  ```
- **Prefill (`DP=8, EP=8`):**
  ```bash
  PYTHONHASHSEED=0 CED_PREFILL=1 vllm serve /dev/shm/DeepSeek-V4.1-Flash \
    --port 8000 --api-server-count 8 \
    --data-parallel-size 8 --data-parallel-size-local 8 --tensor-parallel-size 1 \
    --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 \
    --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching --prefix-cache-retention-interval 1024 --no-async-scheduling \
    --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_producer"}' \
    --compilation-config '{"compile_sizes":[512,2048,8192]}'
  ```
*(Full kernel and `TPURaidenConnector` environment variables are in [`scripts/launch_pd_role.sh`](./scripts/launch_pd_role.sh) and [`scripts/llmd/deploy_ep8_decode.py`](./scripts/llmd/deploy_ep8_decode.py).)*

---

## 2. Why Single-Host `EP=8` Decode Outperformed 2-Host `EP=16` Decode

On `16` physical `v7x` chips (`8` prefill + `8` decode chips) running `2048 : 1024` uncached prompts ([`results/llmd_disaggregated/ep16_decode_2048_1024/`](./results/llmd_disaggregated/ep16_decode_2048_1024/), [`results/llmd_disaggregated/ep8_2p2d_2048_1024/`](./results/llmd_disaggregated/ep8_2p2d_2048_1024/)), two independent 4-chip `EP=8` decode pods beat one 8-chip `EP=16` `LeaderWorkerSet` decode instance by **`+13.3%` to `+46.3%`** at every concurrency level:

- **Matched interactivity (`~61 tok/s/user`):** `2 × EP=8` (`c=768`) achieved **`42,719 tok/s` (`5,340 tok/s/decode chip`)** at `61.3 tok/s/user` vs. **`22,051 tok/s` (`2,756 tok/s/decode chip`)** for `1 × EP=16` (`c=384`) — **`1.94×` higher throughput**.
- **Peak throughput (`c=1024`):** `2 × EP=8` achieved **`46,823 tok/s` (`5,853 tok/s/decode chip`)** vs. **`40,907 tok/s` (`5,113 tok/s/decode chip`)** for `1 × EP=16` (`+14.5%`).
- **Root cause:** `EP=16` spans two `v7x` hosts across an inter-host ICI hop (`TPOT 18.4 ms` vs. `16.1 ms` at matched batch) and couples all 16 DP ranks to one step barrier, so whenever any single rank steps up an XLA token bucket (`96 → 192` or `192 → 288`), all 16 ranks pay the larger bucket latency. (`EP=16` only helps when a workload requires `>64M` live `FP8` KV tokens across 8 decode chips: `99.9M` vs. `63.7M` tokens.)

---

## 3. Full `1P3D` vs. `2P2D` Ratio Concurrency Sweep (`2048 : 1024`, `16` Chips)

Reassigning one 4-chip host from prefill to decode (`1P3D` = `4P + 12D` chips vs. `2P2D` = `8P + 8D` chips; [`results/llmd_disaggregated/ep8_1p3d_2048_1024/`](./results/llmd_disaggregated/ep8_1p3d_2048_1024/)) increased output per **total** chip at every concurrency from `c=128` upward:

| Concurrency | `1P3D` Output `tok/s` | `1P3D` / Decode Chip (`/12`) | `1P3D` / Total Chip (`/16`) | `1P3D` User `Med / P90_low` | `1P3D` `TTFT p50 / p95` (s) | `DSpark` $\tau$ | `2P2D` / Total Chip (`/16`) | `1P3D` vs. `2P2D` (`/16`) |
| ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| 32 | 4,200.5 | 350.0 | 262.5 | 146.3 / 116.9 | 0.42 / 0.98 | 3.70 | 259.3 | +1.2% |
| 64 | 7,773.6 | 647.8 | 485.9 | 135.5 / 118.0 | 0.87 / 1.33 | 3.92 | 502.8 | -3.4% |
| 128 | 14,951.4 | 1,246.0 | 934.5 | 133.7 / 116.0 | 0.98 / 1.82 | 4.05 | 847.7 | +10.2% |
| 256 | 22,459.5 | 1,871.6 | 1,403.7 | 103.0 / 88.7 | 1.21 / 2.11 | 4.11 | 1,183.2 | +18.6% |
| 384 | 32,370.4 | 2,697.5 | 2,023.2 | 97.2 / 81.5 | 1.23 / 2.00 | 4.14 | 1,638.3 | **+23.5%** |
| 512 | 33,960.5 | 2,830.1 | 2,122.5 | 73.8 / 65.8 | 1.30 / 2.26 | 4.14 | 2,020.7 | +5.0% |
| 768 | **49,512.4** | 4,126.0 | **3,094.5** | **79.0 / 68.6** | 1.99 / 4.14 | 4.15 | 2,669.9 | **+15.9%** |
| 1,024 | **51,134.9** | 4,261.2 | **3,195.9** | 76.5 / 58.5 | 6.98 / 8.27 | 4.17 | 2,926.4 | **+9.2%** |

*Bottleneck transition:* Above `c=768`, the single 4-chip prefill pod in `1P3D` saturates at `~101–105k` uncached prompt `tok/s` (`TTFT p50` rises from `1.99 s` to `6.98 s`). Conversely, on prefill-heavy `8192 : 1024` prompts (`8:1` input:output), even `2P2D` (`~215k` prompt `tok/s` across 8 prefill chips) is prefill-limited (`26,229 tok/s` at `c=512`), so `2P2D` or higher prefill allocation is preferred there.

---

## 4. `llm-d` Prefix-Aware Routing & DP-Rank Affinity Ablation (`2P2D`, `~18.8K` ISL, `80%` Target Hit)

Evaluated across 4 routing configurations on `256` distinct conversation trunks ([`results/llmd_disaggregated/agentic_routing_ablation/`](./results/llmd_disaggregated/agentic_routing_ablation/)):

| Routing Policy | Sessions `n` | Realized Prefix Hit | `TTFT p50 / p95` (s) | Output `tok/s` | `tok/s` / Total Chip (`/16`) | Per-User `Med / P90_low` |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- |
| **(a) Cache-blind (`token-load`)** | 64 / 128 / 256 | 1.6% / 2.1% / 2.7% | 2.32 / 7.03 · 2.92 / 8.47 · 10.84 / 19.37 | 4,504 / 8,190 / 9,762 | 281.5 / 511.8 / 610.2 | 95.4 / 91.6 / 89.3 |
| **(b) `llm-d` prefix-aware (pod-level only)** | 64 / 128 / 256 | 2.9% / 2.1% / 3.2% | 2.35 / 7.04 · 3.12 / 8.95 · 10.90 / 18.74 | 4,461 / 7,968 / 9,642 | 278.8 / 498.0 / 602.6 | 94.2 / 90.3 / 87.6 |
| **(c) `llm-d` prefix-aware + `X-data-parallel-rank`** | 64 / 128 / 256 | **45.7% / 68.1% / 63.0%** | **1.83 / 7.05 · 1.78 / 30.88 · 1.83 / 167.8** | 4,585 / 7,231 / 5,940 | 286.6 / 451.9 / 371.3 | 91.9 / 82.9 / 86.5 |
| **(d) Cache-blind + `X-data-parallel-rank`** | 64 / 128 / 256 | 29.0% / 48.3% / 49.9% | 1.92 / 7.01 · 1.82 / 44.13 · 1.88 / 99.13 | 4,458 / 5,722 / 6,769 | 278.6 / 357.6 / 423.1 | 93.5 / 90.9 / 86.8 |

**Finding:** Without DP-rank affinity (`(a)` vs. `(b)`), `EndpointPicker` routes requests to the right 4-chip prefill pod, only for vLLM's internal `DP=8` balancer to scatter turns across 8 rank-local prefix caches (`<3.3%` hit rate). Supplying `X-data-parallel-rank` (`(c)` vs. `(d)`) allows `llm-d` prefix-aware routing to lift realized cache hits by **+13.1 to +19.8 percentage points** (`68.1%` vs. `48.3%` at `n=128`) and cut `TTFT p50` to `1.78–1.83 s`, though static modulo rank pinning creates tail-latency skew (`p95`) when rank loads diverge or when decode ranks also inherit the header. Exposing each prefill DP rank as an independent `InferencePool` endpoint (or pinning prefill ranks while load-balancing decode ranks) resolves both issues.

---

## 5. Transparent Comparison Caveats vs. Published `GB200 NVL72` Numbers

In `README.md`, we cite the public LMSYS `GB200 NVL72` figure of **`13,386` output tok/s per decode GPU** (`FP8` attention + `NVFP4` MoE; earlier LMSYS posts report `9,087` for `BF16` attention + `FP8` MoE and `7,583` for `FP8`). Our highest `v7x` decode-chip measurement at `2048 : 1024` is **`5,852.9` output tok/s per decode chip** (`43.7%` per decode accelerator at `c=1024`, `62.2 tok/s/user`), or **`5,339.9` tok/s per decode chip** (`39.9%` at `c=768`, `61.3 tok/s/user`, `TTFT p50 1.28 s`). Key architectural and setup differences:

1. **Expert-parallel scale (`EP=8` on 4 chips vs. `EP=48` on 48 GPUs):** Our decode instance spans a single 4-chip `v7x` host (`EP=8`), whereas `GB200 NVL72` spans 48 GPUs inside a single NVLink domain (`EP=48`).
2. **Model architecture:** Our measurements run `DeepSeek-V4.1-Flash-1017` (`384` routed `MXFP4` `W4A8` experts, `top-6` routing, sparse `CSA` MLA, `DSpark K=5` speculative decoding), whereas the LMSYS `GB200` post benchmarks DeepSeek-V3/R1 (`256` routed `NVFP4` experts, `top-8` routing).
3. **Decode slot cap:** Our `v7x` runs capped `--max-num-seqs 48` (`--max-num-batched-tokens 288`) per DP rank, leaving `>85%` of `FP8` KV-cache HBM unused at peak concurrency.
