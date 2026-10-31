# DeepSeek-V4.1-Flash-1017 (`1P1D` 8-Chip & `2P2D` 16-Chip) TPU v7x Results — `20261031_1p1d_2p2d`

- **Updated (UTC):** `2026-10-31T02:45:45.825414+00:00`
- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`MXFP4` MoE + `CED=1` on Prefill + `DSpark K=5` on Decode + `TPURaidenConnector` Stage 3)
- **Nodes (`2x2x4` = 4x `2x2x1` hosts, 16 `v7x` chips / 32 TensorCores):** `{"0": {"idx": 0, "role": "prefill", "pair": "pairA", "run_id": "pairA_20261031_1p1d_2p2d", "ip": "192.168.100.19", "hostname": "gke-tpu-b44c435a-w61p.us-central1-c.c.YOUR_GCP_PROJECT_ID.internal"}, "1": {"idx": 1, "role": "decode", "pair": "pairA", "run_id": "pairA_20261031_1p1d_2p2d", "ip": "192.168.100.16", "hostname": "gke-tpu-b44c435a-6d34.us-central1-c.c.YOUR_GCP_PROJECT_ID.internal"}, "2": {"idx": 2, "role": "prefill", "pair": "pairB", "run_id": "pairB_20261031_1p1d_2p2d", "ip": "192.168.100.14", "hostname": "gke-tpu-b44c435a-5cch.us-central1-c.c.YOUR_GCP_PROJECT_ID.internal"}, "3": {"idx": 3, "role": "decode", "pair": "pairB", "run_id": "pairB_20261031_1p1d_2p2d", "ip": "192.168.100.9", "hostname": "gke-tpu-b44c435a-kxr6.us-central1-c.c.YOUR_GCP_PROJECT_ID.internal"}}`
- **Cold Boot Time:** `1590.2 s`

## 1. `1P1D` (8 Chips / 16 TensorCores) Correctness & Micro-Latency Gates (`pd_gates.py`)
- **Hard Gates Passed (`2+2 x16` & `64-token greedy equality`):** `False`
- **USA 50-States (`PD` / `Local`):** `50` / `50`
- **4K/512 Streaming Probe (`c=8`):** `{"n": 16, "concurrency": 8, "prompt_tokens": 8192, "out_len": 128, "duration_s": 5.98, "ttft_mean_s": 0.8764, "ttft_p50_s": 0.6713, "ttft_max_s": 1.3909, "tpot_mean_ms": 13.134, "output_tok_s": 342.48, "input_tok_s": 21918.87, "prefill_active_window_s": 4.0312, "prefill_active_input_tok_s": 32514.12, "natural_mean_accept_len": 2.0771, "per_pos_accept_rates": [0.5866, 0.2372, 0.1461, 0.0611, 0.046], "steady_state_interior_window_tok_s": 521.61, "steady_state_interior_step_throughput_steps_s": 251.12, "steady_state_step_ms": 31.86, "steady_state_interior_throughput_at_accept_3p5_tok_s": 878.92, "steady_state_interior_throughput_at_accept_4p25_tok_s": 1067.26, "steady_state_interior_throughput_at_accept_5p0_tok_s": 1255.6, "wall_clock_step_throughput_steps_s": 164.88}`
- **4K/512 Streaming Probe (`c=64`):** `{"n": 64, "concurrency": 64, "prompt_tokens": 1024, "out_len": 512, "duration_s": 13.777, "ttft_mean_s": 2.4686, "ttft_p50_s": 2.9064, "ttft_max_s": 3.2357, "tpot_mean_ms": 16.098, "output_tok_s": 2378.51, "input_tok_s": 4757.01, "prefill_active_window_s": 3.247, "prefill_active_input_tok_s": 20183.69, "natural_mean_accept_len": 2.6861, "per_pos_accept_rates": [0.6331, 0.4231, 0.2807, 0.1996, 0.1496], "steady_state_interior_window_tok_s": 5801.96, "steady_state_interior_step_throughput_steps_s": 2159.99, "steady_state_step_ms": 29.63, "steady_state_interior_throughput_at_accept_3p5_tok_s": 7559.96, "steady_state_interior_throughput_at_accept_4p25_tok_s": 9179.96, "steady_state_interior_throughput_at_accept_5p0_tok_s": 10799.95, "wall_clock_step_throughput_steps_s": 885.49}`

## 2. `1P1D` (8 Chips) Empirical 50:20:12:9:9 ISL Sweep (~80% Prefix Cache)

| Sessions (`n`) | Trunks | Mean ISL | Cache Hit (%) | Mean OSL | Win Out tok/s | Drain Out tok/s | Win In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | DSpark Accept Len | OK/Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `40` | 20,297 | 79.9% | 782.9 | **117.5** | 118.0 | 3,213.8 | **126.58** | 7.90 | 0.50 | - | `19/19` |
| `8` | `40` | 18,692 | 80.0% | 600.9 | **771.2** | 723.3 | 25,078.0 | **119.50** | 8.37 | 0.91 | - | `161/161` |
| `16` | `40` | 18,718 | 80.0% | 774.4 | **1,522.4** | 1,230.0 | 39,921.5 | **110.20** | 9.07 | 1.09 | - | `258/258` |
| `32` | `40` | 18,745 | 80.0% | 668.8 | **2,511.0** | 2,258.1 | 72,133.2 | **94.28** | 10.61 | 1.44 | - | `469/469` |
| `64` | `64` | 18,793 | 80.0% | 645.2 | **4,554.7** | 3,953.6 | 137,764.7 | **90.09** | 11.10 | 1.69 | - | `890/890` |
| `128` | `128` | 18,796 | 75.2% | 670.0 | **6,848.4** | 5,418.3 | 205,743.3 | **69.19** | 14.45 | 1.85 | - | `1339/1339` |

## 3. `1P1D` (8 Chips) Empirical 50:20:12:9:9 ISL Sweep (~95% Prefix Cache)

| Sessions (`n`) | Trunks | Mean ISL | Cache Hit (%) | Mean OSL | Win Out tok/s | Drain Out tok/s | Win In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | DSpark Accept Len | OK/Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8` | `40` | 17,986 | 95.5% | 482.1 | **809.6** | 783.7 | 30,728.0 | **111.79** | 8.95 | 0.32 | - | `155/155` |
| `32` | `40` | 17,986 | 95.5% | 480.5 | **2,592.8** | 1,962.8 | 100,084.7 | **89.60** | 11.16 | 0.85 | - | `507/507` |
| `64` | `64` | 17,905 | 95.5% | 477.0 | **4,797.8** | 3,480.1 | 186,863.4 | **86.17** | 11.61 | 1.04 | - | `950/950` |
| `128` | `128` | 17,934 | 94.4% | 484.3 | **7,446.2** | 6,082.5 | 287,198.9 | **69.10** | 14.47 | 1.27 | - | `1462/1462` |

## 4. `2P2D` (16 Chips / Full Slice) Empirical 50:20:12:9:9 ISL Sweep (~80% Prefix Cache)

| Sessions (`n`) | Trunks | Mean ISL | Cache Hit (%) | Mean OSL | Win Out tok/s | Drain Out tok/s | Win In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | DSpark Accept Len | OK/Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `16` | `80` | 18,895 | 80.0% | 675.3 | **1,674.2** | 1,579.6 | 47,829.6 | **120.31** | 8.31 | 0.89 | - | `308/308` |
| `32` | `80` | 18,770 | 80.0% | 667.2 | **3,039.9** | 2,724.0 | 86,874.6 | **111.90** | 8.94 | 1.11 | - | `561/561` |
| `64` | `80` | 18,771 | 80.0% | 653.6 | **4,905.3** | 4,259.7 | 146,333.2 | **94.50** | 10.58 | 1.46 | - | `948/948` |
| `128` | `128` | 18,789 | 80.0% | 672.1 | **9,396.6** | 7,727.0 | 270,202.5 | **90.81** | 11.01 | 1.68 | - | `1747/1747` |
| `256` | `256` | 18,812 | 76.8% | 691.3 | **14,389.6** | 11,036.7 | 410,231.0 | **71.15** | 14.05 | 1.85 | - | `2675/2675` |

## 5. `2P2D` (16 Chips / Full Slice) Empirical 50:20:12:9:9 ISL Sweep (~95% Prefix Cache)

| Sessions (`n`) | Trunks | Mean ISL | Cache Hit (%) | Mean OSL | Win Out tok/s | Drain Out tok/s | Win In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | DSpark Accept Len | OK/Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `16` | `80` | 17,912 | 95.5% | 482.9 | **1,641.0** | 1,536.9 | 62,276.0 | **114.89** | 8.70 | 0.33 | - | `316/316` |
| `64` | `80` | 17,950 | 95.5% | 477.1 | **5,059.1** | 4,531.8 | 195,217.4 | **89.54** | 11.17 | 0.75 | - | `995/995` |
| `128` | `128` | 17,929 | 95.5% | 501.7 | **9,527.9** | 7,835.8 | 352,107.2 | **85.45** | 11.70 | 1.04 | - | `1797/1797` |
| `256` | `256` | 17,905 | 94.7% | 487.8 | **14,541.5** | 10,887.3 | 563,251.7 | **67.55** | 14.80 | 1.29 | - | `2874/2874` |

## 6. Fixed-Shape `InferenceX` (`4096:1024`) & Cold-Cache (`1024:512`) Probes (`1P1D` vs `2P2D`)

| Topology | Chips | Shape (`ISL:OSL`) | Concurrency | Output tok/s | Total tok/s | Med Stream tok/s | Med TPOT (ms) | Med TTFT (ms) | OK/Total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `1P1D` | `8` | `1024:512` | `1` | **136.55** | 409.66 | **144.93** | 6.94 | 250.85 | `4/4` |
| `1P1D` | `8` | `1024:512` | `8` | **898.42** | 2,695.25 | **136.97** | 7.3 | 471.72 | `16/16` |
| `1P1D` | `8` | `4096:1024` | `8` | **874.48** | 4,372.38 | **166.16** | 6.02 | 843.16 | `16/16` |
| `1P1D` | `8` | `4096:1024` | `64` | **5,224.30** | 26,121.51 | **139.37** | 7.18 | 2102.42 | `64/64` |
| `2P2D` | `16` | `1024:512` | `16` | **1,721.48** | 5,164.45 | **139.86** | 7.15 | 420.25 | `32/32` |
| `2P2D` | `16` | `4096:1024` | `16` | **2,076.33** | 10,381.64 | **164.92** | 6.06 | 1213.43 | `32/32` |
| `2P2D` | `16` | `4096:1024` | `128` | **8,251.62** | 41,258.09 | **136.93** | 7.3 | 2324.55 | `128/128` |

