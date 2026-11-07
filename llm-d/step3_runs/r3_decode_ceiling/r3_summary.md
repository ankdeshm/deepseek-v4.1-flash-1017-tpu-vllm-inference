# Step 3 (`R3`) Summary Report — Decode-Only Ceiling (`S1 = 2048 : 1024`, `~99.5%` Prefix Hit on `2P2D` `EP=8`)

- **Cluster:** `ank-tpu-v7x` (`us-central1`, project `YOUR_GCP_PROJECT`)
- **GCS Artifacts:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/step3_runs/r3_decode_ceiling/`
- **Topology:** `llm-d` `v1.0.0` Path A (`2 x 4-chip` `DP=8, EP=8` Prefill on Pool A + `2 x 4-chip` `DP=8, EP=8` Decode on Pool B = `8` Decode chips / `16` total `tpu7x` chips)
- **Decode Isolation Protocol:** `16` `S1` trunks (`>= 2,050` shared prefix tokens = `2 x 1,024` pinned hybrid KV + Mamba/CSA cache blocks) explicitly pre-warmed across **all `8` DP ranks (`0..7`) on BOTH Prefill pods (`llmd-prefill-poola-h0` & `h1`)** (`256/256` direct rank warmups + `32/32` router warmups), achieving **`~99.5%` API prefix cache hit** and keeping Pool A Prefill `<5%` utilized so Pool B Decode is 100% isolated from Prefill queueing.
- **Reliability:** **`0 / 47,647` failed requests (`0.000%` error rate)** and **`0` preemptions** across all levels.
- **Generated At (UTC):** `2026-11-07T05:53:00Z`

---

## 1. Complete Rule 4 Metrics Table (`R3` Decode-Only Ceiling)

| Concurrency (`c`) | Steady Window | Steady Out tok/s (Gain) | Out tok/s / Decode Chip (`/8`) | Out tok/s / Total Chip (`/16`) | Server Gen tok/s (`60-290s`) | API Cache Hit % & `P_run` | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\tau$ (`Full` / `60-290s`) | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Pod Split (`P h0/h1` · `D b0/b1`) | Preempt (`P / D`) | Failures |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---|---|---|---:|---:|
| **`256`** | `300.0 s` | **`15,393.99`** *(base)* | **`1,924.25`** | **`962.12`** | `18310.32` | `99.33%` (`P_run=0.00`) | `76.22 / 51.54 / 41.62` \| `75.76 / 45.50 / 29.30` | `0.430 / 10.878 / 28.974` | `0.431 / 12.204 / 28.975` | `13.20 / 34.13 / 56.81` | **`4.1357`** / `4.1204` | `15.51 / 16.0 / 20.0` (`248.2` run, `2.2` wait) | `D`: `3.420% / 4.693%` · `P`: `2.646% / 3.927%` | `P`:`50.0/50.0%` · `D`:`48.3/51.7%` | `0 / 0` | **`0 / 4,642`** |
| **`384`** | `300.0 s` | **`26,196.26`** (`+70.17%`) | **`3,274.53`** | **`1,637.27`** | `26322.39` | `99.33%` (`P_run=0.00`) | `71.71 / 64.38 / 59.61` \| `71.81 / 64.42 / 59.67` | `0.459 / 0.614 / 1.209` | `0.460 / 0.625 / 1.212` | `13.93 / 16.76 / 22.17` | **`4.1526`** / `4.164` | `23.39 / 24.0 / 27.0` (`374.2` run, `3.4` wait) | `D`: `5.161% / 5.990%` · `P`: `0.039% / 0.589%` | `P`:`50.0/50.0%` · `D`:`50.1/49.9%` | `0 / 0` | **`0 / 7,875`** |
| **`512`** | `300.0 s` | **`28,466.99`** (`+8.67%`) | **`3,558.37`** | **`1,779.19`** | `31311.18` | `99.33%` (`P_run=0.00`) | `63.17 / 39.15 / 35.33` \| `63.41 / 39.46 / 35.88` | `0.488 / 1.070 / 3.665` | `0.488 / 1.103 / 3.691` | `15.77 / 27.87 / 32.02` | **`4.1462`** / `4.1598` | `31.17 / 31.0 / 34.0` (`498.8` run, `4.4` wait) | `D`: `6.867% / 7.822%` · `P`: `0.052% / 0.589%` | `P`:`50.0/50.0%` · `D`:`49.9/50.1%` | `0 / 0` | **`0 / 8,606`** |
| **`768`** | `300.0 s` | **`42,866.27`** (`+50.58%`) | **`5,358.28`** | **`2,679.14`** | `43238.86` | `99.33%` (`P_run=0.00`) | `59.03 / 52.81 / 48.23` \| `59.11 / 52.82 / 48.18` | `0.546 / 1.441 / 2.234` | `0.548 / 1.508 / 2.278` | `16.92 / 20.75 / 26.67` | **`4.1483`** / `4.1651` | `46.60 / 47.0 / 48.0` (`745.7` run, `8.7` wait) | `D`: `10.296% / 11.198%` · `P`: `0.085% / 0.785%` | `P`:`50.0/50.0%` · `D`:`49.9/50.1%` | `0 / 0` | **`0 / 12,948`** |
| **`1024`** **STOP** | `300.0 s` | **`44,100.66`** (`+2.88%`) | **`5,512.58`** | **`2,756.29`** | `44395.98` | `99.28%` (`P_run=0.00`) | `59.40 / 52.93 / 47.75` \| `59.49 / 53.02 / 47.87` | `5.585 / 11.998 / 17.477` | `5.511 / 12.338 / 17.608` | `16.81 / 20.89 / 26.71` | **`4.1362`** / `4.1442` | `47.78 / 48.0 / 48.0` (`764.4` run, `246.9` wait) | `D`: `12.019% / 14.615%` · `P`: `1.351% / 2.258%` | `P`:`49.0/51.0%` · `D`:`50.1/49.9%` | `0 / 0` | **`0 / 13,576`** |

---

## 2. `R3` (Cached Prefill -> Pure Decode Ceiling) vs `R1b` (`0.00%` Uncached Prefill) Comparison

| Concurrency (`c`) | `R1b` (`0%` Hit) Out tok/s (`/8` Dec) & `D_run` | `R1b` `TTFT P50` (s) & `tau` | `R3` (`~99.5%` Hit) Out tok/s (`/8` Dec) & `D_run` | `R3` `TTFT P50` (s) & `tau` | `R3` vs `R1b` Out tok/s |
|---:|---|---|---|---:|---:|
| **`256`** | `18,931.05` (`2,366.38`) · `235.8/768` | `1.088s` · `τ=4.13` | **`15,393.99`** (**`1,924.25`**) · `248.2/768` | `0.430s` · `τ=4.14` | **`-18.68%`** |
| **`384`** | `26,213.37` (`3,276.67`) · `356.3/768` | `1.153s` · `τ=4.14` | **`26,196.26`** (**`3,274.53`**) · `374.2/768` | `0.459s` · `τ=4.15` | **`-0.07%`** |
| **`512`** | `32,330.91` (`4,041.36`) · `471.0/768` | `1.238s` · `τ=4.17` | **`28,466.99`** (**`3,558.37`**) · `498.8/768` | `0.488s` · `τ=4.15` | **`-11.95%`** |
| **`768`** | `42,719.11` (`5,339.89`) · `714.7/768` | `1.284s` · `τ=4.18` | **`42,866.27`** (**`5,358.28`**) · `745.7/768` | `0.546s` · `τ=4.15` | **`+0.34%`** |
| **`1024`** | `46,822.96` (`5,852.87`) · `764.3/768` | `5.537s` · `τ=4.14` | **`44,100.66`** (**`5,512.58`**) · `764.4/768` | `5.585s` · `τ=4.14` | **`-5.81%`** |

---

## 3. Decode-Ceiling Physics & `tpu-info` Hardware Telemetry

- **Pure Pool B Decode Ceiling (`c=1024`):** **`44,100.66` out tok/s** (**`5,512.58` out tok/s / decode chip (`/8`)**, **`2,756.29` out tok/s / total chip (`/16`)**, server generation rate `44395.98` tok/s) with Pool A Prefill averaging only **`0.00` active sequences** across `16` Prefill DP ranks.
- **Exact Saturation Mechanism:** Once global Decode active sequences pin `max_num_seqs = 48` x `16` DP ranks = **`768` slots** (`764.4/768` = `99.53%` slot occupancy), every DP rank executes the maximum `288`-token (`48 x 6`) compiled XLA `DSpark` verification graph back-to-back while Decode KV HBM remains at **`12.02%` steady / `14.62%` peak** (`>87%` free KV HBM, `0` preemptions).

## 4. Read-Only `tpu-info` Snapshot on `llmd-decode-poolb-h0`

### `tpu_info_r3_c1024.txt`
```text
=== tpu-info snapshot (r3_c1024) rc=0 ===
Libtpu version: 0.0.48                                                          
Accelerator type: 7x                                                            
                                                                                
TPU Chips                                          
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip        ┃ Type       ┃ Devices ┃ PID        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ /dev/vfio/1 │ TPU7x chip │ 2       │ 2873, 2878 │
│ /dev/vfio/2 │ TPU7x chip │ 2       │ 2884, 2887 │
│ /dev/vfio/3 │ TPU7x chip │ 2       │ 2869, 2875 │
│ /dev/vfio/4 │ TPU7x chip │ 2       │ 2881, 2866 │
└─────────────┴────────────┴─────────┴────────────┘
TPU Runtime Utilization                      
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 73.34 GiB / 94.74 GiB │ 47.99%     │
│ 0    │ 73.44 GiB / 94.74 GiB │ 47.47%     │
│ 1    │ 73.44 GiB / 94.74 GiB │ 47.74%     │
│ 1    │ 73.44 GiB / 94.74 GiB │ 47.81%     │
│ 2    │ 73.44 GiB / 94.74 GiB │ 48.05%     │
│ 2    │ 75.36 GiB / 94.74 GiB │ 47.41%     │
│ 3    │ 73.44 GiB / 94.74 GiB │ 48.46%     │
│ 3    │ 73.44 GiB / 94.74 GiB │ 47.96%     │
└──────┴───────────────────────┴────────────┘
TensorCore Utilization              
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 23.13%                 │
│ 1       │ 23.32%                 │
│ 2       │ 23.13%                 │
│ 3       │ 23.15%                 │
│ 4       │ 22.89%                 │
│ 5       │ 23.32%                 │
│ 6       │ 22.89%                 │
│ 7       │ 23.15%                 │
└─────────┴────────────────────────┘
╭─────────────────────── Buffer Transfer Latency Status ───────────────────────╮
│ WARNING: Buffer Transfer Latency metrics unavailable. Did you start a        │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────── gRPC TCP Minimum RTT Status ─────────────────────────╮
│ WARNING: gRPC TCP Minimum RTT metrics unavailable. Did you start a           │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── gRPC TCP Delivery Rate Status ────────────────────────╮
│ WARNING: gRPC TCP Delivery Rate metrics unavailable. Did you start a         │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
```
### `tpu_info_r3_c768.txt`
```text
=== tpu-info snapshot (r3_c768) rc=0 ===
Libtpu version: 0.0.48                                                          
Accelerator type: 7x                                                            
                                                                                
TPU Chips                                          
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip        ┃ Type       ┃ Devices ┃ PID        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ /dev/vfio/1 │ TPU7x chip │ 2       │ 2878, 2873 │
│ /dev/vfio/2 │ TPU7x chip │ 2       │ 2884, 2887 │
│ /dev/vfio/3 │ TPU7x chip │ 2       │ 2869, 2875 │
│ /dev/vfio/4 │ TPU7x chip │ 2       │ 2866, 2881 │
└─────────────┴────────────┴─────────┴────────────┘
TPU Runtime Utilization                      
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 73.44 GiB / 94.74 GiB │ 48.12%     │
│ 0    │ 73.14 GiB / 94.74 GiB │ 48.05%     │
│ 1    │ 73.44 GiB / 94.74 GiB │ 47.77%     │
│ 1    │ 73.44 GiB / 94.74 GiB │ 47.69%     │
│ 2    │ 73.44 GiB / 94.74 GiB │ 48.04%     │
│ 2    │ 75.36 GiB / 94.74 GiB │ 47.35%     │
│ 3    │ 73.44 GiB / 94.74 GiB │ 47.89%     │
│ 3    │ 73.44 GiB / 94.74 GiB │ 47.09%     │
└──────┴───────────────────────┴────────────┘
TensorCore Utilization              
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 22.93%                 │
│ 1       │ 23.10%                 │
│ 2       │ 22.93%                 │
│ 3       │ 23.18%                 │
│ 4       │ 22.68%                 │
│ 5       │ 23.10%                 │
│ 6       │ 22.68%                 │
│ 7       │ 23.18%                 │
└─────────┴────────────────────────┘
╭─────────────────────── Buffer Transfer Latency Status ───────────────────────╮
│ WARNING: Buffer Transfer Latency metrics unavailable. Did you start a        │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────── gRPC TCP Minimum RTT Status ─────────────────────────╮
│ WARNING: gRPC TCP Minimum RTT metrics unavailable. Did you start a           │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── gRPC TCP Delivery Rate Status ────────────────────────╮
│ WARNING: gRPC TCP Delivery Rate metrics unavailable. Did you start a         │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
```
### `tpu_info_s2_c1024.txt` (from `R2-S2`)
```text
=== tpu-info snapshot (s2_c1024) rc=0 ===
Libtpu version: 0.0.48                                                          
Accelerator type: 7x                                                            
                                                                                
TPU Chips                                          
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip        ┃ Type       ┃ Devices ┃ PID        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ /dev/vfio/1 │ TPU7x chip │ 2       │ 2873, 2878 │
│ /dev/vfio/2 │ TPU7x chip │ 2       │ 2884, 2887 │
│ /dev/vfio/3 │ TPU7x chip │ 2       │ 2869, 2875 │
│ /dev/vfio/4 │ TPU7x chip │ 2       │ 2866, 2881 │
└─────────────┴────────────┴─────────┴────────────┘
TPU Runtime Utilization                      
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 73.09 GiB / 94.74 GiB │ 47.98%     │
│ 0    │ 73.09 GiB / 94.74 GiB │ 47.80%     │
│ 1    │ 73.09 GiB / 94.74 GiB │ 47.59%     │
│ 1    │ 73.08 GiB / 94.74 GiB │ 47.93%     │
│ 2    │ 73.09 GiB / 94.74 GiB │ 48.13%     │
│ 2    │ 75.01 GiB / 94.74 GiB │ 47.96%     │
│ 3    │ 73.09 GiB / 94.74 GiB │ 47.75%     │
│ 3    │ 73.09 GiB / 94.74 GiB │ 47.55%     │
└──────┴───────────────────────┴────────────┘
TensorCore Utilization              
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 23.04%                 │
│ 1       │ 22.96%                 │
│ 2       │ 23.04%                 │
│ 3       │ 23.19%                 │
│ 4       │ 23.16%                 │
│ 5       │ 22.96%                 │
│ 6       │ 23.16%                 │
│ 7       │ 23.19%                 │
└─────────┴────────────────────────┘
╭─────────────────────── Buffer Transfer Latency Status ───────────────────────╮
│ WARNING: Buffer Transfer Latency metrics unavailable. Did you start a        │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────── gRPC TCP Minimum RTT Status ─────────────────────────╮
│ WARNING: gRPC TCP Minimum RTT metrics unavailable. Did you start a           │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── gRPC TCP Delivery Rate Status ────────────────────────╮
│ WARNING: gRPC TCP Delivery Rate metrics unavailable. Did you start a         │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
```
### `tpu_info_s2_c768.txt` (from `R2-S2`)
```text
=== tpu-info snapshot (s2_c768) rc=0 ===
Libtpu version: 0.0.48                                                          
Accelerator type: 7x                                                            
                                                                                
TPU Chips                                          
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip        ┃ Type       ┃ Devices ┃ PID        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ /dev/vfio/1 │ TPU7x chip │ 2       │ 2873, 2878 │
│ /dev/vfio/2 │ TPU7x chip │ 2       │ 2887, 2884 │
│ /dev/vfio/3 │ TPU7x chip │ 2       │ 2869, 2875 │
│ /dev/vfio/4 │ TPU7x chip │ 2       │ 2866, 2881 │
└─────────────┴────────────┴─────────┴────────────┘
TPU Runtime Utilization                      
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 73.09 GiB / 94.74 GiB │ 47.77%     │
│ 0    │ 73.09 GiB / 94.74 GiB │ 47.78%     │
│ 1    │ 73.09 GiB / 94.74 GiB │ 47.90%     │
│ 1    │ 73.09 GiB / 94.74 GiB │ 48.12%     │
│ 2    │ 75.01 GiB / 94.74 GiB │ 47.83%     │
│ 2    │ 73.09 GiB / 94.74 GiB │ 47.84%     │
│ 3    │ 73.09 GiB / 94.74 GiB │ 48.03%     │
│ 3    │ 73.09 GiB / 94.74 GiB │ 47.90%     │
└──────┴───────────────────────┴────────────┘
TensorCore Utilization              
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 23.22%                 │
│ 1       │ 23.14%                 │
│ 2       │ 23.22%                 │
│ 3       │ 23.27%                 │
│ 4       │ 23.14%                 │
│ 5       │ 23.27%                 │
└─────────┴────────────────────────┘
╭─────────────────────── Buffer Transfer Latency Status ───────────────────────╮
│ WARNING: Buffer Transfer Latency metrics unavailable. Did you start a        │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────── gRPC TCP Minimum RTT Status ─────────────────────────╮
│ WARNING: gRPC TCP Minimum RTT metrics unavailable. Did you start a           │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── gRPC TCP Delivery Rate Status ────────────────────────╮
│ WARNING: gRPC TCP Delivery Rate metrics unavailable. Did you start a         │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
```
