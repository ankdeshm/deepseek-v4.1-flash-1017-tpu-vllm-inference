# DeepSeek-V4.1-Flash-1017: Disaggregated Serving on TPU v7x

Prefill/decode (P/D) disaggregated serving for [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on TPU v7x with `vllm-tpu`. The repo includes deployment manifests, parallel checkpoint staging to GCS, and latency/throughput benchmarks for two layouts:

- **1P1D:** 1 prefill + 1 decode instance (8 chips)
- **2P2D:** 2 prefill + 2 decode instances (16 chips, full `2x2x4` slice)

**Full results:** [`results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`summary.json`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/summary.json)

> ⚠️ **Status:** In this run the P/D correctness hard gates (`pd_gates.py`: 2+2 ×16 and 64-token greedy equality) **did not pass**. The 50-states probe matched (50/50). Treat the throughput numbers below as performance data pending a correctness fix.

---

## 1. Model and serving setup

| Item | Details |
|:---|:---|
| **Checkpoint** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): 48 safetensors shards, 510.31 GB, staged to GCS with [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) and [`k8s/stage-deepseek-v41-job.yaml`](./k8s/stage-deepseek-v41-job.yaml) |
| **Architecture** | `DeepseekV4ForCausalLM`: 40 layers, `hidden_size=5120`, 384 routed MXFP4 experts + 1 shared expert, top-6 routing |
| **Attention** | `DEEPSEEK_V4_FLASH` sparse attention (Pallas), `index_topk=512` |
| **Speculative decoding** | DSpark on the decode instance (5 draft positions) |
| **Prefill** | Context encoder enabled (`CED=1`) |
| **KV transfer** | `TPURaidenConnector` (Stage 3) between prefill and decode hosts |
| **Hardware** | TPU v7x `2x2x4` = 16 chips / 32 TensorCores / 3,072 GiB HBM3e |
| **Instance size** | Each P or D instance runs on one `tpu7x-standard-4t` host (`2x2x1` = 4 chips / 8 TensorCores). Exact parallelism flags are in [`scripts/run_dsv41_pd_1017.py`](./scripts/run_dsv41_pd_1017.py). |
| **Router** | Embedded FastAPI P/D gateway ([`run_dsv41_pd_1017.py#L33-L165`](./scripts/run_dsv41_pd_1017.py#L33-L165)); orchestration in [`main()`](./scripts/run_dsv41_pd_1017.py#L401-L560) |

---

## 2. Results

All throughput values are **output** tokens per second. "Per chip" divides by all chips (prefill + decode). Per-user speed and latency values are **medians**.

### 2.1 Agentic workload (multi-turn, ~18K-token prompts, natural stopping)

Input-length mix 50:20:12:9:9 across buckets 1–8K / 8–16K / 16–32K / 32–64K / 64K+, with a synthetic shared-prefix cache. Highest-throughput row for each layout and cache target:

| Layout | Chips | Cache hit | Sessions | Mean ISL : OSL | Output tok/s | Per chip | Per decode chip | Per-user tok/s | TTFT (s) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1P1D | 8 | 94.4% | 128 | 17,934 : 484 | **7,446.2** | 931 | 1,862 | 69.10 | 1.27 | 1,462 / 1,462 |
| 1P1D | 8 | 75.2% | 128 | 18,796 : 670 | 6,848.4 | 856 | 1,712 | 69.19 | 1.85 | 1,339 / 1,339 |
| 2P2D | 16 | 94.7% | 256 | 17,905 : 488 | **14,541.5** | 909 | 1,818 | 67.55 | 1.29 | 2,874 / 2,874 |
| 2P2D | 16 | 76.8% | 256 | 18,812 : 691 | 14,389.6 | 899 | 1,799 | 71.15 | 1.85 | 2,675 / 2,675 |

At lower concurrency, per-user speed is higher. For example, 1P1D at n=8 runs at 111.8–119.5 tok/s per user, with TTFT of 0.32–0.91 s. See [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) for full sweeps.

### 2.2 Fixed-shape probes (InferenceX-style, random prompts)

| Layout | Chips | Shape (ISL : OSL) | Concurrency | Output tok/s | Per chip | Per-user tok/s | TPOT (ms) | TTFT (ms) | OK / Total |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1P1D | 8 | 1024 : 512 | 1 | 136.55 | 17.1 | 144.93 | 6.94 | 250.85 | 4 / 4 |
| 1P1D | 8 | 1024 : 512 | 8 | 898.42 | 112.3 | 136.97 | 7.30 | 471.72 | 16 / 16 |
| 1P1D | 8 | 4096 : 1024 | 8 | 874.48 | 109.3 | **166.16** | **6.02** | 843.16 | 16 / 16 |
| 1P1D | 8 | 4096 : 1024 | 64 | **5,224.30** | 653.0 | 139.37 | 7.18 | 2,102.42 | 64 / 64 |
| 2P2D | 16 | 1024 : 512 | 16 | 1,721.48 | 107.6 | 139.86 | 7.15 | 420.25 | 32 / 32 |
| 2P2D | 16 | 4096 : 1024 | 16 | 2,076.33 | 129.8 | 164.92 | 6.06 | 1,213.43 | 32 / 32 |
| 2P2D | 16 | 4096 : 1024 | 128 | **8,251.62** | 515.7 | 136.93 | 7.30 | 2,324.55 | 128 / 128 |

### 2.3 Scaling from 1P1D to 2P2D

| Workload | 1P1D (8 chips) | 2P2D (16 chips) | Scaling |
|:---|:---|:---|:---|
| Agentic, ~95% cache hit (n=128 → 256) | 7,446.2 tok/s | 14,541.5 tok/s | **1.95×** |
| Agentic, ~75–77% cache hit (n=128 → 256) | 6,848.4 tok/s | 14,389.6 tok/s | **2.10×** |
| Fixed-shape 1024:512 (c=8 → 16) | 898.42 tok/s | 1,721.48 tok/s | **1.92×** |
| Fixed-shape 4096:1024 (c=64 → 128) | 5,224.30 tok/s | 8,251.62 tok/s | **1.58×** |

2P2D runs two independent 1P1D pairs behind one router. It adds capacity at roughly constant per-chip efficiency (agentic, ~95% hit: 931 → 909 output tok/s per chip).

### 2.4 Known limitations

- **Correctness gate:** P/D vs. local greedy-equality hard gates failed in this run (see status note above).
- **Synthetic cache:** prompts reuse pre-seeded shared prefixes. Above ~128 distinct prefixes per prefill node, the realized cache hit rate drops.
- **Throughput vs. per-user speed:** the highest-throughput rows run at ~68–71 tok/s per user. Higher per-user speed (110–166 tok/s) comes at lower concurrency.
- **Small samples:** low-concurrency fixed-shape probes completed only 4–16 requests each.
- **Speculative acceptance:** not recorded in the sweep tables. Probes show mean accept lengths of 2.08–2.69.

---

## 3. Extended rerun: `A0–A4` & asymmetric `xPyD` scaling (`1P2D` / `1P3D` / `1P4D`, 8–20 chips)

Follow-up rerun using Raiden Stage 3 P/D disaggregation (`DP=8, TP=1, EP=8` per 4-chip `v7x` host, DSpark `K=5` speculative decoding, 150–180 s steady-state windows plus full tail drain, `ignore_eos=False`, mean ISL ≈ 17.9K–18.8K tokens; full flags and overlay commits in [`DEEPSEEK_V41_REPRODUCTION_MANIFEST.md`](./DEEPSEEK_V41_REPRODUCTION_MANIFEST.md)). Per-chip output divides by **all physical chips (`P + D`)**; `P90_low` is the 10th-percentile slowest-stream decode speed.

### 3.1 `1P1D` cache-realism (`A2`), Poisson arrivals (`A3`), and asymmetric `1P2D / 1P3D / 1P4D` (`C1`)

| Run & layout (total chips) | Prefix cache (trunks) | Operating point | Steady Out (`tok/s`) | Out / chip (`tok/s`) | Drain Out (`tok/s`) | Med user `tok/s` | `P90_low` `tok/s` | TTFT `p50 / p95` (`s`) | JSON |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|:---|
| `A1-b` `1P1D` (8 ch, caps `48, 288`) | 95.47% (40 shared) | Gate (`n=128`)<br>Max (`n=384`) | **7,408.6**<br>**12,820.0** | 926.1<br>**1,602.5** | 6,440.0<br>10,642.2 | 66.62<br>37.71 | **53.91**<br>29.00 | 1.20 / 1.61<br>1.62 / 5.10 | [`A1b`](./results/v7x-dsv41-rerun-a0-a4-c1/A1b_dsv41_1p1d_cap48_288.json) |
| `A2` `1P1D` (8 ch, `N` distinct trunks) | 95.12% (128 distinct)<br>77.33% (128 distinct)<br>8.10% (256 distinct, LRU cliff) | Gate & Max (`n=128`)<br>Gate & Max (`n=128`)<br>Eviction cliff (`n=256`) | **7,600.5**<br>**7,926.2**<br>3,486.2 | 950.1<br>990.8<br>435.8 | 6,783.3<br>6,656.2<br>3,116.0 | 67.17<br>72.16<br>84.73 | **54.76**<br>**61.62**<br>68.01 | 1.22 / 1.55 *(p90)*<br>1.79 / 2.99 *(p90)*<br>28.78 / 36.72 *(p90)* | [`A2_95`](./results/v7x-dsv41-rerun-a0-a4-c1/A2_dsv41_1p1d_ntrun_95pct.json) · [`A2_80`](./results/v7x-dsv41-rerun-a0-a4-c1/A2_dsv41_1p1d_ntrun_80pct.json) |
| `A3` `1P1D` (8 ch, Poisson open-loop) | 95.46% (122 distinct) | Gate (`11.2 RPS`)<br>Max (`23.5 RPS`) | **5,557.5**<br>**10,678.1** | 694.7<br>1,334.8 | 5,017.4<br>8,586.2 | 68.65<br>38.65 | **57.59**<br>32.74 | 1.06 / 1.48 *(p99 1.57)*<br>1.36 / 1.68 *(p99 1.79)* | [`A3_poisson`](./results/v7x-dsv41-rerun-a0-a4-c1/A3_dsv41_1p1d_poisson.json) |
| `C1` `1P2D` (12 ch = `4P + 8D`) | 95.47% (40)<br>78.4% (128) | Gate (`n=192`)<br>Max (`n=512`) | **10,600.0**<br>**15,831.4** | 883.3<br>1,319.3 | 8,288.4<br>12,648.1 | 64.92<br>52.89 | **53.30**<br>43.31 | 1.30 / 2.31<br>6.95 / 24.69 | [`C1_1p2d_95`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p2d_95pct_40trunks.json) · [`C1_1p2d_80`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p2d_80pct_128trunks.json) |
| `C1` `1P3D` (16 ch = `4P + 12D`) | 95.47% (40)<br>78.5% (128) | Gate (`n=256`)<br>Max (`n=384` / `n=512`) | **15,540.5**<br>**20,280.9** / 17,185.0 | **971.3**<br>1,267.6 / 1,074.1 | 12,719.7<br>16,198.4 / 14,372.4 | 73.15<br>65.90 / 75.34 | **56.06**<br>47.90 / **62.19** | 1.36 / 2.38<br>1.51 / 3.97 | [`C1_1p3d_95`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p3d_95pct_40trunks.json) · [`C1_1p3d_80`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p3d_80pct_128trunks.json) |
| `C1` `1P4D` (20 ch = `4P + 16D`) | 95.4% (40)<br>78.7% (128) | Gate (`n=256`)<br>Max (`n=640` / `n=384`) | **17,445.1**<br>**18,855.5** / 17,690.3 | 872.3<br>942.8 / 884.5 | 13,810.6<br>15,575.8 / 14,340.7 | 84.43<br>65.54 / 90.72 | **67.94**<br>45.40 / **77.35** | 1.38 / 2.33<br>8.46 / 14.26 | [`C1_1p4d_95`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p4d_95pct_40trunks.json) · [`C1_1p4d_80`](./results/v7x-dsv41-rerun-a0-a4-c1/C1_1p4d_80pct_128trunks.json) |

### 3.2 3-way TTFT decomposition (`A3`) & standalone `1P` uncached prefill (`A4`, 4 chips)

- **UUID-joined TTFT decomposition (14,867 requests, [`A3_ttft_3way_decomposition.json`](./results/v7x-dsv41-rerun-a0-a4-c1/A3_ttft_3way_decomposition.json)):** `p50` TTFT of `1,232.66 ms` splits into **911.60 ms (74.0%)** prefill queue + compute, **58.95 ms (4.8%)** Raiden Stage 3 DCN KV transfer (`p90 = 87.77 ms`), and **257.36 ms (20.9%)** decode slot admission + first-step 128-token SWA tail replay.
- **Standalone `1P` uncached prefill (4 chips, `0.0%` cache hit, [`A4_dsv41_1p_uncached_prefill.json`](./results/v7x-dsv41-rerun-a0-a4-c1/A4_dsv41_1p_uncached_prefill.json)):** Reaches **27,635.4 tok/s** (`6,908.9/chip`) at 4K ISL, **53,792.3 tok/s** (`13,448.1/chip`) at 16K, **82,360.2 tok/s** (`20,590.0/chip`) at 32K, **102,123.0 tok/s** (`25,530.8/chip`) at 64K, and **111,737.6 tok/s** (`27,934.4/chip`, `894.6 TFLOP/s/chip`) at 120K ISL.

---

## 4. Repository structure

| Path | Purpose |
|:---|:---|
| [`docker/Dockerfile`](./docker/Dockerfile) | `vllm` + TorchTPU + [`SemiAnalysisAI/InferenceX@f6c1f5b5`](https://github.com/SemiAnalysisAI/InferenceX) |
| [`DEEPSEEK_V41_REPRODUCTION_MANIFEST.md`](./DEEPSEEK_V41_REPRODUCTION_MANIFEST.md) | Exact server flags, overlay commits, and reproduction steps for `A0–A4` and `C1` |
| [`scripts/run_dsv41_pd_1017.py`](./scripts/run_dsv41_pd_1017.py) | 1P1D / 2P2D launcher, P/D router, and benchmark suite |
| [`scripts/launch_dsv41_role_rerun.sh`](./scripts/launch_dsv41_role_rerun.sh) · [`scripts/run_c1_phase2_ratios.sh`](./scripts/run_c1_phase2_ratios.sh) | Disaggregated `xPyD` role launcher and `1P2D` / `1P3D` / `1P4D` ratio driver |
| [`scripts/bench_empirical_isl_dsv41_rerun.py`](./scripts/bench_empirical_isl_dsv41_rerun.py) · [`scripts/dsv41_glm53_roofline.py`](./scripts/dsv41_glm53_roofline.py) | Multi-turn empirical ISL closed-loop / Poisson harness and analytical roofline model |
| [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) | OpenAI-compatible streaming benchmark driver: 0% / 80% / 95% prefix-cache sweeps plus InferenceX shapes |
| [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) | Parallel 48-shard Hugging Face → GCS checkpoint stager |
| [`k8s/v7x-16-flex-dsv41-pd-1017.yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | 4-host Indexed Job running 1P1D and 2P2D on TPU v7x-16 |
| [`k8s/v7x-16-deepseek-v41-flash-1017.yaml`](./k8s/v7x-16-deepseek-v41-flash-1017.yaml) | Single-deployment manifest |
| [`k8s/stage-deepseek-v41-job.yaml`](./k8s/stage-deepseek-v41-job.yaml) | Checkpoint staging job |
| [`k8s/storage-gcsfuse.yaml`](./k8s/storage-gcsfuse.yaml) | GCS FUSE storage |
| [`results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) | Initial 1P1D and 2P2D JSON, logs, and `SUMMARY.md` |
| [`results/v7x-dsv41-rerun-a0-a4-c1/`](./results/v7x-dsv41-rerun-a0-a4-c1/) | Rerun JSON artifacts (`A0–A4`, `C1` `1P2D` / `1P3D` / `1P4D`, 3-way TTFT, analytical rooflines) |

---

## 5. Quickstart

```bash
# 1. Stage deepseek-ai/DeepSeek-V4.1-Flash-1017 (510.31 GB, 48 shards) to GCS
kubectl apply -f k8s/stage-deepseek-v41-job.yaml

# 2. Deploy the 1P1D and 2P2D runner (ConfigMap + Job) on TPU v7x-16
kubectl create configmap dsv41-pd-script \
  --from-file=run_dsv41_pd_1017.py=scripts/run_dsv41_pd_1017.py
kubectl apply -f k8s/v7x-16-flex-dsv41-pd-1017.yaml
```

