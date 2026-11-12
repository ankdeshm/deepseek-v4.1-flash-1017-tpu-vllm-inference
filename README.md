# DeepSeek-V4.1-Flash-1017 Disaggregated Serving on Google Cloud TPU v7x (Ironwood)

Prefill/Decode (`P/D`) disaggregated serving benchmarks for [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on Google Cloud TPU `v7x` (Ironwood) using `vllm-tpu`, `TPURaidenConnector`, and [`llm-d`](https://llm-d.ai/).

**Summary:** Across 16 TPU `v7x` chips (`32` TensorCores) with `FP8` attention, `MXFP4` routed experts, and `DSpark` (`K=5`) speculative decoding, disaggregated `llm-d` serving reaches **51,135 output tok/s (3,196 tok/s per total chip; 4,261 tok/s per decode chip) in `1P3D` (`EP=8`) and 46,823 output tok/s (2,926 tok/s per total chip; 5,853 tok/s per decode chip) in `2P2D` (`EP=8`)** on `2048:1024` prompts (`62–77 tok/s/user`, `20/20` bit-exact greedy parity). Single-host `EP=8` decode outperforms 2-host `EP=16` decode by **14.5% at peak (`5,853` vs. `5,113` tok/s/decode chip) and 1.94× at matched `~61 tok/s/user`**, and reaches **43.7% of the published LMSYS `GB200 NVL72` per-decode-GPU figure (`5,853` vs. `13,386 tok/s/decode GPU` on DeepSeek-V3/R1 at `EP=48`)**.

---

## 1. Model, Framework, and Hardware

- **Model:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`DeepseekV4ForCausalLM`, `510.3 GB` across 48 `safetensors` shards; `131,072` max context length).
  - **Architecture & Weights:** 40 transformer layers (`hidden_size = 5,120`), **384 routed `MXFP4` (`W4A8`) experts + 1 shared expert** (`top-6` routing), **`FP8` (`float8_e4m3fn`) sparse `CSA` MLA** (`index_topk = 512`) + `BF16` sliding-window latent attention, **`FP8` KV cache**, context encoder (`CED=1`) on prefill, and **`DSpark` speculative decoding** (`K = 5` draft tokens) on decode.
- **Framework:** `vllm/vllm-tpu` + `vllm-torchtpu` (`libtpu` `0.0.48`, `PYTHONHASHSEED=0` across roles for bit-exact `P/D` greedy parity), `TPURaidenConnector` Stage-3 host-DRAM/TCP KV resharding, and [`llm-d`](https://llm-d.ai/) `v1.0.0` (Gateway API Inference Extension `v1.6.2` `InferencePool`, `EndpointPicker`, Envoy router, and `nixlv2` disagg sidecar).
- **Hardware:** Google Cloud TPU `v7x` (Ironwood) — **`192 GiB` HBM3e and `~7.4 TB/s` HBM bandwidth per physical chip** (`2` TensorCores / `96 GiB` per TensorCore; `4` physical chips = `8` TensorCores per `tpu7x-standard-4t` host). Evaluated from `4` chips (`1P` standalone prefill) and `8` chips (`1P1D`) up to `16` chips (`2P2D`, `1P3D`) and `20` chips (`1P4D`).

---

## 2. Main Results

All per-total-chip columns divide output throughput by **all physical `v7x` chips (`Prefill + Decode`)**; per-decode-chip columns divide by physical decode chips only. All `llm-d` runs below passed the 20-prompt 64-token bit-exact greedy parity check (`0` request failures across every table).

### A. Fixed-Shape `llm-d` Disaggregated Serving (`16` TPU `v7x` Chips) & Comparisons

| Topology & Decode Parallelism | Prompt Shape (`ISL : OSL`) | Concurrency | Steady Output `tok/s` | `tok/s` / Total Chip (`/16`) | `tok/s` / Decode Chip | Median User `tok/s` (`TTFT p50`) | Comparison |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **`1P3D` (`4P + 12D` chips, `EP=8`)** | `2048 : 1024` (uncached) | 1,024 | **51,134.9** | **3,195.9** | 4,261.2 (`/12`) | 76.5 (`6.98 s`) | **Highest per-total-chip (`+9.2%` over `2P2D` `EP=8`)** |
| **`1P3D` (`4P + 12D` chips, `EP=8`)** | `2048 : 1024` (uncached) | 768 | **49,512.4** | **3,094.5** | 4,126.0 (`/12`) | **79.0** (`1.99 s`) | **+15.9% per total chip** and faster user speed than `2P2D` at `c=768` |
| **`2P2D` (`8P + 8D` chips, `EP=8`)** | `2048 : 1024` (uncached) | 1,024 | **46,823.0** | **2,926.4** | **5,852.9** (`/8`) | 62.2 (`5.54 s`) | **Highest per-decode-chip (`+14.5%` over `EP=16` decode)** |
| **`2P2D` (`8P + 8D` chips, `EP=8`)** | `2048 : 1024` (uncached) | 768 | **42,719.1** | 2,669.9 | **5,339.9** (`/8`) | 61.3 (`1.28 s`) | **1.94× higher tok/s/decode chip** than `EP=16` at `~61 tok/s/user` |
| `2P + 1×8c D` (`8P + 8D` chips, `EP=16`) | `2048 : 1024` (uncached) | 1,024 | 40,907.1 | 2,556.7 | 5,113.4 (`/8`) | 54.4 (`6.20 s`) | 2-host `EP=16` ICI hop + shared 16-rank step barrier slows decode vs. `EP=8` |
| `2P2D` (`8P + 8D` chips, `EP=8`) | `1024 : 1024` (uncached) | 1,024 | **44,166.0** | **2,760.4** | 5,520.8 (`/8`) | 59.1 (`5.83 s`) | Slot-cap bound (`768` total decode slots across `2D`) |
| `2P2D` (`8P + 8D` chips, `EP=8`) | `8192 : 1024` (uncached) | 512 | **26,229.5** | **1,639.3** | 3,278.7 (`/8`) | 75.5 (`5.88 s`) | Prefill-bound (`~215k` prompt `tok/s` across `8` prefill chips) |
| `2P2D` (`8P + 8D` chips, `EP=8`) | `2048 : 1024` (`99.3%` cached) | 1,024 | **44,100.7** | **2,756.3** | 5,512.6 (`/8`) | 59.5 (`5.59 s`) | Confirms `S1` is decode-slot-bound (`<15%` KV HBM used) |
| Published GPU reference: LMSYS `GB200 NVL72` (`EP=48`) | `~2000 : 1024` (`FP8` attn + `NVFP4` MoE) | — | — | — | 13,386 (`/GPU`) | Not paired | `v7x` `EP=8` (`5,852.9`) is **43.7% per decode accelerator** (4-chip `EP=8` DSV4.1 vs. 48-GPU NVLink `EP=48` DeepSeek-V3/R1) |

### B. Multi-Turn Agentic Serving (`~18K` Mean Input Tokens, Natural Stop) & Prefill Scaling

Measured across `8–20` `v7x` chips (`DP=8, EP=8` per 4-chip host, `48/288` decode scheduler caps, `DSpark K=5`, `ignore_eos=False`; [`results/agentic_xpyd_scaling/`](./results/agentic_xpyd_scaling/), [`results/llmd_disaggregated/`](./results/llmd_disaggregated/)):

| Layout (Total Physical Chips) | Prefix Cache Hit | Sessions / Arrival Rate | Steady Output `tok/s` | `tok/s` / Total Chip | Median / `P90_low` User `tok/s` | `TTFT p50 / p95` (s) | Notes & Comparison |
| :--- | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| **`1P1D` (`8` chips = `4P + 4D`)** | 95.1% (`128` trunks) | `n = 128` closed-loop | **7,600.5** | **950.1** | 67.2 / 54.8 | 1.22 / 1.55 *(p90)* | Interactive baseline (`12,820 tok/s` = `1,602.5/chip` at `n=384`, `37.7 tok/s/user`) |
| **`1P1D` (`8` chips = `4P + 4D`)** | 77.3% (`128` trunks) | `n = 128` closed-loop | **7,926.2** | **990.8** | 72.2 / 61.6 | 1.79 / 2.99 *(p90)* | Realistic multi-session cache hit |
| **`1P1D` (`8` chips = `4P + 4D`)** | 95.5% (`122` trunks) | `11.2 / 23.5 RPS` Poisson | **5,557.5 / 10,678.1** | **694.7 / 1,334.8** | 68.7 / 38.7 | 1.06 / 1.48 · 1.36 / 1.68 | Open-loop arrivals (`p50` TTFT = `74%` prefill + `4.8%` Raiden KV + `20.9%` decode admit) |
| **`1P2D` (`12` chips = `4P + 8D`)** | 95.5% / 78.4% | `n = 192` / `n = 512` | **10,600.0 / 15,831.4** | **883.3 / 1,319.3** | 64.9 / 52.9 | 1.30 / 2.31 · 6.95 / 24.7 | **1.39–2.00× total tok/s** over `1P1D` (`1.5×` chips) |
| **`1P3D` (`16` chips = `4P + 12D`)** | 95.5% (`40` trunks) | `n = 256` / `n = 384` | **15,540.5 / 20,280.9** | **971.3 / 1,267.6** | 73.2 / 65.9 | 1.36 / 2.38 · 1.51 / 3.97 | **2.04–2.67× total tok/s** over `1P1D` (`2.0×` chips; best `>65 tok/s/user` efficiency) |
| **`1P4D` (`20` chips = `4P + 16D`)** | 95.4% (`40` trunks) | `n = 256` / `n = 640` | **17,445.1 / 18,855.5** | 872.3 / 942.8 | **84.4** / 65.5 | 1.38 / 2.33 · 8.46 / 14.3 | Single `4`-chip prefill host saturates above `3D` (`1P3D` is higher tok/s/chip) |
| **`2P2D` `llm-d` (`16` chips = `8P + 8D`)** | 95.3% (DP-pinned) | `n = 128` / `n = 256` | **8,529.9 / 10,753.5** | 533.1 / 672.1 | 70.6 / 43.8 | 0.97 / 1.45 · 1.15 / 1.60 | Prefix-aware `llm-d` routing + `X-data-parallel-rank` lifts cache hit `+13–20 pts` |
| **Standalone `1P` Prefill (`4` chips)** | 0.0% (uncached) | `4K → 32K → 120K` ISL | **27,635 → 82,360 → 111,738** *(prompt tok/s)* | **6,909 → 20,590 → 27,934** *(prompt tok/s/chip)* | — | — | Reaches `894.6 TFLOP/s/chip` at `120K` uncached ISL |

---

## 3. Reproducing the Main Results

### 1. Stage Checkpoint (`510.3 GB`, 48 Shards) to GCS
Script: [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) and [`k8s/stage-deepseek-v41-job.yaml`](./k8s/stage-deepseek-v41-job.yaml):

```bash
kubectl apply -f k8s/stage-deepseek-v41-job.yaml
```

### 2. Deploy `llm-d` `2P2D` or `1P3D` Disaggregated Stack & Run Shape / Routing Benchmarks
Manifests & scripts: [`k8s/llmd/`](./k8s/llmd/) and [`scripts/llmd/`](./scripts/llmd/):

```bash
# Deploy Gateway API Inference Extension CRDs, llm-d router/EndpointPicker, and EP=8 prefill/decode pods
kubectl apply -f k8s/llmd/gaie-v1.6.2-manifests.yaml
kubectl apply -f k8s/llmd/llmd-router-standalone-rendered.yaml
kubectl apply -f k8s/llmd/llmd-prefill-poola.yaml
kubectl apply -f k8s/llmd/llmd-decode-poolb-ep8.yaml

# Verify 20/20 bit-exact 64-token greedy parity against a standalone server
python3 scripts/llmd/check_greedy_correctness.py --gateway-url http://<LLMD_GATEWAY_IP>:8000

# Run S1 (2048:1024), S2 (1024:1024), or S3 (8192:1024) concurrency ladders
python3 scripts/llmd/benchmark_llmd_shapes.py --gateway-url http://<LLMD_GATEWAY_IP>:8000 --isl 2048 --osl 1024
```

### 3. Run Multi-Turn Agentic `xPyD` (`1P1D` / `1P2D` / `1P3D` / `1P4D`) Sweeps
Scripts: [`scripts/launch_pd_role.sh`](./scripts/launch_pd_role.sh), [`scripts/run_xpyd_ratios.sh`](./scripts/run_xpyd_ratios.sh), and [`scripts/benchmark_agentic_isl.py`](./scripts/benchmark_agentic_isl.py):

```bash
# Launch prefill or decode role on each 4-chip TPU v7x host (DP=8, EP=8, caps 48/288, DSpark K=5)
ROLE=prefill bash scripts/launch_pd_role.sh
ROLE=decode  bash scripts/launch_pd_role.sh

# Run multi-turn ~18K ISL closed-loop or Poisson benchmark
python3 scripts/benchmark_agentic_isl.py --proxy-url http://<PROXY_HOST>:8400 --target-hit 0.95 --sessions 256
```

---

## 4. Detailed Ablations, Server Flags, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete `vllm serve` and `LIBTPU_INIT_ARGS` reproduction flags, `DSpark` scheduler-cap root-cause analysis (`16/96` collapse vs. `48/288`), `EP=8` vs. `EP=16` per-concurrency tables, `llm-d` prefix-aware routing + DP-rank affinity 4-way ablation, `PYTHONHASHSEED=0` greedy parity fix, and full `GB200 NVL72` comparison caveats.
- **[`results/`](./results/):** Raw JSON and time-series outputs under [`llmd_disaggregated/`](./results/llmd_disaggregated/), [`agentic_xpyd_scaling/`](./results/agentic_xpyd_scaling/), and [`initial_1p1d_2p2d/`](./results/initial_1p1d_2p2d/).
