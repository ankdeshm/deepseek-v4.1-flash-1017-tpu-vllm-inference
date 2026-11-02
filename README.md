# `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`510 GB` FP8/FP4 MoE) Disaggregated `1P1D` & `2P2D` Serving on Cloud TPU `v7x`

Production Disaggregated Prefill-Decode (`1P1D` and `2P2D` via gRPC `TPUConnector` KV-cache transfer) deployment manifests, parallel GCS checkpoint staging, and empirical latency/throughput benchmarks for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (`48` safetensors shards, `510.31 GB` total) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm-tpu` (`DeepseekV4ForCausalLM` + `DEEPSEEK_V4_FLASH` Pallas attention + `DSpark` block-5 speculative decoding)**.

- **Full Disaggregated Benchmark Summary:** [`results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`summary.json`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/summary.json)

---

## 1. Model Architecture & Disaggregated P/D Code Anchors

| Property | Specification & Code Anchor |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`48` shards, `510.31 GB` staged via [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) & [`k8s/stage-deepseek-v41-job.yaml`](./k8s/stage-deepseek-v41-job.yaml)) |
| **Architecture & MoE Routing** | `DeepseekV4ForCausalLM` (`40` layers, `hidden_size=5120`, `384` routed `FP4` experts + `1` shared expert, **top-6** `noaux_tc` routing, `32×32` `ue8m0` dynamic block-FP8) |
| **Attention & Speculative Decoding** | `DEEPSEEK_V4_FLASH` Sparse MLA (`index_topk=512`, `candidate_topk_blocks=2048`) + Engram layers `[1, 14]` + **`DSpark` (`dspark_block_size=5`, `num_speculative_tokens=4`)** |
| **Disaggregated `1P1D` / `2P2D` Gateway & `TPUConnector`** | Embedded FastAPI P/D router ([`GATEWAY_2P2D_PY` `#L33`](./scripts/run_dsv41_pd_1017.py#L33-L165)) + per-host `TP=8` Prefill/Decode orchestration ([`main()` `#L401`](./scripts/run_dsv41_pd_1017.py#L401-L560)) |

---

## 2. Measured Disaggregated `1P1D` (8 Chips) & `2P2D` (16 Chips) Results → Exact Script, Manifest & Proof (`0` Errors Across All 24 Runs)

Each Prefill (`P`) or Decode (`D`) instance runs on **1 `tpu7x-standard-4t` host (`2x2x1` = 4 `v7x` chips / 8 TensorCores, `TP=8`)** launched via [`k8s/v7x-16-flex-dsv41-pd-1017.yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml):

| Topology | Chips (`TC`) | Workload (`In : Out`) | Conc | Output TPS | Total TPS (`In + Out`) | Mean Stream TPS | Mean TTFT (ms) | Mean TPOT (ms) | Script & Manifest | Empirical Result Proof |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| **`1P1D`** | `8` (`16`) | `1,024 : 1,024` *(Interactive)* | `1` | `157.82` | `315.63` | **`166.16`** | `308.6` | **`6.02`** | [`pd_1017.py#L341`](./scripts/run_dsv41_pd_1017.py#L341-L353) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`Gates JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/pd_gates_1p1d.json) · [`80% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_1p1d_80pct.json) |
| **`1P1D`** | `8` (`16`) | `4,096 : 1,024` *(Standard)* | `8` | `1,004.58` | `5,022.90` | **`139.04`** | `723.5` | **`7.19`** | [`pd_1017.py#L341`](./scripts/run_dsv41_pd_1017.py#L341-L353) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`80% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_1p1d_80pct.json) · [`95% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_1p1d_95pct.json) |
| **`1P1D`** | `8` (`16`) | `4,096 : 1,024` *(High-Tput)* | `64` | `4,884.19` | `24,420.94` | `96.18` | `3,581.2` | `10.42` | [`pd_1017.py#L341`](./scripts/run_dsv41_pd_1017.py#L341-L353) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_1p1d_80pct.json) |
| **`1P1D`** | `8` (`16`) | `1,024 : 1,024` *(Max Output)* | `128` | **`7,446.23`** | `14,892.46` | `80.19` | `2,941.6` | `12.51` | [`pd_1017.py#L341`](./scripts/run_dsv41_pd_1017.py#L341-L353) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`Log`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_1p1d_80pct.log) |
| **`2P2D`** | `16` (`32`) | `4,096 : 1,024` *(Linear Scale)* | `16` | `2,008.74` | `10,043.70` | **`139.42`** | `741.8` | **`7.17`** | [`pd_1017.py#L354`](./scripts/run_dsv41_pd_1017.py#L354-L370) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`80% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_2p2d_80pct.json) · [`95% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_2p2d_95pct.json) |
| **`2P2D`** | `16` (`32`) | `4,096 : 1,024` *(High-Tput)* | `128` | `9,664.81` | `48,324.05` | `95.44` | `3,642.5` | `10.50` | [`pd_1017.py#L354`](./scripts/run_dsv41_pd_1017.py#L354-L370) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_2p2d_80pct.json) |
| **`2P2D`** | `16` (`32`) | `1,024 : 1,024` *(Peak Decode)* | `256` | **`14,541.52`** | `29,083.04` | `79.31` | `3,018.4` | `12.64` | [`pd_1017.py#L354`](./scripts/run_dsv41_pd_1017.py#L354-L370) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`Log`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_2p2d_80pct.log) |
| **`2P2D`** | `16` (`32`) | `32,768 : 256` *(Peak Prefill)* | `64` | `4,400.40` | **`567,652.10`** *(`563.3K` in)* | `98.12` | `9,842.0` | `10.21` | [`pd_1017.py#L227`](./scripts/run_dsv41_pd_1017.py#L227-L320) · [`yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) | [`SUMMARY.md`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) · [`95% JSON`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/empirical_2p2d_95pct.json) |

- **Near-Linear `1.95×–2.00×` Scaling (`1P1D -> 2P2D`):** Scaling from `1P1D` (8 chips) to `2P2D` (16 chips) doubles output throughput from **`7,446.2 -> 14,541.5` output tok/s (`1.95×`)** while preserving **`6.02 ms` (`166.16` stream tok/s)** interactive `c=1` TPOT and reaching **`563,251.7` prefill input tok/s** on `32K` long-context workloads!

---

## 3. Repository Structure & Quickstart

- [`docker/Dockerfile`](./docker/Dockerfile) — `vllm` + `TorchTPU` + `SemiAnalysisAI/InferenceX@f6c1f5b5`
- [`scripts/run_dsv41_pd_1017.py`](./scripts/run_dsv41_pd_1017.py) — Disaggregated `1P1D` & `2P2D` launcher + `TPUConnector` proxy + benchmark suite
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) — OpenAI SSE streaming + `0%/80%/95%` prefix-cache + `InferenceX` driver
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel 48-shard Hugging Face -> GCS checkpoint stager
- [`k8s/v7x-16-flex-dsv41-pd-1017.yaml`](./k8s/v7x-16-flex-dsv41-pd-1017.yaml) — **4-host Indexed Job running `1P1D` & `2P2D` on TPU `v7x-16`**
- [`k8s/v7x-16-deepseek-v41-flash-1017.yaml`](./k8s/v7x-16-deepseek-v41-flash-1017.yaml), [`k8s/stage-deepseek-v41-job.yaml`](./k8s/stage-deepseek-v41-job.yaml), [`k8s/storage-gcsfuse.yaml`](./k8s/storage-gcsfuse.yaml)
- [`results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/`](./results/v7x-16-dsv41-1017-pd__20261031_1p1d_2p2d/SUMMARY.md) — Complete `1P1D` & `2P2D` JSON, log & `SUMMARY.md` outputs

```bash
# 1. Stage deepseek-ai/DeepSeek-V4.1-Flash-1017 (510.31 GB, 48 shards) to GCS
kubectl apply -f k8s/stage-deepseek-v41-job.yaml

# 2. Deploy Disaggregated 1P1D & 2P2D Runner ConfigMap & Job on TPU v7x-16
kubectl create configmap dsv41-pd-script --from-file=run_dsv41_pd_1017.py=scripts/run_dsv41_pd_1017.py
kubectl apply -f k8s/v7x-16-flex-dsv41-pd-1017.yaml
```
