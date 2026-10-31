# `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`510 GB` FP8/FP4 MoE) Disaggregated `1P1D` & `2P2D` Serving on Cloud TPU `v7x`

Production Disaggregated Prefill-Decode (`1P1D` and `2P2D` via gRPC `TPUConnector` KV-cache transfer) deployment manifests, parallel GCS checkpoint staging, and empirical latency/throughput benchmarks for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (`48` safetensors shards, `510.31 GB` total) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm-tpu` (`DeepseekV4ForCausalLM` + `DEEPSEEK_V4_FLASH` Pallas attention + `DSpark` block-5 speculative decoding)**.

---

## 1. Model Architecture — [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`48` safetensors shards, `510.31 GB`) |
| **Architecture** | `DeepseekV4ForCausalLM` (`40` Transformer layers, `hidden_size=5120`, `moe_intermediate_size=2304`, `vocab_size=129,280`) |
| **MoE Routing** | `384` routed experts + `1` shared expert per layer, **top-6** active routing (`noaux_tc`, `sqrtsoftplus`) |
| **Quantization** | `32×32` dynamic block-FP8 (`scale_fmt="ue8m0"`) + **`FP4` routed expert weights (`expert_dtype="fp4"`)** |
| **Attention & Context** | `DEEPSEEK_V4_FLASH` Sparse MLA (`index_topk=512`, `candidate_topk_blocks=2048`) + Hierarchical Context Compression + `hc_mult=4` residual stream |
| **Engram & Speculative** | HBM-resident N-gram Engram memory layers (`engram_layer_ids=[1, 14]`) + **`DSpark` (`dspark_block_size=5`, `num_speculative_tokens=4`)** |

---

## 2. Measured Disaggregated `1P1D` (8 Chips) & `2P2D` (16 Chips) Results on TPU `v7x` (`0` Errors Across All 24 Runs)

Each Prefill (`P`) or Decode (`D`) instance runs on **1 `tpu7x-standard-4t` host (`2x2x1` = 4 `v7x` chips / 8 TensorCores, `TP=8`)** connected through an asynchronous proxy + `TPUConnector` KV-cache transfer:

| Topology | Chips (`TensorCores`) | Workload (`Input : Output`) | Concurrency | Output TPS | Total TPS (`In + Out`) | Mean Stream TPS | Mean TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`1P1D`** | `8` (`16` TC) | `1,024 : 1,024` *(Interactive)* | `1` | `157.82` | `315.63` | **`166.16`** | `308.6` | **`6.02`** | `6.14` |
| **`1P1D`** | `8` (`16` TC) | `4,096 : 1,024` *(Standard)* | `8` | `1,004.58` | `5,022.90` | **`139.04`** | `723.5` | **`7.19`** | `7.63` |
| **`1P1D`** | `8` (`16` TC) | `4,096 : 1,024` *(High-Throughput)* | `64` | `4,884.19` | `24,420.94` | `96.18` | `3,581.2` | `10.42` | `11.83` |
| **`1P1D`** | `8` (`16` TC) | `1,024 : 1,024` *(Max Output)* | `128` | **`7,446.23`** | `14,892.46` | `80.19` | `2,941.6` | `12.51` | `14.20` |
| **`2P2D`** | `16` (`32` TC) | `4,096 : 1,024` *(Linear Scale)* | `16` | `2,008.74` | `10,043.70` | **`139.42`** | `741.8` | **`7.17`** | `7.59` |
| **`2P2D`** | `16` (`32` TC) | `4,096 : 1,024` *(High-Throughput)* | `128` | `9,664.81` | `48,324.05` | `95.44` | `3,642.5` | `10.50` | `11.96` |
| **`2P2D`** | `16` (`32` TC) | `1,024 : 1,024` *(Peak Decode)* | `256` | **`14,541.52`** | `29,083.04` | `79.31` | `3,018.4` | `12.64` | `14.38` |
| **`2P2D`** | `16` (`32` TC) | `32,768 : 256` *(Peak Prefill)* | `64` | `4,400.40` | **`567,652.10`** *(`563,251.7` in tok/s)* | `98.12` | `9,842.0` | `10.21` | `11.74` |

- **Near-Linear `1.95×–2.00×` Scaling (`1P1D -> 2P2D`):** Scaling from `1P1D` (8 chips) to `2P2D` (16 chips) doubles output throughput from **`7,446.2 -> 14,541.5` output tok/s (`1.95×`)** while preserving sub-`7.2 ms` interactive TPOT (`166.16` stream tok/s at `c=1`) and reaching **`563,251.7` prefill input tok/s** on `32K` long-context workloads!

---

## 3. Repository Structure & Quickstart

```text
├── README.md
├── docker/
│   └── Dockerfile                              # vllm + TorchTPU + SemiAnalysisAI/InferenceX@f6c1f5b5
├── scripts/
│   ├── run_dsv41_pd_1017.py                    # Disaggregated 1P1D & 2P2D launcher + TPUConnector proxy + benchmark suite
│   ├── run_tpu_benchmark_sweep.py              # OpenAI SSE streaming + 0%/80%/95% prefix-cache + InferenceX driver
│   └── stage_hf_to_gcs.py                      # Parallel 48-shard Hugging Face -> GCS checkpoint stager
├── k8s/
│   ├── storage-gcsfuse.yaml                    # GKE GCS Fuse PV/PVC manifest
│   ├── stage-deepseek-v41-job.yaml             # Kubernetes Job to stream 510.31 GB checkpoint to GCS
│   └── v7x-16-dsv41-pd-1017.yaml               # 4-host Indexed Job running 1P1D & 2P2D on TPU v7x-16
└── results/                                    # Complete 1P1D & 2P2D JSON & SUMMARY.md benchmark outputs
```

```bash
# 1. Stage deepseek-ai/DeepSeek-V4.1-Flash-1017 (510.31 GB, 48 shards) to GCS
kubectl apply -f k8s/stage-deepseek-v41-job.yaml

# 2. Deploy Disaggregated 1P1D & 2P2D Runner ConfigMap & Job on TPU v7x-16
kubectl create configmap dsv41-pd-script --from-file=run_dsv41_pd_1017.py=scripts/run_dsv41_pd_1017.py
kubectl apply -f k8s/v7x-16-dsv41-pd-1017.yaml
```
