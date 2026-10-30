# `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`510 GB` FP8/FP4 MoE) Serving & Benchmarking on Cloud TPU `v7x`

Multi-host deployment manifests, parallel GCS checkpoint staging, and streaming + prefix-cache (`0%`, `80%`, `95%`) + `InferenceX` benchmarking suite for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (`48` safetensors shards, `510.31 GB` total) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm 0.32.0` + `vllm-torchtpu` + `TorchTPU 0.2.0` GA**.

---

## 1. Model Architecture — [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`48` safetensors shards, `510.31 GB`) |
| **Architecture** | `DeepseekV41ForCausalLM` (`40` Transformer layers, `hidden_size=5120`, `moe_intermediate_size=2304`, `vocab_size=129,280`) |
| **MoE Routing** | `384` routed experts + `1` shared expert per layer, **top-6** active routing (`noaux_tc`, `sqrtsoftplus`) |
| **Quantization** | `32×32` dynamic block-FP8 (`scale_fmt="ue8m0"`) + **`FP4` routed expert weights (`expert_dtype="fp4"`)** |
| **Attention & Context** | Sparse MLA (`index_topk=512`, `candidate_topk_blocks=2048`) + Hierarchical Context Compression (`compress_ratios`) + `hc_mult=4` residual stream, up to `1,048,576` (`1M`) tokens |
| **Engram & Speculative** | N-gram Engram memory layers (`engram_layer_ids=[1, 14]`) + `num_nextn_predict_layers=3` / `DSpark` (`dspark_block_size=5`) |

---

## 2. Benchmark Matrix & Targets on TPU `v7x-16` (`2x2x4`, `32` TensorCores)

- **Parallelism Sweeps:** Intra-chip `TP=2, DP=16, EP=32` and Zero-TP `TP=1, DP=32, EP=32` across `32` TensorCores (`3,072 GiB` HBM3e), keeping HBM-resident Engram embeddings (`--engram-config='{"cpu_offload":false,"embedding_across_dp":true}'`) and EP-filtered Run:AI GCS weight streaming.
- **Prefix-Cache Hit Sweeps:** `0%` uncached, `80%` prefix-cache hit, and `95%` prefix-cache hit across concurrencies `c = 1, 4, 8, 16, 32, 64`.
- **Context Length Sweeps:** `2,048`, `8,192`, `16,384`, and `30,000` input tokens (`1,024` output tokens), plus first-hand [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) (`4096:1024`, `c=8`) cross-checks.
- *Live TPU `v7x-16` benchmark artifacts are automatically synced into [`results/`](./results/) as runs complete.*

---

## 3. Repository Structure & Quickstart

```text
├── README.md
├── docker/
│   └── Dockerfile                              # vllm 0.32.0 + TorchTPU 0.2.0 GA + SemiAnalysisAI/InferenceX@f6c1f5b5
├── scripts/
│   ├── run_tpu_benchmark_sweep.py              # OpenAI SSE streaming + 0%/80%/95% prefix-cache + InferenceX driver
│   └── stage_hf_to_gcs.py                      # Parallel 48-shard Hugging Face -> GCS checkpoint stager
├── k8s/
│   ├── storage-gcsfuse.yaml                    # GKE GCS Fuse PV/PVC manifest
│   ├── stage-deepseek-v41-job.yaml             # Kubernetes Job to stream 510.31 GB checkpoint to GCS
│   └── v7x-16-deepseek-v41-flash-1017.yaml     # Multi-host LeaderWorkerSet + Benchmark Job for TPU v7x-16
└── results/                                    # Auto-synced JSON & Markdown benchmark outputs
```

```bash
# 1. Stage deepseek-ai/DeepSeek-V4.1-Flash-1017 (510.31 GB, 48 shards) to GCS
kubectl apply -f k8s/stage-deepseek-v41-job.yaml

# 2. Deploy Benchmark Driver ConfigMap (0%, 80%, and 95% prefix-cache hit + InferenceX)
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py

# 3. Launch Multi-Host vLLM Server & Benchmark Sweep on TPU v7x-16 (2x2x4)
kubectl apply -f k8s/v7x-16-deepseek-v41-flash-1017.yaml
```
