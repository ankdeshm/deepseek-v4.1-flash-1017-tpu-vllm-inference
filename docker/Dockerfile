ARG BASE_IMAGE=vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl jq ca-certificates procps && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "huggingface_hub[hf_transfer]" \
      hf_transfer \
      google-cloud-storage \
      aiohttp \
      requests && \
    (pip install --no-cache-dir "runai-model-streamer>=0.11.0" || true)

RUN git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
    cd /workspace/inferencex && \
    git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd

COPY run_tpu_benchmark_sweep.py /workspace/run_tpu_benchmark_sweep.py
COPY stage_hf_to_gcs.py /workspace/stage_hf_to_gcs.py
RUN chmod +x /workspace/run_tpu_benchmark_sweep.py /workspace/stage_hf_to_gcs.py
