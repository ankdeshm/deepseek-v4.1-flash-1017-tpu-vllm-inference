# STEP 1: Reproducibility Anchor (`R0`) Summary — DeepSeek-V4.1-Flash-1017 on TPU v7x (`Pool A`)

**Date (UTC):** 2026-11-07
**Cluster / Pool:** `gke_YOUR_GCP_PROJECT_us-central1_ank-tpu-v7x` · `ank-tpu-v7x-8-pool-a` (`2x2x2` = 8 TPU v7x chips / 16 TensorCores)
**Pinned Image:** `docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6` + `dsv41-stack/stack_27be32f509754e88.tar` (`libtpu==0.0.48`, `PYTHONHASHSEED=0`)
**PASS Gate (`6,700 – 8,200` window out tok/s, `DSpark` $\tau > 2.0$):** **PASS (`7,423.72` out tok/s, `+0.20%` vs `A1-b` `7,408.59`; `DSpark` $\tau = 2.8755$)**

---

## 1. Rule 4 Full Metrics Table (`R0` vs Prior `A1-b` Reference)

| Run | Workload / Shape | Topology & Hosts | EP / Role Config | Concurrency (`n`) | Steady Window | Total Out tok/s (Win / Drain) | Out tok/s / Decode Chip (`/4`) | Out tok/s / Total Chip (`/8`) | Total In+Out tok/s (Win) | Per-User tok/s `Med / P90_low / P95_low` (In-Win \| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | Realized Cache Hit % | Mean ISL / OSL (tok) | `DSpark` $\tau$ | KV Usage % `Steady / Peak-Rank` | Preemptions (`P / D`) | Failures |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---:|---:|
| **`R0` (Pass 2, Warm)** | Empirical `50:20:12:9:9` ISL, 40 trunks, 95% target hit, exact temps, `--no-ignore-eos` | `1P1D` (`toy_proxy_server` + Raiden Stage 3): `P`=`rqkr` (`4c`, `2x2x1`), `D`=`xq6s` (`4c`, `2x2x1`) | `P`: `DP=8, TP=1, EP=8, CED=1, 64/8192` · `D`: `DP=8, TP=1, EP=8, CED=0, 48/288, DSpark K=5, async=1` | `128` | `300.0 s` | **`7,423.72`** / `6,620.90` | **`1,855.93`** | **`927.97`** | **`283,403.02`** (`275,979.29` in) | `67.83 / 54.65 / 47.43` \| `67.74 / 54.44 / 47.37` (**GATE-COMPLIANT**) | `1.205 / 1.556 / 8.726` | `1.205 / 1.559 / 8.731` | **`95.47%`** | `17,916.9` / `488.8` | **`2.8755`** | `D`: `9.02% / 13.57%` · `P`: `0.82% / 7.46%` | `0 / 0` | **`0 / 4,631`** |
| **`R0` (Pass 1, Cold-Sampler)** | Same (`50:20:12:9:9`, 40 trunks, 95% hit) | Same (`Pool A` `rqkr` + `xq6s`) | Same | `128` | `300.0 s` | `7,030.59` / `5,929.90` | `1,757.65` | `878.82` | `269,712.65` (`262,682.07` in) | `66.17 / 48.91 / 43.16` \| `65.96 / 48.11 / 42.06` | `1.207 / 1.718 / 16.838` | `1.206 / 1.741 / 16.842` | `95.47%` | `17,909.7` / `488.3` | `2.9192` | `D`: `8.28% / 13.85%` · `P`: `0.74% / 7.15%` | `0 / 0` | `0 / 4,413` |
| **Prior `A1-b` Ref** | Same (`50:20:12:9:9`, 40 trunks, 95% hit) | `1P1D` (`4c` P + `4c` D) | Same (`48/288, DSpark K=5, async=1`) | `128` | `180.0 s` | `7,408.59` / `6,439.98` | `1,852.15` | `926.07` | `287,099.52` (`279,690.93` in) | `66.64 / 54.32 / 47.78` \| `66.62 / 53.91 / 47.74` | `1.205 / 1.613 / 3.491` | `1.201 / — / 3.499` | `95.47%` | `17,907.1` / `487.6` | `2.8699` | — | `0 / 0` | `0 / 2,828` |

---

## 2. Per-ISL-Bucket Breakdown (`R0` Pass 2, `4,631` Completed Requests)

| ISL Bucket | Configured Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Realized Cache Hit % | Median TTFT (s) | P90 TTFT (s) | Median E2E (s) | Per-User Decode tok/s (`Med / P90_low / P95_low`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`1-8K` (50%)** | `8` | `2,315` | `6,293.9` | `349.2` | `97.62%` | `1.177` | `1.457` | `6.226` | `65.46 / 47.73 / 43.66` |
| **`8-16K` (20%)** | `8` | `926` | `10,467.1` | `438.3` | `97.83%` | `1.200` | `1.475` | `6.739` | `72.81 / 60.43 / 58.23` |
| **`16-32K` (12%)** | `8` | `556` | `22,240.3` | `741.0` | `96.69%` | `1.191` | `1.488` | `8.665` | `69.74 / 59.97 / 57.94` |
| **`32-64K` (9%)** | `8` | `417` | `48,952.9` | `828.8` | `94.13%` | `1.243` | `1.531` | `9.974` | `67.67 / 59.68 / 57.10` |
| **`64K+` (9%)** | `8` | `417` | `62,185.5` | `699.7` | `93.86%` | `1.294` | `1.582` | `9.496` | `66.47 / 58.98 / 55.79` |

---

## 3. Prometheus Engine Diagnostics & Per-DP-Rank Load Balance (`R0` Pass 2)

- **`DSpark` Speculative Decoding (`K=5`):**
  - Draft steps: `788,429` · Draft tokens proposed: `3,942,145` · Draft tokens accepted: `1,478,678`
  - **Mean acceptance length:** $\tau = 1 + \frac{1,478,678}{788,429} = \mathbf{2.8755}$ (`PASS > 2.0`)
  - Per-position acceptance probabilities (`pos 0..4`): **`[0.5680, 0.4255, 0.3421, 0.2881, 0.2518]`**
- **KV Cache Capacity & Utilization (`gpu_memory_utilization=0.85`, FP8 KV):**
  - **Prefill (`192.168.100.43:8400`):** `1,464,212` tok/rank (`11,713,696` tok total across 8 DP ranks); steady mean `0.82%`, peak rank-mean `7.19%`, peak single-rank `7.46%`; `0` preemptions.
  - **Decode (`192.168.100.38:9400`):** `3,980,492` tok/rank (`31,843,936` tok total across 8 DP ranks); steady mean **`9.02%`**, peak rank-mean `9.96%`, peak single-rank **`13.57%`**; `0` preemptions.
- **Per-DP-Rank Running Batch Size & Request Distribution (`engine=0..7`):**
  - **Decode steady running batch size (`108.91` total mean, `120` max across 8 ranks):**
    - `DP0`: mean `13.47` (med `14`, max `16`) · `618` reqs · `279,476` gen tok
    - `DP1`: mean `13.60` (med `14`, max `16`) · `598` reqs · `283,178` gen tok
    - `DP2`: mean `13.65` (med `14`, max `16`) · `597` reqs · `283,969` gen tok
    - `DP3`: mean `13.67` (med `14`, max `16`) · `589` reqs · `285,950` gen tok
    - `DP4`: mean `13.53` (med `14`, max `16`) · `618` reqs · `280,015` gen tok
    - `DP5`: mean `13.31` (med `13`, max `16`) · `644` reqs · `278,026` gen tok
    - `DP6`: mean `13.82` (med `14`, max `16`) · `581` reqs · `283,771` gen tok
    - `DP7`: mean `13.85` (med `14`, max `16`) · `595` reqs · `295,569` gen tok
  - **Prefill completed requests (`engine=0..7`, including warmup):** `[611, 626, 557, 618, 596, 615, 604, 613]` (measured pass client-pinned `[584, 600, 531, 592, 570, 589, 578, 587]`).

---

## 4. Artifact Inventory

All manifests, scripts, raw server logs, Prometheus `/metrics` snapshots, timeseries, per-request timings, and summaries are stored at:
- **Local:** `/Users/deshmukhankita/ankita-dev/github-repos/deepseek-v4.1-flash-1017-tpu-vllm-inference/llm-d/step1_r0/`
- **GCS:** `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/step1_r0/`
  - `step1_r0_pods.yaml`, `step1_r0_driver.py`, `run_r0_bench_with_metrics.py`
  - `r0_n128_300s.json`, `r0_requests_n128.json`, `metrics_summary.json`, `metrics_timeseries.jsonl`
  - `prefill_metrics_{before,mid,after}.txt`, `decode_metrics_{before,mid,after}.txt`
  - `pd_prefill.log`, `pd_decode.log`, `pd_proxy.log`, `pd_setup_prefill.log`, `pd_setup_decode.log`
  - `pass1/` (archived Pass 1 300s run + metrics)
  - Warm compile-cache recovery helper in `step1_r0_driver.py` (`restore_warm_compile_cache()`) for `gs://YOUR_GCS_BUCKET/dsv41-v7x-llmd-20261106/compile_cache/compile_cache_{prefill,decode}_dp8.tar.gz`
