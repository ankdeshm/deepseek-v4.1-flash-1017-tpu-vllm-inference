#!/usr/bin/env python3
"""First-Principles Roofline, MFU & HBM-Bandwidth Analyzer on Cloud TPU v7x (Ironwood).

Covers both models in the Atlas Cloud TPU v7x Benchmark Rerun Pack using exact `config.json`
architecture constants and actual serving sharding layouts:
  1. DeepSeek-V4.1-Flash-1017 (40L Sparse-CSA/SWA + 384 MXFP4 Routed Experts + 3 DSpark K=5 layers,
     served on vLLM + TorchTPU with DP=8, TP=1, EP=8 per 4-chip 2x2x1 slice)
  2. GLM-5.3 FP8 (78L = 3 Dense + 75 MoE x 256 FP8 Experts + DSA Indexer topk=2048 + 1 Eagle3 MTP,
     served on SGLang-JAX 6a43163 with DP=16, TP=2, EP=32 on 16-chip 2x2x4 or DP=8, TP=2, EP=16 on 8-chip)

Reports both:
  - Logical (single-copy) vs Actual Sharded Per-Chip HBM Bandwidth Utilization (% of 7,373 GB/s/chip)
  - Achieved Compute MFU (% of 4,614 TFLOP/s FP8 peak AND % of 2,307 TFLOP/s BF16 peak per v7x chip)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ==============================================================================
# Cloud TPU v7x (Ironwood) Per-Chip Public Hardware Specifications
# 2 TensorCores/chip, 192 GiB HBM3e, 7,373 GB/s (7.373 TB/s) HBM3e bandwidth,
# 2,307 TFLOP/s BF16 peak, and 4,614 TFLOP/s FP8/FP4 MXU peak.
# ==============================================================================
V7X_BF16_TFLOPS = 2307.0
V7X_FP8_TFLOPS = 4614.0
V7X_HBM_GIB = 192.0
V7X_HBM_BW_GBPS = 7373.0  # GB/s (1e9 B/s) per chip

# ==============================================================================
# 1. GLM-5.3 FP8 Architecture Constants (exact from zai-org/GLM-5.3/config.json)
# ==============================================================================
GLM_H = 6144
GLM_LAYERS = 78          # 3 dense + 75 MoE (+ 1 NEXTN Eagle3 draft layer)
GLM_DENSE_LAYERS = 3
GLM_MOE_LAYERS = 75
GLM_N_EXPERTS = 256
GLM_TOP_K = 8
GLM_MOE_INTER = 2048
GLM_DENSE_INTER = 12288
GLM_VOCAB = 154880
GLM_HEADS = 64
GLM_Q_LORA, GLM_KV_LORA = 2048, 512
GLM_QK_NOPE, GLM_QK_ROPE, GLM_V_HEAD = 192, 64, 256
GLM_IDX_HEADS, GLM_IDX_DIM = 32, 128
GLM_IDX_TOPK = 2048
GLM_IDX_FULL_LAYERS = 21  # 21 "full" + 57 "shared" in config.json indexer_types

GLM_ATTN_PARAMS_PER_LAYER = (
    GLM_H * GLM_Q_LORA
    + GLM_Q_LORA * GLM_HEADS * (GLM_QK_NOPE + GLM_QK_ROPE)
    + GLM_H * (GLM_KV_LORA + GLM_QK_ROPE)
    + GLM_KV_LORA * GLM_HEADS * (GLM_QK_NOPE + GLM_V_HEAD)
    + GLM_HEADS * GLM_V_HEAD * GLM_H
    + GLM_Q_LORA * GLM_IDX_HEADS * GLM_IDX_DIM
    + GLM_H * GLM_IDX_DIM
    + GLM_H * GLM_IDX_HEADS
)
GLM_EXPERT_PARAMS = 3 * GLM_H * GLM_MOE_INTER
GLM_DENSE_MLP_PARAMS = 3 * GLM_H * GLM_DENSE_INTER
GLM_ROUTER_PARAMS = GLM_N_EXPERTS * GLM_H
GLM_EMBED_PARAMS = GLM_VOCAB * GLM_H

GLM_P_ATTN = GLM_LAYERS * GLM_ATTN_PARAMS_PER_LAYER
GLM_P_ROUTED = GLM_MOE_LAYERS * GLM_N_EXPERTS * GLM_EXPERT_PARAMS
GLM_P_SHARED = GLM_MOE_LAYERS * GLM_EXPERT_PARAMS
GLM_P_DENSE = GLM_DENSE_LAYERS * GLM_DENSE_MLP_PARAMS
GLM_P_ROUTER = GLM_MOE_LAYERS * GLM_ROUTER_PARAMS
GLM_P_EMB = 2 * GLM_EMBED_PARAMS
GLM_P_TOTAL = GLM_P_ATTN + GLM_P_ROUTED + GLM_P_SHARED + GLM_P_DENSE + GLM_P_ROUTER + GLM_P_EMB

GLM_P_ACTIVE = (
    GLM_P_ATTN
    + GLM_MOE_LAYERS * GLM_TOP_K * GLM_EXPERT_PARAMS
    + GLM_P_SHARED
    + GLM_P_DENSE
    + GLM_P_ROUTER
    + GLM_EMBED_PARAMS
)
GLM_MTP_LAYER_ACTIVE = (
    GLM_ATTN_PARAMS_PER_LAYER
    + (GLM_TOP_K + 1) * GLM_EXPERT_PARAMS
    + GLM_ROUTER_PARAMS
    + GLM_EMBED_PARAMS
)
GLM_KV_BYTES_PER_TOK = GLM_LAYERS * (GLM_KV_LORA + GLM_QK_ROPE) + GLM_IDX_FULL_LAYERS * GLM_IDX_DIM

# ==============================================================================
# 2. DeepSeek-V4.1-Flash-1017 Constants (exact from DeepSeek-V4.1-Flash/config.json)
#    40 MoE layers (384 routed MXFP4 experts + 1 shared FP8 expert, moe_inter=2304),
#    Compressed Sparse Attention (CSA + SWA=128, kv_source_layer_ids=[2,8,14,20],
#    index_source_layer_ids=8 layers, index_topk=512), + 3 DSpark layers (K=5).
# ==============================================================================
DS_H = 5120
DS_LAYERS = 40
DS_MOE_LAYERS = 40
DS_N_EXPERTS = 384
DS_TOP_K = 6
DS_MOE_INTER = 2304
DS_VOCAB = 129280
DS_HEADS = 64
DS_Q_LORA, DS_KV_LORA, DS_O_LORA = 1280, 512, 1024
DS_QK_NOPE, DS_QK_ROPE, DS_V_HEAD = 128, 64, 128
DS_IDX_HEADS, DS_IDX_DIM = 32, 128
DS_IDX_LAYERS = 8
DS_IDX_TOPK = 512
DS_SWA_WINDOW = 128
MXFP4_BYTES = 0.5 + 1.0 / 32.0  # 4-bit + ue8m0 block-32 scale = 0.53125 B/param

DS_ATTN_PARAMS_PER_LAYER = (
    DS_H * DS_Q_LORA
    + DS_Q_LORA * DS_HEADS * (DS_QK_NOPE + DS_QK_ROPE)
    + DS_H * (DS_KV_LORA + DS_QK_ROPE)
    + DS_KV_LORA * DS_HEADS * (DS_QK_NOPE + DS_V_HEAD)
    + DS_HEADS * DS_V_HEAD * DS_O_LORA
    + DS_O_LORA * DS_H
)
DS_IDX_PARAMS_TOTAL = DS_IDX_LAYERS * (DS_Q_LORA * DS_IDX_HEADS * DS_IDX_DIM + DS_H * DS_IDX_DIM)
DS_EXPERT_PARAMS = 3 * DS_H * DS_MOE_INTER
DS_ROUTER_PARAMS = DS_N_EXPERTS * DS_H
DS_EMBED_PARAMS = DS_VOCAB * DS_H

DS_P_ATTN = DS_LAYERS * DS_ATTN_PARAMS_PER_LAYER + DS_IDX_PARAMS_TOTAL
DS_P_ROUTED = DS_MOE_LAYERS * DS_N_EXPERTS * DS_EXPERT_PARAMS
DS_P_SHARED = DS_MOE_LAYERS * DS_EXPERT_PARAMS
DS_P_ROUTER = DS_MOE_LAYERS * DS_ROUTER_PARAMS
DS_P_EMB = 2 * DS_EMBED_PARAMS
DS_P_TOTAL = DS_P_ATTN + DS_P_ROUTED + DS_P_SHARED + DS_P_ROUTER + DS_P_EMB

DS_P_ACTIVE = (
    DS_P_ATTN
    + DS_MOE_LAYERS * DS_TOP_K * DS_EXPERT_PARAMS
    + DS_P_SHARED
    + DS_P_ROUTER
    + DS_EMBED_PARAMS
)
# Measured from tpu_connector.py raiden_stage3_dsv4_request_blocks_registered:
# 22,241 tokens -> 26.09 MB variable CSA+Indexer pages (~1,173 B/tok) + 2.62 MB fixed SWA/CSA-state pages
DS_KV_VARIABLE_BYTES_PER_TOK = 1173
DS_KV_FIXED_STATE_BYTES_PER_SEQ = 2_621_440


def glm_prefill_flops_per_token(context_len: int) -> tuple[float, float]:
    """Returns (fp8_matmul_flops, bf16_attn_flops) per prompt token at average causal prefix ~ context_len/2."""
    avg_ctx = max(1, context_len // 2)
    sel = min(avg_ctx, GLM_IDX_TOPK)
    core_attn = GLM_LAYERS * GLM_HEADS * (2 * GLM_KV_LORA + GLM_QK_ROPE) * 2 * sel
    idx_attn = GLM_IDX_FULL_LAYERS * GLM_IDX_HEADS * GLM_IDX_DIM * 2 * avg_ctx
    return 2.0 * GLM_P_ACTIVE, float(core_attn + idx_attn)


def ds_prefill_flops_per_token(context_len: int) -> tuple[float, float]:
    """Returns (fp8_fp4_matmul_flops, bf16_attn_flops) per prompt token for DeepSeek-V4.1-Flash."""
    avg_ctx = max(1, context_len // 2)
    sel_csa = min(avg_ctx, DS_IDX_TOPK)
    sel_swa = min(avg_ctx, DS_SWA_WINDOW)
    # 2 layers full/SWA, 38 layers CSA+SWA hybrid + 8 indexer layers
    csa_attn = 38 * DS_HEADS * (2 * DS_KV_LORA + DS_QK_ROPE) * 2 * sel_csa
    swa_attn = 40 * DS_HEADS * (DS_QK_NOPE + DS_QK_ROPE + DS_V_HEAD) * 2 * sel_swa
    idx_attn = DS_IDX_LAYERS * DS_IDX_HEADS * DS_IDX_DIM * 2 * avg_ctx
    return 2.0 * DS_P_ACTIVE, float(csa_attn + swa_attn + idx_attn)


def glm_spec_step_hbm_per_chip(
    active_seqs: int,
    context_len: int,
    num_chips: int = 16,
    dp_size: int = 16,
    verify_tokens_per_seq: int = 5,
    mtp_steps: int = 4,
) -> dict[str, float]:
    """Computes BOTH per-chip sharded HBM bytes read AND logical single-copy HBM bytes per EAGLE-3 step.

    On 16 chips (dp=16, tp=2, ep=32):
      - Routed experts (256 experts x 75 layers = 708 GB FP8) are sharded 32-way across 16 chips -> 44.24 GB/chip max.
      - Non-routed weights (MLA + Shared Expert + Dense L0-2 + LM Head + Embed) are TP=2 sharded within each
        1-chip DP replica (replicated 16-way across the 16 chips) -> ~25.9 GB/chip read on active DP ranks!
      - 4 MTP draft steps run on the single Eagle3 draft layer (replicated non-routed + EP=32 routed).
    """
    q_tokens_global = active_seqs * verify_tokens_per_seq
    touched_frac = 1.0 - (1.0 - GLM_TOP_K / GLM_N_EXPERTS) ** max(1, q_tokens_global)

    # Logical single-copy bytes (1e9 GB)
    routed_logical_gb = (GLM_MOE_LAYERS * GLM_N_EXPERTS * touched_frac * GLM_EXPERT_PARAMS) / 1e9
    non_routed_one_copy_gb = (
        (GLM_P_SHARED + GLM_P_ATTN + GLM_P_DENSE + GLM_P_ROUTER) * 1.0 + GLM_EMBED_PARAMS * 2.0
    ) / 1e9
    kv_total_gb = (
        active_seqs
        * (
            GLM_LAYERS * min(context_len, GLM_IDX_TOPK) * (GLM_KV_LORA + GLM_QK_ROPE)
            + GLM_IDX_FULL_LAYERS * context_len * GLM_IDX_DIM
        )
    ) / 1e9

    draft_touched = 1.0 - (1.0 - GLM_TOP_K / GLM_N_EXPERTS) ** max(1, active_seqs)
    draft_routed_logical_gb = mtp_steps * (GLM_N_EXPERTS * draft_touched * GLM_EXPERT_PARAMS) / 1e9
    draft_non_routed_one_copy_gb = (
        mtp_steps * (GLM_EXPERT_PARAMS + GLM_ATTN_PARAMS_PER_LAYER + GLM_ROUTER_PARAMS + GLM_EMBED_PARAMS * 2.0)
    ) / 1e9

    # Per-chip physical HBM bytes on an active DP replica chip (1 DP rank = num_chips / dp_size chips = 1 chip for dp=16)
    chips_per_dp = num_chips / dp_size
    routed_per_chip_gb = routed_logical_gb / num_chips
    non_routed_per_chip_gb = non_routed_one_copy_gb / chips_per_dp
    kv_per_chip_gb = kv_total_gb / num_chips
    # Note: In EAGLE-3 MTP, LM head/embed is shared across verify + draft, read per step
    draft_per_chip_gb = (draft_routed_logical_gb / num_chips) + (draft_non_routed_one_copy_gb / chips_per_dp)

    per_chip_total_gb = routed_per_chip_gb + non_routed_per_chip_gb + kv_per_chip_gb + draft_per_chip_gb
    logical_total_gb = (
        routed_logical_gb
        + non_routed_one_copy_gb
        + kv_total_gb
        + draft_routed_logical_gb
        + draft_non_routed_one_copy_gb
    )
    return {
        "routed_experts_touched_pct": round(100.0 * touched_frac, 2),
        "per_chip_verify_routed_gb": round(routed_per_chip_gb, 2),
        "per_chip_verify_replicated_non_routed_gb": round(non_routed_per_chip_gb, 2),
        "per_chip_verify_kv_gb": round(kv_per_chip_gb, 3),
        "per_chip_draft_4steps_gb": round(draft_per_chip_gb, 2),
        "per_chip_total_hbm_read_gb": round(per_chip_total_gb, 2),
        "per_chip_ideal_hbm_ms": round(1000.0 * per_chip_total_gb / V7X_HBM_BW_GBPS, 3),
        "logical_single_copy_total_gb": round(logical_total_gb, 2),
    }


def analyze_glm_b1(b1_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(b1_path.read_text())
    out = []
    for r in rows:
        if r.get("mode") != "prefill_uncached":
            continue
        isl = int(r["target_isl"])
        chips = int(r["num_chips"])
        tps = float(r["uncached_tok_per_s"])
        fp8_f, bf16_f = glm_prefill_flops_per_token(isl)
        total_flops_per_tok = fp8_f + bf16_f
        achieved_tflops_per_chip = (tps * total_flops_per_tok / chips) / 1e12
        ideal_mixed_tps = 1.0 / (
            fp8_f / (chips * V7X_FP8_TFLOPS * 1e12) + bf16_f / (chips * V7X_BF16_TFLOPS * 1e12)
        )
        out.append({
            "label": r.get("label", ""),
            "isl": isl,
            "concurrency": r["concurrency"],
            "measured_uncached_tok_per_s": tps,
            "measured_tok_per_s_per_chip": r["uncached_tok_per_s_per_chip"],
            "achieved_tflops_per_chip": round(achieved_tflops_per_chip, 1),
            "mfu_vs_fp8_4614_pct": round(100.0 * achieved_tflops_per_chip / V7X_FP8_TFLOPS, 2),
            "mfu_vs_mixed_fp8_bf16_roofline_pct": round(100.0 * tps / ideal_mixed_tps, 2),
            "ideal_mixed_roofline_tok_per_s": round(ideal_mixed_tps, 1),
            "ttft_p50_s": r["ttft_s"]["p50"],
        })
    return out


def analyze_dsv41_a4(a4_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(a4_path.read_text())
    out = []
    for r in rows:
        if r.get("mode") != "prefill_uncached":
            continue
        isl = int(r["target_isl"])
        chips = int(r["num_chips"])
        tps = float(r["uncached_tok_per_s"])
        fp_f, bf16_f = ds_prefill_flops_per_token(isl)
        total_flops_per_tok = fp_f + bf16_f
        achieved_tflops_per_chip = (tps * total_flops_per_tok / chips) / 1e12
        ideal_mixed_tps = 1.0 / (
            fp_f / (chips * V7X_FP8_TFLOPS * 1e12) + bf16_f / (chips * V7X_BF16_TFLOPS * 1e12)
        )
        out.append({
            "label": r.get("label", ""),
            "isl": isl,
            "concurrency": r["concurrency"],
            "measured_uncached_tok_per_s": tps,
            "measured_tok_per_s_per_chip": r["uncached_tok_per_s_per_chip"],
            "achieved_tflops_per_chip": round(achieved_tflops_per_chip, 1),
            "mfu_vs_fp8_4614_pct": round(100.0 * achieved_tflops_per_chip / V7X_FP8_TFLOPS, 2),
            "mfu_vs_mixed_fp8_bf16_roofline_pct": round(100.0 * tps / ideal_mixed_tps, 2),
            "ideal_mixed_roofline_tok_per_s": round(ideal_mixed_tps, 1),
            "ttft_p50_s": r["ttft_s"]["p50"],
        })
    return out


def analyze_glm_b2(b2_path: Path, step_timing_by_bs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = json.loads(b2_path.read_text())
    out = []
    for r in rows:
        n = int(r["num_sessions"])
        chips = int(r["num_chips"])
        win_tps = float(r["throughput"]["window_output_tok_per_s"])
        ctx = int(r["mean_prompt_tokens"]) + int(r["mean_completion_tokens"] * 0.5)
        hbm = glm_spec_step_hbm_per_chip(n, max(128, ctx), num_chips=chips, dp_size=16)

        meas_ms = None
        if step_timing_by_bs and str(n) in step_timing_by_bs:
            meas_ms = float(step_timing_by_bs[str(n)]["total_step_ms"]["p50"])
        bw_util = round(100.0 * hbm["per_chip_ideal_hbm_ms"] / meas_ms, 2) if meas_ms else None

        # Output MFU based on active FLOPs per emitted token (~2 * GLM_P_ACTIVE)
        achieved_tflops_chip = (win_tps * 2.0 * GLM_P_ACTIVE / chips) / 1e12
        out.append({
            "target_isl": r["target_decode_isl"],
            "num_sessions": n,
            "batch_per_chip": r["batch_per_chip"],
            "window_output_tok_per_s": win_tps,
            "window_output_tok_per_s_per_chip": r["throughput"]["window_output_tok_per_s_per_chip"],
            "stream_tps_p50": r["stream_tps_all_ok"]["p50"],
            "stream_tps_p90_low": r["stream_tps_all_ok"]["p90_low"],
            **hbm,
            "measured_step_p50_ms": meas_ms,
            "achieved_sharded_hbm_bw_util_pct": bw_util,
            "decode_mfu_vs_fp8_4614_pct": round(100.0 * achieved_tflops_chip / V7X_FP8_TFLOPS, 2),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-json", nargs="*", default=[])
    ap.add_argument("--b2-json", nargs="*", default=[])
    ap.add_argument("--b2-step-json", default="")
    ap.add_argument("--a4-json", nargs="*", default=[])
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    step_timing = None
    if args.b2_step_json and Path(args.b2_step_json).exists():
        step_timing = json.loads(Path(args.b2_step_json).read_text()).get("spec_step_timing_by_real_bs")

    summary: dict[str, Any] = {
        "hardware_v7x_per_chip": {
            "tensorcores": 2,
            "hbm3e_gib": V7X_HBM_GIB,
            "hbm3e_bw_gbps": V7X_HBM_BW_GBPS,
            "fp8_peak_tflops": V7X_FP8_TFLOPS,
            "bf16_peak_tflops": V7X_BF16_TFLOPS,
        },
        "glm53_constants": {
            "total_params_b": round(GLM_P_TOTAL / 1e9, 2),
            "active_params_b": round(GLM_P_ACTIVE / 1e9, 2),
            "mtp_draft_active_params_b": round(GLM_MTP_LAYER_ACTIVE / 1e9, 2),
            "fp8_kv_bytes_per_token": GLM_KV_BYTES_PER_TOK,
        },
        "dsv41_constants": {
            "total_params_b": round(DS_P_TOTAL / 1e9, 2),
            "active_params_b": round(DS_P_ACTIVE / 1e9, 2),
            "csa_idx_variable_kv_bytes_per_token": DS_KV_VARIABLE_BYTES_PER_TOK,
            "swa_csa_fixed_state_bytes_per_seq": DS_KV_FIXED_STATE_BYTES_PER_SEQ,
        },
        "glm53_b1_prefill_mfu": [],
        "glm53_b2_decode_hbm_and_mfu": [],
        "dsv41_a4_prefill_mfu": [],
    }

    for p_str in args.b1_json:
        p = Path(p_str)
        if p.exists():
            summary["glm53_b1_prefill_mfu"].extend(analyze_glm_b1(p))
    for p_str in args.b2_json:
        p = Path(p_str)
        if p.exists():
            summary["glm53_b2_decode_hbm_and_mfu"].extend(analyze_glm_b2(p, step_timing))
    for p_str in args.a4_json:
        p = Path(p_str)
        if p.exists():
            summary["dsv41_a4_prefill_mfu"].extend(analyze_dsv41_a4(p))

    print(json.dumps(summary["glm53_constants"], indent=2))
    print(json.dumps(summary["dsv41_constants"], indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"Saved roofline analysis to {args.output_json}")


if __name__ == "__main__":
    main()
