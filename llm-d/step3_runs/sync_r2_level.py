#!/usr/bin/env python3
"""Updates llm-d/STATUS.md with live or completed Step 3 R2 (S2=1024:1024 & S3=8192:1024) results."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_MD = REPO_ROOT / "llm-d" / "STATUS.md"
R2_S2_DIR = REPO_ROOT / "llm-d" / "step3_runs" / "r2_ep8_s2"
R2_S3_DIR = REPO_ROOT / "llm-d" / "step3_runs" / "r2_ep8_s3"

MARKER_START = "<!-- R2_RESULTS_START -->"
MARKER_END = "<!-- R2_RESULTS_END -->"

BANNED_PATTERNS = [
    r"sla-compliant",
    r"sla sweet spot",
    r"matches gb200",
    r"matches b300",
    r"break-even",
    r"closes the gap",
]


def fmt_row(s: dict, is_last: bool) -> str:
    c = s["concurrency"]
    win_s = s["window"]["steady_window_s"]
    tp = s["throughput"]
    gain_s = (
        f" (`{tp['gain_vs_prev_pct']:+.2f}%`)"
        if tp.get("gain_vs_prev_pct") is not None
        else " *(base)*"
    )
    pu_w = s["per_user_decode_tps"]["in_window"]
    pu_a = s["per_user_decode_tps"]["all_requests"]
    tt_a = s["ttft_s"]["all_requests"]
    tt_w = s["ttft_s"]["in_window"]
    tpot_a = s["tpot_ms"]["all_requests"]
    tau = s["dspark"]["mean_accept_len_tau"]
    db = s["decode_batch_30_295"]
    kv = s["kv_cache_usage_pct"]
    pr = s["preemptions"]
    w = s["window"]
    p_pods = s["routing_distribution"].get("prefill_pods", {})
    h0_pct = p_pods.get("h0", {}).get("requests_share_pct", 0.0)
    h1_pct = p_pods.get("h1", {}).get("requests_share_pct", 0.0)
    d_insts = s["routing_distribution"].get("decode_instances", {})
    d0_req = d_insts.get("d0", {}).get("requests_delta", 0.0)
    d1_req = d_insts.get("d1", {}).get("requests_delta", 0.0)
    d_tot = d0_req + d1_req
    d0_pct = (100.0 * d0_req / d_tot) if d_tot > 0 else 0.0
    d1_pct = (100.0 * d1_req / d_tot) if d_tot > 0 else 0.0
    srv_gen = tp.get("server_gen_tok_s_60_290")
    srv_s = f"`{srv_gen:,.1f}`" if srv_gen is not None else "—"
    stop_flag = " **STOP**" if (is_last and s["stop_evaluation"]["should_stop"]) else ""

    return (
        f"| **`c={c}`**{stop_flag} | `{win_s:.0f} s` | **`{tp['steady_out_tok_s']:,.2f}`**{gain_s} | "
        f"**`{tp['out_tok_s_per_decode_chip']:,.2f}`** | **`{tp['out_tok_s_per_total_chip']:,.2f}`** | "
        f"{srv_s} | `{tp['steady_total_tok_s']:,.2f}` | "
        f"`{pu_w['median']:.2f} / {pu_w['p90_low']:.2f} / {pu_w['p95_low']:.2f}` \\| "
        f"`{pu_a['median']:.2f} / {pu_a['p90_low']:.2f} / {pu_a['p95_low']:.2f}` | "
        f"`{tt_a['p50']:.3f} / {tt_a['p95']:.3f} / {tt_a['p99']:.3f}` | "
        f"`{tt_w['p50']:.3f} / {tt_w['p95']:.3f} / {tt_w['p99']:.3f}` | "
        f"`{tpot_a['p50']:.2f} / {tpot_a['p95']:.2f} / {tpot_a['p99']:.2f}` | "
        f"**`{tau:.4f}`** | "
        f"`{db['per_dp_rank_mean']:.2f} / {db['per_dp_rank_median']:.1f} / {db['per_dp_rank_max']:.1f}` (`{db['total_running_mean']:.1f}` tot) | "
        f"`D`: `{kv['decode_steady_mean_pct']:.3f}% / {kv['decode_peak_single_rank_pct']:.3f}%` · "
        f"`P`: `{kv['prefill_steady_mean_pct']:.3f}% / {kv['prefill_peak_single_rank_pct']:.3f}%` | "
        f"`P`:`{h0_pct:.1f}/{h1_pct:.1f}%` · `D`:`{d0_pct:.1f}/{d1_pct:.1f}%` | "
        f"`{pr['prefill']:.0f} / {pr['decode']:.0f}` | "
        f"**`{w['failed_requests']} / {w['total_requests_launched']}`** |"
    )


def load_shape_levels(shape_dir: Path, shape_name: str, target_isl: int, max_c: int, min_stop_c: int) -> list[dict]:
    if not shape_dir.exists():
        return []
    files = sorted(
        shape_dir.glob(f"r2_{shape_name}_c*.json"),
        key=lambda p: int(re.search(r"_c(\d+)\.json$", p.name).group(1)),
    )
    if not files:
        return []
    levels = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    sweep_path = shape_dir / f"r2_{shape_name}_sweep_summary.json"
    wc = int(levels[0].get("workload", {}).get("calibrated_word_count", target_isl)) if levels else target_isl
    sweep_path.write_text(
        json.dumps(
            {
                "run_name": "r2",
                "shape_name": shape_name,
                "ep_per_decode_instance": 8,
                "calibrated_word_count": wc,
                "min_concurrency_for_gain_stop": min_stop_c,
                "completed_concurrencies": [s["concurrency"] for s in levels],
                "stopped_early": levels[-1]["stop_evaluation"]["should_stop"],
                "stop_reasons": levels[-1]["stop_evaluation"]["reasons"],
                "levels": levels,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return levels


def render_shape_subsection(title: str, isl: int, levels: list[dict], max_c: int) -> list[str]:
    if not levels:
        return []
    lines = [
        f"### {title}",
        "",
        f"- **Workload:** Exact `ISL = {isl:,}` tokens (`0.00%` realized prefix cache hit, `/reset_prefix_cache` on all `4` vLLM servers before every level), forced `OSL = 1,024` tokens (`ignore_eos=True, temperature=0.6`), `300.0 s` steady window + full drain per level.",
        "",
        "| Level | Window | Steady Out tok/s (Gain) | Out tok/s / Dec Chip (`/8`) | Out tok/s / Tot Chip (`/16`) | Server Gen tok/s (`60-290s`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \\| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Pod Split (`P h0/h1` · `D b0/b1`) | Preempt (`P / D`) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for i, s in enumerate(levels):
        lines.append(fmt_row(s, is_last=(i == len(levels) - 1)))

    done = levels[-1]["stop_evaluation"]["should_stop"] or (levels[-1]["concurrency"] >= max_c)
    if done:
        peak = max(levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
        tot_f = sum(x["window"]["failed_requests"] for x in levels)
        tot_r = sum(x["window"]["total_requests_launched"] for x in levels)
        if levels[-1]["stop_evaluation"]["should_stop"]:
            stop_str = f"Stopped after `c={levels[-1]['concurrency']}` ({'; '.join(levels[-1]['stop_evaluation']['reasons'])})"
        else:
            stop_str = f"Completed full ladder through `c={levels[-1]['concurrency']}`"
        lines.extend(
            [
                "",
                f"- **Completion:** {stop_str} (`{tot_f} / {tot_r:,}` failures). Peak **`{peak['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (**`{peak['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip**, **`{peak['throughput']['out_tok_s_per_total_chip']:,.2f}`/tot-chip**, `{peak['throughput']['steady_total_tok_s']:,.2f}` total `in+out` tok/s) at `c={peak['concurrency']}` (`DSpark tau = {peak['dspark']['mean_accept_len_tau']:.4f}`).",
                "",
            ]
        )
    else:
        lines.append("")
    return lines


def main() -> int:
    s2_levels = load_shape_levels(R2_S2_DIR, "s2", 1024, 1024, 1024)
    s3_levels = load_shape_levels(R2_S3_DIR, "s3", 8192, 768, 256)
    if not s2_levels and not s3_levels:
        print("No R2 level JSONs found yet.")
        return 0

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    section_lines = [
        MARKER_START,
        "---",
        "",
        "## Step 3 (`R2`) Results — Matched-Shape Sweep (`S2 = 1024:1024` & `S3 = 8192:1024` on Winning `2P2D` `EP=8`)",
        "",
    ]
    if s2_levels:
        section_lines.extend(render_shape_subsection("`R2-S2`: `S2 = 1024 : 1024` (`1:1` Balanced)", 1024, s2_levels, 1024))
    if s3_levels:
        section_lines.extend(render_shape_subsection("`R2-S3`: `S3 = 8192 : 1024` (`8:1` Prefill-Heavy)", 8192, s3_levels, 768))
    section_lines.append(MARKER_END)
    section_block = "\n".join(section_lines)

    s2_done = bool(s2_levels) and (s2_levels[-1]["stop_evaluation"]["should_stop"] or s2_levels[-1]["concurrency"] >= 1024)
    s3_done = bool(s3_levels) and (s3_levels[-1]["stop_evaluation"]["should_stop"] or s3_levels[-1]["concurrency"] >= 768)

    parts_summary = []
    if s2_levels:
        p2 = max(s2_levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
        c2 = ",".join(str(x["concurrency"]) for x in s2_levels)
        f2 = sum(x["window"]["failed_requests"] for x in s2_levels)
        n2 = sum(x["window"]["total_requests_launched"] for x in s2_levels)
        parts_summary.append(
            f"**`S2` (`1024:1024`):** `c={c2}` {'done' if s2_done else 'running'}, peak **`{p2['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (`{p2['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip @ `c={p2['concurrency']}`, `{f2}/{n2:,}` err)"
        )
    if s3_levels:
        p3 = max(s3_levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
        c3 = ",".join(str(x["concurrency"]) for x in s3_levels)
        f3 = sum(x["window"]["failed_requests"] for x in s3_levels)
        n3 = sum(x["window"]["total_requests_launched"] for x in s3_levels)
        parts_summary.append(
            f"**`S3` (`8192:1024`):** `c={c3}` {'done' if s3_done else 'running'}, peak **`{p3['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (`{p3['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip, `{p3['throughput']['steady_in_tok_s']:,.0f}` in tok/s @ `c={p3['concurrency']}`, `{f3}/{n3:,}` err)"
        )

    if s2_done and s3_done:
        status_cell = "**COMPLETE (`S2` & `S3` Done)**"
        next_cell = "Publish `r2_summary.md` & launch Step 3 `R3` (`Decode-Only Ceiling`)."
    elif s2_done and not s3_levels:
        status_cell = "**IN PROGRESS (`S2` Done, Starting `S3`)**"
        next_cell = "Run `R2-S3` (`8192:1024`, `c=32..768`)."
    elif s2_done:
        status_cell = "**IN PROGRESS (`S2` Done, `S3` Running)**"
        next_cell = "Complete `R2-S3` (`8192:1024`), then launch `R3`."
    else:
        status_cell = "**IN PROGRESS (`S2` Running)**"
        next_cell = "Complete `R2-S2` (`1024:1024`), then `R2-S3` (`8192:1024`)."

    r2_row = f"| `{now_utc[:16]}Z` | **STEP 3 `R2`: Matched-Shape Sweep (`S2=1024:1024`, `S3=8192:1024`, `2P2D` `EP=8`)** | {status_cell} | {' · '.join(parts_summary)}. | {next_cell} |"

    txt = STATUS_MD.read_text(encoding="utf-8")
    txt = re.sub(
        r"- \*\*Last Updated \(UTC\):\*\* `[^`]+`",
        f"- **Last Updated (UTC):** `{now_utc}`",
        txt,
    )

    if re.search(r"\| `[^`]+` \| \*\*STEP 3 `R2`:[^\n]+", txt):
        txt = re.sub(r"\| `[^`]+` \| \*\*STEP 3 `R2`:[^\n]+", lambda _: r2_row, txt)
    else:
        txt = re.sub(
            r"(\| `[^`]+` \| \*\*STEP 3 `R1b`:[^\n]+)",
            lambda m: m.group(1) + "\n" + r2_row,
            txt,
        )

    if MARKER_START in txt and MARKER_END in txt:
        txt = re.sub(
            rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
            lambda _: section_block,
            txt,
            flags=re.S,
        )
    else:
        txt = txt.rstrip() + "\n\n" + section_block + "\n"

    assert "diesel-patrol-382622" not in txt
    assert "ank-usc1-bucket-314837540096" not in txt
    lower_txt = txt.lower()
    for pat in BANNED_PATTERNS:
        assert not re.search(pat, lower_txt), f"Banned phrase matched: {pat}"

    STATUS_MD.write_text(txt, encoding="utf-8")
    print(f"Updated {STATUS_MD} (S2 levels={len(s2_levels)}, S3 levels={len(s3_levels)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
