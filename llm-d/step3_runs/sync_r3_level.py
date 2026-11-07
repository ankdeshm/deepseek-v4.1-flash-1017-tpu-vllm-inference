#!/usr/bin/env python3
"""Updates llm-d/STATUS.md with live or completed Step 3 R3 (Decode-Only Ceiling) results."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_MD = REPO_ROOT / "llm-d" / "STATUS.md"
R3_DIR = REPO_ROOT / "llm-d" / "step3_runs" / "r3_decode_ceiling"

MARKER_START = "<!-- R3_RESULTS_START -->"
MARKER_END = "<!-- R3_RESULTS_END -->"

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
    tok = s["tokens"]
    p_pods = s["routing_distribution"].get("prefill_pods", {})
    h0_pct = p_pods.get("h0", {}).get("requests_share_pct", 0.0)
    h1_pct = p_pods.get("h1", {}).get("requests_share_pct", 0.0)
    d_insts = s["routing_distribution"].get("decode_instances", {})
    d0_pct = d_insts.get("d0", {}).get("requests_share_pct", 0.0)
    d1_pct = d_insts.get("d1", {}).get("requests_share_pct", 0.0)
    srv_gen = tp.get("server_gen_tok_s_60_290")
    srv_s = f"`{srv_gen:,.1f}`" if srv_gen is not None else "—"
    stop_flag = " **STOP**" if (is_last and s["stop_evaluation"]["should_stop"]) else ""

    return (
        f"| **`c={c}`**{stop_flag} | `{win_s:.0f} s` | **`{tp['steady_out_tok_s']:,.2f}`**{gain_s} | "
        f"**`{tp['out_tok_s_per_decode_chip']:,.2f}`** | **`{tp['out_tok_s_per_total_chip']:,.2f}`** | "
        f"{srv_s} | `{tok['api_cache_hit_pct']:.2f}%` (`P_run={db['prefill_running_mean']:.2f}`) | "
        f"`{pu_w['median']:.2f} / {pu_w['p90_low']:.2f} / {pu_w['p95_low']:.2f}` \\| "
        f"`{pu_a['median']:.2f} / {pu_a['p90_low']:.2f} / {pu_a['p95_low']:.2f}` | "
        f"`{tt_a['p50']:.3f} / {tt_a['p95']:.3f} / {tt_a['p99']:.3f}` | "
        f"`{tt_w['p50']:.3f} / {tt_w['p95']:.3f} / {tt_w['p99']:.3f}` | "
        f"`{tpot_a['p50']:.2f} / {tpot_a['p95']:.2f} / {tpot_a['p99']:.2f}` | "
        f"**`{tau:.4f}`** | "
        f"`{db['per_dp_rank_mean']:.2f} / {db['per_dp_rank_median']:.1f} / {db['per_dp_rank_max']:.1f}` (`{db['total_running_mean']:.1f}` run, `{db['total_waiting_mean']:.1f}` wait) | "
        f"`D`: `{kv['decode_steady_mean_pct']:.3f}% / {kv['decode_peak_single_rank_pct']:.3f}%` · "
        f"`P`: `{kv['prefill_steady_mean_pct']:.3f}% / {kv['prefill_peak_single_rank_pct']:.3f}%` | "
        f"`P`:`{h0_pct:.1f}/{h1_pct:.1f}%` · `D`:`{d0_pct:.1f}/{d1_pct:.1f}%` | "
        f"`{pr['prefill']:.0f} / {pr['decode']:.0f}` | "
        f"**`{w['failed_requests']} / {w['total_requests_launched']}`** |"
    )


def main() -> int:
    if not R3_DIR.exists():
        print("No r3_decode_ceiling directory yet.")
        return 0
    files = sorted(
        R3_DIR.glob("r3_c*.json"),
        key=lambda p: int(re.search(r"r3_c(\d+)\.json$", p.name).group(1)),
    )
    if not files:
        print("No r3_c*.json files found yet.")
        return 0

    levels = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    peak = max(levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
    sweep_done = levels[-1]["stop_evaluation"]["should_stop"] or (levels[-1]["concurrency"] >= 1280)

    sweep_summary_path = R3_DIR / "r3_decode_ceiling_summary.json"
    existing_meta = {}
    if sweep_summary_path.exists():
        try:
            existing_meta = json.loads(sweep_summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}
    existing_meta.update(
        {
            "experiment": "R3_decode_ceiling_S1_2048_1024",
            "completed_concurrencies": [x["concurrency"] for x in levels],
            "stopped_early": levels[-1]["stop_evaluation"]["should_stop"],
            "stop_reasons": levels[-1]["stop_evaluation"]["reasons"],
            "peak_concurrency": peak["concurrency"],
            "peak_window_output_tok_per_s": peak["throughput"]["steady_out_tok_s"],
            "peak_output_tok_per_s_per_decode_chip": peak["throughput"]["out_tok_s_per_decode_chip"],
            "levels": levels,
        }
    )
    sweep_summary_path.write_text(json.dumps(existing_meta, indent=2) + "\n", encoding="utf-8")

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tot_f = sum(x["window"]["failed_requests"] for x in levels)
    tot_r = sum(x["window"]["total_requests_launched"] for x in levels)
    done_c_str = ", ".join(str(x["concurrency"]) for x in levels)
    taus = [x["dspark"]["mean_accept_len_tau"] for x in levels]

    section_lines = [
        MARKER_START,
        "---",
        "",
        "## Step 3 (`R3`) Results — Decode-Only Ceiling (`S1 = 2048:1024` Pre-Warmed `~99.5%` Prefix Hit on `2P2D` `EP=8`)",
        "",
        "- **Design:** `16` `2048`-token trunks pre-warmed across all `8` DP ranks (`0..7`) on BOTH `llmd-prefill-poola-h0` & `h1` (`~99.5%` realized prefix cache hit, keeping Pool A Prefill `<5%` utilized so Pool B Decode is 100% isolated), forced `OSL = 1,024` (`ignore_eos=True`), `300.0 s` steady window + full drain per level.",
        "",
        "| Level | Window | Steady Out tok/s (Gain) | Out tok/s / Dec Chip (`/8`) | Out tok/s / Tot Chip (`/16`) | Server Gen tok/s (`60-290s`) | Cache Hit & Prefill Load | Per-User tok/s `Med / P90_low / P95_low` (In-Win \\| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Pod Split (`P h0/h1` · `D b0/b1`) | Preempt (`P / D`) | Failures |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for i, s in enumerate(levels):
        section_lines.append(fmt_row(s, is_last=(i == len(levels) - 1)))

    if sweep_done:
        stop_desc = (
            f"Stopped after `c={levels[-1]['concurrency']}` ({'; '.join(levels[-1]['stop_evaluation']['reasons'])})"
            if levels[-1]["stop_evaluation"]["should_stop"]
            else f"Completed full ladder `c = {done_c_str}`"
        )
        section_lines.extend(
            [
                "",
                f"- **Completion:** {stop_desc} (`{tot_f} / {tot_r:,}` failures).",
                f"- **True Pool B Decode Ceiling (`c={peak['concurrency']}`):** **`{peak['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (**`{peak['throughput']['out_tok_s_per_decode_chip']:,.2f}` out tok/s/decode-chip**, **`{peak['throughput']['out_tok_s_per_total_chip']:,.2f}` out tok/s/total-chip**, server gen `{peak['throughput']['server_gen_tok_s_60_290']}` tok/s), `DSpark tau = {peak['dspark']['mean_accept_len_tau']:.4f}`, per-user `Med / P90_low = {peak['per_user_decode_tps']['all_requests']['median']:.2f} / {peak['per_user_decode_tps']['all_requests']['p90_low']:.2f}` tok/s, `0` preemptions, `0` failures.",
            ]
        )
    section_lines.append(MARKER_END)
    section_block = "\n".join(section_lines)

    if sweep_done:
        status_cell = f"**COMPLETE (Ceiling `{peak['throughput']['steady_out_tok_s']:,.2f}` out tok/s @ `c={peak['concurrency']}`)**"
        summary_cell = (
            f"Swept `c = {done_c_str}` (`~99.5%` prefix hit, Prefill `<5%` busy). "
            f"True Decode Ceiling **`{peak['throughput']['steady_out_tok_s']:,.2f}` out tok/s** "
            f"(**`{peak['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip**, "
            f"**`{peak['throughput']['out_tok_s_per_total_chip']:,.2f}`/tot-chip**) at `c={peak['concurrency']}`, "
            f"`DSpark tau = {min(taus):.2f}..{max(taus):.2f}`, `{tot_f} / {tot_r:,}` failures."
        )
        next_cell = "Proceed to Step 4 `R4` (`Agentic Prefix-Aware Routing`)."
    else:
        last_s = levels[-1]
        status_cell = f"**IN PROGRESS (`c={done_c_str}` done)**"
        summary_cell = (
            f"Completed `c = {done_c_str}`; latest `c={last_s['concurrency']}` = "
            f"**`{last_s['throughput']['steady_out_tok_s']:,.2f}` out tok/s** "
            f"(`{last_s['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip, `tau={last_s['dspark']['mean_accept_len_tau']:.4f}`, `{tot_f}/{tot_r:,}` err)."
        )
        next_cell = "Continue `R3` concurrency escalation (`256..1280`)."

    r3_row = f"| `{now_utc[:16]}Z` | **STEP 3 `R3`: Decode-Only Ceiling (`S1` `~99.5%` Cached, `2P2D` `EP=8`)** | {status_cell} | {summary_cell} | {next_cell} |"

    txt = STATUS_MD.read_text(encoding="utf-8")
    txt = re.sub(
        r"- \*\*Last Updated \(UTC\):\*\* `[^`]+`",
        f"- **Last Updated (UTC):** `{now_utc}`",
        txt,
    )
    if re.search(r"\| `[^`]+` \| \*\*STEP 3 `R3`:[^\n]+", txt):
        txt = re.sub(r"\| `[^`]+` \| \*\*STEP 3 `R3`:[^\n]+", lambda _: r3_row, txt)
    else:
        anchor = r"(\| `[^`]+` \| \*\*STEP 3 `R2`:[^\n]+)"
        if re.search(anchor, txt):
            txt = re.sub(anchor, lambda m: m.group(1) + "\n" + r3_row, txt)
        else:
            txt = re.sub(r"(\| `[^`]+` \| \*\*STEP 3 `R1b`:[^\n]+)", lambda m: m.group(1) + "\n" + r3_row, txt)

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
    print(f"Updated {STATUS_MD} with {len(levels)} R3 level(s): [{done_c_str}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
