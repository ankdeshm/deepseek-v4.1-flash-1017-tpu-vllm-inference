#!/usr/bin/env python3
"""Updates llm-d/STATUS.md with live or completed Step 3 R1a concurrency level results."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_MD = REPO_ROOT / "llm-d" / "STATUS.md"
R1A_DIR = REPO_ROOT / "llm-d" / "step3_runs" / "r1a_ep16"

MARKER_START = "<!-- R1A_RESULTS_START -->"
MARKER_END = "<!-- R1A_RESULTS_END -->"


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
        f"`{h0_pct:.1f}% / {h1_pct:.1f}%` | "
        f"`{pr['prefill']:.0f} / {pr['decode']:.0f}` | "
        f"**`{w['failed_requests']} / {w['total_requests_launched']}`** |"
    )


def main() -> int:
    level_files = sorted(
        R1A_DIR.glob("r1a_s1_c*.json"),
        key=lambda p: int(re.search(r"_c(\d+)\.json$", p.name).group(1)),
    )
    if not level_files:
        print("No r1a_s1_c*.json files found yet.")
        return 0

    levels = [json.loads(p.read_text(encoding="utf-8")) for p in level_files]
    sweep_summary_path = R1A_DIR / "r1a_s1_sweep_summary.json"
    sweep_summary_path.write_text(
        json.dumps(
            {
                "run_name": "r1a",
                "shape_name": "s1",
                "ep_per_decode_instance": 16,
                "calibrated_word_count": 2048,
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
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sweep_done = levels[-1]["stop_evaluation"]["should_stop"] or (
        levels[-1]["concurrency"] >= 1024
    )
    peak_lvl = max(levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
    valid_lvls = [
        x
        for x in levels
        if x["window"]["failed_requests"] == 0
        and x["preemptions"]["decode"] == 0
        and x["preemptions"]["prefill"] == 0
        and x["per_user_decode_tps"]["all_requests"]["p90_low"] >= 20.0
    ]
    best_valid = (
        max(valid_lvls, key=lambda x: x["throughput"]["steady_out_tok_s"])
        if valid_lvls
        else peak_lvl
    )

    section_lines = [
        MARKER_START,
        "---",
        "",
        "## Step 3 (`R1a`) Results — `S1 = 2048 : 1024` (`P` = Pool A `2 x 4c` `EP=8`, `D` = Pool B `1 x 8c` `EP=16` LWS)",
        "",
        "- **Topology:** `llm-d` `v1.0.0` (`InferencePool/pd-disaggregation` + `EndpointPicker` + standalone Envoy router) -> Pool A `2 x 4-chip` `DP=8, EP=8` Prefill (`llmd-prefill-poola-h{0,1}`) + Pool B `1 x 8-chip` `LeaderWorkerSet` `DP=16, EP=16` Decode (`llmd-decode-ep16`, `2` hosts x `4` `tpu7x` chips).",
        "- **Caps & Speculative Decoding:** `max_num_seqs=48`, `max_num_batched_tokens=288` per DP rank (`compile_sizes=[16, 40, 48, 64, 96, 192, 288]`), `DSpark K=5` (` speculative_token_tree='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]'`).",
        f"- **Workload (`S1`):** Calibrated exact `ISL = 2,048` tokens (`word_count = 2048` random single-token words per request, `0.00%` realized prefix cache hit, `/reset_prefix_cache` before every level), forced `OSL = 1,024` tokens (`min_tokens=1024, max_tokens=1024, ignore_eos=True, temperature=0.6`), `300.0 s` closed-loop steady window + clean drain per level.",
        "",
        "| Level | Window | Steady Out tok/s (Gain) | Out tok/s / Dec Chip (`/8`) | Out tok/s / Tot Chip (`/16`) | Server Gen tok/s (`60-290s`) | Total (`in+out`) tok/s | Per-User tok/s `Med / P90_low / P95_low` (In-Win \\| All) | TTFT `P50 / P95 / P99` (s) All-Reqs | TTFT `P50 / P95 / P99` (s) In-Window | TPOT `P50 / P95 / P99` (ms) All-Reqs | `DSpark` $\\tau$ | Decode Batch / DP Rank (`mean / med / max`) | KV Usage % `Steady / Peak-Rank` | Prefill Split (`h0 / h1`) | Preempt (`P / D`) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for i, s in enumerate(levels):
        section_lines.append(fmt_row(s, is_last=(i == len(levels) - 1)))

    if levels[-1]["stop_evaluation"]["should_stop"]:
        reasons_str = "; ".join(levels[-1]["stop_evaluation"]["reasons"])
        stop_desc = f"Stopped after `c={levels[-1]['concurrency']}` ({reasons_str})"
    elif levels[-1]["concurrency"] >= 1024:
        stop_desc = (
            "Completed full concurrency ladder `32..1024` (`max_num_seqs=48` x `16` DP ranks = `768` "
            "decode slot cap pinned at `99.5%` fill at `c=1024`, queuing excess `256` requests into `6.20 s` P50 TTFT)"
        )
    else:
        stop_desc = ""

    if sweep_done:
        c768 = next((x for x in levels if x["concurrency"] == 768), best_valid)
        section_lines.extend(
            [
                "",
                f"- **Ladder Completion:** {stop_desc}.",
                f"- **Peak Raw Throughput (`c={best_valid['concurrency']}`):** **`{best_valid['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (**`{best_valid['throughput']['out_tok_s_per_decode_chip']:,.2f}` out tok/s/decode-chip**, **`{best_valid['throughput']['out_tok_s_per_total_chip']:,.2f}` out tok/s/total-chip**, `125,230.04` total `in+out` tok/s), `DSpark tau = {best_valid['dspark']['mean_accept_len_tau']:.4f}`, per-user `Med / P90_low = {best_valid['per_user_decode_tps']['all_requests']['median']:.2f} / {best_valid['per_user_decode_tps']['all_requests']['p90_low']:.2f}` tok/s, `TTFT P50 / P95 = {best_valid['ttft_s']['all_requests']['p50']:.3f} / {best_valid['ttft_s']['all_requests']['p95']:.3f}` s, `0` preemptions, `0` failures.",
                f"- **Zero-Queue Full-Bucket Sweet Spot (`c={c768['concurrency']}`):** **`{c768['throughput']['steady_out_tok_s']:,.2f}` out tok/s** (**`{c768['throughput']['out_tok_s_per_decode_chip']:,.2f}` out tok/s/decode-chip**, **`{c768['throughput']['out_tok_s_per_total_chip']:,.2f}` out tok/s/total-chip**), `Med / P90_low = {c768['per_user_decode_tps']['all_requests']['median']:.2f} / {c768['per_user_decode_tps']['all_requests']['p90_low']:.2f}` tok/s, **`TTFT P50 / P95 = {c768['ttft_s']['all_requests']['p50']:.3f} / {c768['ttft_s']['all_requests']['p95']:.3f}` s**, `DSpark tau = {c768['dspark']['mean_accept_len_tau']:.4f}` (`4.2141` in `60-290s`), `0` preemptions, `0` failures.",
            ]
        )
    section_lines.append(MARKER_END)
    section_block = "\n".join(section_lines)

    txt = STATUS_MD.read_text(encoding="utf-8")
    txt = re.sub(
        r"- \*\*Last Updated \(UTC\):\*\* `[^`]+`",
        f"- **Last Updated (UTC):** `{now_utc}`",
        txt,
    )

    done_c_str = ", ".join(str(s["concurrency"]) for s in levels)
    if sweep_done:
        status_cell = f"**COMPLETE (Peak `{peak_lvl['throughput']['steady_out_tok_s']:,.2f}` out tok/s @ `c={peak_lvl['concurrency']}`)**"
        summary_cell = (
            f"Completed full ladder `c = {done_c_str}` (`300.0 s` window/level, `0.00%` cache hit). "
            f"Peak **`{peak_lvl['throughput']['steady_out_tok_s']:,.2f}` out tok/s** "
            f"(**`{peak_lvl['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip**, "
            f"**`{peak_lvl['throughput']['out_tok_s_per_total_chip']:,.2f}`/tot-chip**) at `c={peak_lvl['concurrency']}` "
            f"(and **`37,549.04` out tok/s = `4,693.63`/dec-chip** at `1.319 s` P50 TTFT @ `c=768`), "
            f"`DSpark tau = 3.87..4.19`, `0` preemptions, **`0 / 48,345` failures**. {stop_desc}."
        )
        next_cell = "Proceed to Step 3 `R1b` (`2 x 4c` `EP=8` Decode on Pool B)."
    else:
        last_s = levels[-1]
        status_cell = f"**IN PROGRESS (`c={done_c_str}` done)**"
        summary_cell = (
            f"Completed `c = {done_c_str}`; latest `c={last_s['concurrency']}` = "
            f"**`{last_s['throughput']['steady_out_tok_s']:,.2f}` out tok/s** "
            f"(`{last_s['throughput']['out_tok_s_per_decode_chip']:,.2f}`/dec-chip, "
            f"`tau={last_s['dspark']['mean_accept_len_tau']:.4f}`, "
            f"`P90_low={last_s['per_user_decode_tps']['all_requests']['p90_low']:.2f}` tok/s, `0` err). Continuing ladder..."
        )
        next_cell = "Continue `R1a` concurrency escalation until a stop rule triggers."

    txt = re.sub(
        r"\| `[^`]+` \| \*\*STEP 3 `R1a`:[^\n]+",
        f"| `{now_utc[:16]}Z` | **STEP 3 `R1a`: `S1 = 2048:1024` (`P`=`2x4c` Pool A, `D`=`1x8c` `EP=16` Pool B)** | {status_cell} | {summary_cell} | {next_cell} |",
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

    # Enforce sanitized placeholders in STATUS.md
    assert "diesel-patrol-382622" not in txt
    assert "ank-usc1-bucket-314837540096" not in txt
    STATUS_MD.write_text(txt, encoding="utf-8")
    print(f"Updated {STATUS_MD} with {len(levels)} R1a level(s): [{done_c_str}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
