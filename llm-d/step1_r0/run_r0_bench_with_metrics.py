#!/usr/bin/env python3
"""Wrapper that executes the exact Step 1 R0 n=128 300s benchmark on Pool A
while scraping Prefill (:8400/metrics) and Decode (:9400/metrics) before, every 5s during
steady state, and after, then computes full Rule 4 & Prometheus diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def fetch_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_prom_metrics(text: str) -> dict:
    def by_engine(metric_regex: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in re.finditer(rf"^(?:{metric_regex})(\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M):
            labels = m.group(1) or ""
            val = float(m.group(2))
            em = re.search(r'engine="([^"]+)"', labels)
            eng = em.group(1) if em else "all"
            out[eng] = out.get(eng, 0.0) + val
        return out

    kv_by_eng = by_engine(r"vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc)")
    running_by_eng = by_engine(r"vllm:num_requests_running")
    waiting_by_eng = by_engine(r"vllm:num_requests_waiting")
    preempt_by_eng = by_engine(r"vllm:num_preemptions_total")
    prompt_tok_by_eng = by_engine(r"vllm:prompt_tokens_total")
    gen_tok_by_eng = by_engine(r"vllm:generation_tokens_total")
    req_success_by_eng = by_engine(r"vllm:request_success_total")
    prefix_queries_by_eng = by_engine(r"vllm:(?:prefix_cache_queries_total|gpu_prefix_cache_queries_total)")
    prefix_hits_by_eng = by_engine(r"vllm:(?:prefix_cache_hits_total|gpu_prefix_cache_hits_total)")

    drafts = sum(by_engine(r"vllm:spec_decode_num_drafts_total").values())
    draft_tokens = sum(by_engine(r"vllm:spec_decode_num_draft_tokens_total").values())
    accepted = sum(by_engine(r"vllm:spec_decode_num_accepted_tokens_total").values())

    per_pos: dict[str, float] = {}
    for m in re.finditer(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)',
        text,
        re.M,
    ):
        pos = m.group(1)
        per_pos[pos] = per_pos.get(pos, 0.0) + float(m.group(2))

    kv_vals = list(kv_by_eng.values())
    return {
        "kv_usage_by_engine": kv_by_eng,
        "kv_usage_mean": (sum(kv_vals) / len(kv_vals)) if kv_vals else 0.0,
        "kv_usage_max": max(kv_vals) if kv_vals else 0.0,
        "running_by_engine": running_by_eng,
        "running_total": sum(running_by_eng.values()),
        "waiting_by_engine": waiting_by_eng,
        "waiting_total": sum(waiting_by_eng.values()),
        "preemptions_by_engine": preempt_by_eng,
        "preemptions_total": sum(preempt_by_eng.values()),
        "prompt_tokens_by_engine": prompt_tok_by_eng,
        "generation_tokens_by_engine": gen_tok_by_eng,
        "request_success_by_engine": req_success_by_eng,
        "prefix_queries_total": sum(prefix_queries_by_eng.values()),
        "prefix_hits_total": sum(prefix_hits_by_eng.values()),
        "spec_num_drafts": drafts,
        "spec_num_draft_tokens": draft_tokens,
        "spec_num_accepted": accepted,
        "spec_per_pos": per_pos,
    }


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prefill-url", default="http://127.0.0.1:8400")
    ap.add_argument("--decode-url", default="http://192.168.100.38:9400")
    ap.add_argument("--dataset", default="/tmp/requests_mocked.jsonl")
    ap.add_argument("--duration-per-n", type=float, default=300.0)
    ap.add_argument("--warmup-pass-s", type=float, default=10.0)
    ap.add_argument("--tag", default="r0_n128_300s")
    ap.add_argument("--persist-dir", default="/models/dsv41-v7x-llmd-20261106/step1_r0")
    args = ap.parse_args()

    work_dir = Path("/dev/shm/r0_metrics")
    req_dir = Path("/dev/shm/r0_requests")
    work_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(f"/dev/shm/{args.tag}.json")

    print(f"[R0-WRAP] Scraping BEFORE /metrics from P={args.prefill_url} and D={args.decode_url} ...", flush=True)
    p_raw_before = fetch_text(f"{args.prefill_url}/metrics")
    d_raw_before = fetch_text(f"{args.decode_url}/metrics")
    (work_dir / "prefill_metrics_before.txt").write_text(p_raw_before)
    (work_dir / "decode_metrics_before.txt").write_text(d_raw_before)
    p_m0 = parse_prom_metrics(p_raw_before)
    d_m0 = parse_prom_metrics(d_raw_before)

    stop_evt = threading.Event()
    ts_samples: list[dict] = []
    ts_file = work_dir / "metrics_timeseries.jsonl"
    if ts_file.exists():
        ts_file.unlink()
    t0_wall = time.time()
    saved_mid = False

    def _sampler() -> None:
        nonlocal saved_mid
        last_log_s = 0.0
        while not stop_evt.is_set():
            try:
                el = round(time.time() - t0_wall, 2)
                pr = fetch_text(f"{args.prefill_url}/metrics", timeout=6.0)
                dr = fetch_text(f"{args.decode_url}/metrics", timeout=6.0)
                pm = parse_prom_metrics(pr)
                dm = parse_prom_metrics(dr)
                if (not saved_mid) and el >= 200.0 and dm["running_total"] >= 64:
                    (work_dir / "prefill_metrics_mid.txt").write_text(pr)
                    (work_dir / "decode_metrics_mid.txt").write_text(dr)
                    saved_mid = True
                d_dr = max(0.0, dm["spec_num_drafts"] - d_m0["spec_num_drafts"])
                d_ac = max(0.0, dm["spec_num_accepted"] - d_m0["spec_num_accepted"])
                live_tau = round(1.0 + d_ac / d_dr, 4) if d_dr > 0 else None
                rec = {
                    "elapsed_s": el,
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "prefill_kv_mean": round(pm["kv_usage_mean"], 6),
                    "prefill_kv_max": round(pm["kv_usage_max"], 6),
                    "prefill_running_total": pm["running_total"],
                    "prefill_waiting_total": pm["waiting_total"],
                    "prefill_running_by_engine": pm["running_by_engine"],
                    "decode_kv_mean": round(dm["kv_usage_mean"], 6),
                    "decode_kv_max": round(dm["kv_usage_max"], 6),
                    "decode_kv_by_engine": dm["kv_usage_by_engine"],
                    "decode_running_total": dm["running_total"],
                    "decode_waiting_total": dm["waiting_total"],
                    "decode_running_by_engine": dm["running_by_engine"],
                    "decode_preemptions": dm["preemptions_total"] - d_m0["preemptions_total"],
                    "prefill_preemptions": pm["preemptions_total"] - p_m0["preemptions_total"],
                    "live_dspark_tau": live_tau,
                }
                ts_samples.append(rec)
                with open(ts_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if el - last_log_s >= 30.0:
                    print(
                        f"  [METRICS t={el:5.1f}s] P_run={pm['running_total']:.0f} P_kv={pm['kv_usage_mean']*100:.2f}% | "
                        f"D_run={dm['running_total']:.0f} (per_dp={list(dm['running_by_engine'].values())}) "
                        f"D_kv={dm['kv_usage_mean']*100:.2f}% (max={dm['kv_usage_max']*100:.2f}%) | "
                        f"tau={live_tau} | preempt_D={rec['decode_preemptions']:.0f}",
                        flush=True,
                    )
                    last_log_s = el
            except Exception as exc:
                print(f"  [METRICS warn] {exc}", flush=True)
            stop_evt.wait(5.0)

    sampler_thr = threading.Thread(target=_sampler, daemon=True)
    sampler_thr.start()

    cmd = [
        "python3", "-u", "/tmp/bench_empirical_isl_dsv41_rerun.py",
        "--base-url", args.base_url,
        "--prefill-url", args.prefill_url,
        "--decode-url", args.decode_url,
        "--dataset", args.dataset,
        "--sessions", "128",
        "--trunks-above-n16", "40",
        "--dp-size", "8",
        "--num-chips", "8",
        "--max-pinned-prefix-slots", "40",
        "--warmup-concurrency", "8",
        "--warmup-pass-s", str(args.warmup_pass_s),
        "--duration-per-n", str(args.duration_per_n),
        "--cache-hit-rate", "0.95",
        "--no-ignore-eos",
        "--flush-each-n",
        "--save-requests-dir", str(req_dir),
        "--output-json", str(out_json),
    ]
    print(f"[R0-WRAP] Running benchmark command:\n  {' '.join(cmd)}", flush=True)
    bench_log_path = work_dir / f"{args.tag}_stdout.log"
    with open(bench_log_path, "w", encoding="utf-8") as blog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            blog.write(line)
            blog.flush()
        rc = proc.wait()

    stop_evt.set()
    sampler_thr.join(timeout=10.0)

    p_raw_after = fetch_text(f"{args.prefill_url}/metrics")
    d_raw_after = fetch_text(f"{args.decode_url}/metrics")
    (work_dir / "prefill_metrics_after.txt").write_text(p_raw_after)
    (work_dir / "decode_metrics_after.txt").write_text(d_raw_after)
    p_m1 = parse_prom_metrics(p_raw_after)
    d_m1 = parse_prom_metrics(d_raw_after)

    if rc != 0:
        raise RuntimeError(f"bench_empirical_isl_dsv41_rerun.py exited with rc={rc}")

    # Analyze steady-state samples (where decode_running_total >= 64)
    steady_samples = [s for s in ts_samples if s["decode_running_total"] >= 64]
    if not steady_samples:
        steady_samples = ts_samples

    # Compute exact all-requests vs in-window latencies from per-request JSON
    req_json_path = req_dir / "requests_n_128.json"
    bench_summary = json.loads(out_json.read_text())[0]
    req_analysis = {}
    if req_json_path.exists():
        reqs = json.loads(req_json_path.read_text())
        ok_reqs = [r for r in reqs if r.get("error") is None]
        t_b0 = min(r["t_start"] for r in reqs) if reqs else 0.0
        t_win_end = t_b0 + args.duration_per_n
        in_win_reqs = [r for r in ok_reqs if r["t_end"] <= t_win_end]

        all_ttfts = [r["ttft"] for r in ok_reqs if r.get("ttft") is not None]
        win_ttfts = [r["ttft"] for r in in_win_reqs if r.get("ttft") is not None]
        all_tps = [r["decode_tps"] for r in ok_reqs if r.get("decode_tps") is not None]
        win_tps = [r["decode_tps"] for r in in_win_reqs if r.get("decode_tps") is not None]

        dp_req_counts = {}
        for r in ok_reqs:
            k = str(r.get("dp_rank", "unknown"))
            dp_req_counts[k] = dp_req_counts.get(k, 0) + 1

        req_analysis = {
            "total_requests": len(reqs),
            "ok_requests": len(ok_reqs),
            "in_window_completed_requests": len(in_win_reqs),
            "client_dispatched_per_prefill_dp_rank": dict(sorted(dp_req_counts.items())),
            "ttft_all_requests_s": {
                "p50": round(percentile(all_ttfts, 50), 4),
                "p90": round(percentile(all_ttfts, 90), 4),
                "p95": round(percentile(all_ttfts, 95), 4),
                "p99": round(percentile(all_ttfts, 99), 4),
                "mean": round(statistics.mean(all_ttfts), 4) if all_ttfts else 0.0,
            },
            "ttft_in_window_s": {
                "p50": round(percentile(win_ttfts, 50), 4),
                "p90": round(percentile(win_ttfts, 90), 4),
                "p95": round(percentile(win_ttfts, 95), 4),
                "p99": round(percentile(win_ttfts, 99), 4),
                "mean": round(statistics.mean(win_ttfts), 4) if win_ttfts else 0.0,
            },
            "per_user_decode_tps_all_requests": {
                "median": round(percentile(all_tps, 50), 3),
                "p90_low": round(percentile(all_tps, 10), 3),
                "p95_low": round(percentile(all_tps, 5), 3),
                "p90_hi": round(percentile(all_tps, 90), 3),
            },
            "per_user_decode_tps_in_window": {
                "median": round(percentile(win_tps, 50), 3),
                "p90_low": round(percentile(win_tps, 10), 3),
                "p95_low": round(percentile(win_tps, 5), 3),
                "p90_hi": round(percentile(win_tps, 90), 3),
            },
        }

    # Per-engine deltas from Prometheus before -> after
    prefill_reqs_by_eng = {
        k: round(p_m1["request_success_by_engine"].get(k, 0.0) - p_m0["request_success_by_engine"].get(k, 0.0), 1)
        for k in sorted(p_m1["request_success_by_engine"])
    }
    decode_reqs_by_eng = {
        k: round(d_m1["request_success_by_engine"].get(k, 0.0) - d_m0["request_success_by_engine"].get(k, 0.0), 1)
        for k in sorted(d_m1["request_success_by_engine"])
    }
    decode_gen_toks_by_eng = {
        k: round(d_m1["generation_tokens_by_engine"].get(k, 0.0) - d_m0["generation_tokens_by_engine"].get(k, 0.0), 1)
        for k in sorted(d_m1["generation_tokens_by_engine"])
    }
    steady_decode_running_per_eng = {}
    if steady_samples:
        eng_keys = sorted({k for s in steady_samples for k in s["decode_running_by_engine"]})
        for k in eng_keys:
            vals = [s["decode_running_by_engine"].get(k, 0.0) for s in steady_samples]
            steady_decode_running_per_eng[k] = {
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "max": round(max(vals), 1),
            }

    metrics_summary = {
        "window_output_tok_per_s": bench_summary["throughput"]["window_output_tok_per_s"],
        "per_decode_chip_tok_per_s": bench_summary["throughput"]["window_output_tok_per_s"] / 4.0,
        "per_total_chip_tok_per_s": bench_summary["throughput"]["window_output_tok_per_s"] / 8.0,
        "window_total_in_plus_out_tok_per_s": (
            bench_summary["throughput"]["window_input_tok_per_s"]
            + bench_summary["throughput"]["window_output_tok_per_s"]
        ),
        "pass_gate_range": [6700.0, 8200.0],
        "pass_gate_ok": 6700.0 <= bench_summary["throughput"]["window_output_tok_per_s"] <= 8200.0,
        "dspark_spec_decode_steady": bench_summary.get("spec_decode"),
        "dspark_tau_ok": (bench_summary.get("spec_decode", {}).get("mean_accept_len_tau", 0.0) > 2.0),
        "prefill_kv_cache_usage_pct": {
            "steady_mean_pct": round(statistics.mean(s["prefill_kv_mean"] * 100.0 for s in steady_samples), 3) if steady_samples else 0.0,
            "peak_mean_across_ranks_pct": round(max((s["prefill_kv_mean"] * 100.0 for s in ts_samples), default=0.0), 3),
            "peak_single_rank_pct": round(max((s["prefill_kv_max"] * 100.0 for s in ts_samples), default=0.0), 3),
        },
        "decode_kv_cache_usage_pct": {
            "steady_mean_pct": round(statistics.mean(s["decode_kv_mean"] * 100.0 for s in steady_samples), 3) if steady_samples else 0.0,
            "peak_mean_across_ranks_pct": round(max((s["decode_kv_mean"] * 100.0 for s in ts_samples), default=0.0), 3),
            "peak_single_rank_pct": round(max((s["decode_kv_max"] * 100.0 for s in ts_samples), default=0.0), 3),
        },
        "preemptions_delta": {
            "prefill": p_m1["preemptions_total"] - p_m0["preemptions_total"],
            "decode": d_m1["preemptions_total"] - d_m0["preemptions_total"],
        },
        "steady_running_batch_size": {
            "prefill_total_mean": round(statistics.mean(s["prefill_running_total"] for s in steady_samples), 2) if steady_samples else 0.0,
            "prefill_total_max": max((s["prefill_running_total"] for s in ts_samples), default=0.0),
            "decode_total_mean": round(statistics.mean(s["decode_running_total"] for s in steady_samples), 2) if steady_samples else 0.0,
            "decode_total_max": max((s["decode_running_total"] for s in ts_samples), default=0.0),
            "decode_per_dp_rank": steady_decode_running_per_eng,
        },
        "per_dp_rank_completed_requests_including_warmup": {
            "prefill_by_engine": prefill_reqs_by_eng,
            "decode_by_engine": decode_reqs_by_eng,
            "decode_generation_tokens_by_engine": decode_gen_toks_by_eng,
        },
        "request_latency_and_stream_analysis": req_analysis,
    }
    (work_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))

    # Persist everything to /models/dsv41-v7x-llmd-20261106/step1_r0/
    persist = Path(args.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_json, persist / out_json.name)
    if req_json_path.exists():
        shutil.copy2(req_json_path, persist / "r0_requests_n128.json")
    for mf in work_dir.iterdir():
        if mf.is_file():
            shutil.copy2(mf, persist / mf.name)
    for lf in ("/dev/shm/pd_prefill.log", "/dev/shm/pd_proxy.log", "/dev/shm/pd_setup.log"):
        lp = Path(lf)
        if lp.exists():
            shutil.copy2(lp, persist / lp.name)

    print("\n" + "=" * 96, flush=True)
    print(
        f"[R0 SUMMARY] window_out_tok/s={metrics_summary['window_output_tok_per_s']:,.2f} "
        f"(PASS_GATE={metrics_summary['pass_gate_ok']}) | "
        f"DSpark_tau={bench_summary.get('spec_decode', {}).get('mean_accept_len_tau')} "
        f"(TAU_OK={metrics_summary['dspark_tau_ok']}) | "
        f"Preemptions(P/D)={metrics_summary['preemptions_delta']['prefill']}/{metrics_summary['preemptions_delta']['decode']}",
        flush=True,
    )
    print("=" * 96, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
