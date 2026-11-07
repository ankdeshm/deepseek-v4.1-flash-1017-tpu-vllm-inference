#!/usr/bin/env python3
"""Reusable closed-loop llm-d shape sweep benchmark harness (Step 3 R1a / R1b / R2).

Pure Python stdlib + aiohttp ONLY. Never imports torch, transformers, or vllm so it
never touches TPU VFIO devices when executed from a host-networked pod or CPU node.

Implements:
- Auto-calibrated synthetic token prompts from 32 verified single-token words for
  DeepSeek-V4.1-Flash, with deterministic per-request RNG seeds (`0.00%` prefix cache hit
  by default, or configurable shared trunks if `--cache-hit-ratio > 0`).
- Explicit `POST /reset_prefix_cache` across all Prefill and Decode servers before warmup,
  after warmup, and before every concurrency level.
- Exact `steady_seconds` (`300.0 s`) closed-loop window accounting with proportional
  in-window token credit for requests straddling `t_window_end` and clean in-flight drain
  (identical to `scripts/bench_empirical_isl_dsv41_rerun.py`).
- Per-request metrics (`decode_tps`, `TTFT`, `TPOT`, `E2E`) reported over BOTH
  `all_requests` (including drain) and `in_window` (`t_end <= t_window_end`).
- Background `5.0 s` Prometheus `/metrics` scraper across all Prefill pods, Decode
  servers, and `llm-d` EPP (:9090/metrics), capturing:
  * `DSpark tau = 1.0 + delta_accepted / delta_drafts` (full run + `[60s, 290s]` steady)
  * Server-side `vllm:generation_tokens_total` steady rate (`t in [60s, 290s]`)
  * Per-DP-rank decode batch (`mean / median / max` across `t in [30s, 295s]`)
  * `KV usage %` (`steady mean` in `[30s, 295s]` and `peak single-rank`) for `D` and `P`
  * `Preemptions` (`P` and `D`)
  * Per-Prefill-pod request & prompt-token split (`h0` vs `h1`)
- Automatic evaluation of all 5 Step 3 Stop Rules after every concurrency level, plus
  immediate persistence of summary JSON, Prometheus timeseries JSONL, and gzipped
  per-request JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import time
from typing import Any

# Enforce strict isolation from torch_tpu / vllm PYTHONPATH before importing aiohttp.
os.environ["PYTHONPATH"] = ""
assert "torch" not in sys.modules and "transformers" not in sys.modules, (
    "FATAL: torch/transformers must never be imported in run_llmd_shape_sweep.py"
)

import aiohttp  # noqa: E402

# Verified 32 single-token words for DeepSeek-V4.1-Flash tokenizer.
WORDS: list[str] = [
    " alpha", " beta", " gamma", " delta",
    " tensor", " matrix", " router", " expert",
    " kernel", " vector", " scalar", " stream",
    " buffer", " socket", " packet", " switch",
    " fabric", " memory", " device", " engine",
    " worker", " leader", " server", " client",
    " thread", " queue", " batch", " layer",
    " block", " scale", " group", " index",
]


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


def summarize_dist_tps(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90_low": 0.0,
            "p95_low": 0.0,
            "p90_hi": 0.0,
        }
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(percentile(vals, 50), 4),
        "p90_low": round(percentile(vals, 10), 4),
        "p95_low": round(percentile(vals, 5), 4),
        "p90_hi": round(percentile(vals, 90), 4),
    }


def summarize_dist_lat(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "p50": round(percentile(vals, 50), 4),
        "p90": round(percentile(vals, 90), 4),
        "p95": round(percentile(vals, 95), 4),
        "p99": round(percentile(vals, 99), 4),
    }


def parse_vllm_prom(text: str, prefix: str) -> dict[str, Any]:
    """Parse Prometheus metrics from a single vLLM /metrics endpoint, namespacing engines."""

    def by_engine(metric_regex: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in re.finditer(
            rf"^(?:{metric_regex})(\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M
        ):
            labels = m.group(1) or ""
            val = float(m.group(2))
            em = re.search(r'engine="([^"]+)"', labels)
            eng = f"{prefix}:{em.group(1)}" if em else f"{prefix}:all"
            out[eng] = out.get(eng, 0.0) + val
        return out

    kv_by_eng = by_engine(r"vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc)")
    running_by_eng = by_engine(r"vllm:num_requests_running")
    waiting_by_eng = by_engine(r"vllm:num_requests_waiting")
    preempt_by_eng = by_engine(r"vllm:num_preemptions_total")
    prompt_tok_by_eng = by_engine(r"vllm:prompt_tokens_total")
    gen_tok_by_eng = by_engine(r"vllm:generation_tokens_total")
    req_success_by_eng = by_engine(r"vllm:request_success_total")
    prefix_queries_by_eng = by_engine(
        r"vllm:(?:prefix_cache_queries_total|gpu_prefix_cache_queries_total)"
    )
    prefix_hits_by_eng = by_engine(
        r"vllm:(?:prefix_cache_hits_total|gpu_prefix_cache_hits_total)"
    )

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

    return {
        "kv_by_eng": kv_by_eng,
        "running_by_eng": running_by_eng,
        "waiting_by_eng": waiting_by_eng,
        "preempt_by_eng": preempt_by_eng,
        "prompt_tok_by_eng": prompt_tok_by_eng,
        "gen_tok_by_eng": gen_tok_by_eng,
        "req_success_by_eng": req_success_by_eng,
        "prefix_queries_total": sum(prefix_queries_by_eng.values()),
        "prefix_hits_total": sum(prefix_hits_by_eng.values()),
        "spec_num_drafts": drafts,
        "spec_num_draft_tokens": draft_tokens,
        "spec_num_accepted": accepted,
        "spec_per_pos": per_pos,
    }


def merge_vllm_prom(parsed_list: list[dict[str, Any]]) -> dict[str, Any]:
    kv_by_eng: dict[str, float] = {}
    running_by_eng: dict[str, float] = {}
    waiting_by_eng: dict[str, float] = {}
    preempt_by_eng: dict[str, float] = {}
    prompt_tok_by_eng: dict[str, float] = {}
    gen_tok_by_eng: dict[str, float] = {}
    req_success_by_eng: dict[str, float] = {}
    per_pos: dict[str, float] = {}
    prefix_queries_total = 0.0
    prefix_hits_total = 0.0
    spec_num_drafts = 0.0
    spec_num_draft_tokens = 0.0
    spec_num_accepted = 0.0

    for p in parsed_list:
        kv_by_eng.update(p["kv_by_eng"])
        running_by_eng.update(p["running_by_eng"])
        waiting_by_eng.update(p["waiting_by_eng"])
        preempt_by_eng.update(p["preempt_by_eng"])
        prompt_tok_by_eng.update(p["prompt_tok_by_eng"])
        gen_tok_by_eng.update(p["gen_tok_by_eng"])
        req_success_by_eng.update(p["req_success_by_eng"])
        prefix_queries_total += p["prefix_queries_total"]
        prefix_hits_total += p["prefix_hits_total"]
        spec_num_drafts += p["spec_num_drafts"]
        spec_num_draft_tokens += p["spec_num_draft_tokens"]
        spec_num_accepted += p["spec_num_accepted"]
        for k, v in p["spec_per_pos"].items():
            per_pos[k] = per_pos.get(k, 0.0) + v

    kv_vals = list(kv_by_eng.values())
    return {
        "kv_by_eng": kv_by_eng,
        "kv_mean": (sum(kv_vals) / len(kv_vals)) if kv_vals else 0.0,
        "kv_max": max(kv_vals) if kv_vals else 0.0,
        "running_by_eng": running_by_eng,
        "running_total": sum(running_by_eng.values()),
        "waiting_by_eng": waiting_by_eng,
        "waiting_total": sum(waiting_by_eng.values()),
        "preempt_by_eng": preempt_by_eng,
        "preemptions_total": sum(preempt_by_eng.values()),
        "prompt_tok_by_eng": prompt_tok_by_eng,
        "prompt_tokens_total": sum(prompt_tok_by_eng.values()),
        "gen_tok_by_eng": gen_tok_by_eng,
        "generation_tokens_total": sum(gen_tok_by_eng.values()),
        "req_success_by_eng": req_success_by_eng,
        "request_success_total": sum(req_success_by_eng.values()),
        "prefix_queries_total": prefix_queries_total,
        "prefix_hits_total": prefix_hits_total,
        "spec_num_drafts": spec_num_drafts,
        "spec_num_draft_tokens": spec_num_draft_tokens,
        "spec_num_accepted": spec_num_accepted,
        "spec_per_pos": per_pos,
    }


def parse_epp_prom(text: str) -> dict[str, Any]:
    inflight_reqs: dict[str, float] = {}
    queue_size: dict[str, float] = {}
    ready_endpoints = 0.0
    disagg_decisions = 0.0

    for m in re.finditer(
        r'^llm_d_epp_inflight_requests\{[^}]*endpoint_name="([^"]+)"[^}]*\}\s+([0-9.eE+-]+)',
        text,
        re.M,
    ):
        ep = m.group(1)
        inflight_reqs[ep] = inflight_reqs.get(ep, 0.0) + float(m.group(2))

    for m in re.finditer(
        r'^llm_d_epp_per_endpoint_queue_size\{[^}]*model_server_endpoint="([^"]+)"[^}]*\}\s+([0-9.eE+-]+)',
        text,
        re.M,
    ):
        ep = m.group(1)
        queue_size[ep] = queue_size.get(ep, 0.0) + float(m.group(2))

    for m in re.finditer(
        r"^llm_d_epp_ready_endpoints\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M
    ):
        ready_endpoints += float(m.group(1))

    for m in re.finditer(
        r"^llm_d_epp_disagg_decision_total\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M
    ):
        disagg_decisions += float(m.group(1))

    return {
        "ready_endpoints": ready_endpoints,
        "disagg_decisions_total": disagg_decisions,
        "inflight_requests_by_endpoint": inflight_reqs,
        "queue_size_by_endpoint": queue_size,
    }


async def scrape_cluster_snapshot(
    session: aiohttp.ClientSession,
    prefill_urls: list[str],
    decode_urls: list[str],
    epp_metrics_url: str,
) -> dict[str, Any]:
    async def _fetch(url: str) -> tuple[bool, str]:
        if not url:
            return False, ""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=8.0)
            ) as resp:
                if resp.status == 200:
                    return True, await resp.text()
                return False, f"HTTP {resp.status}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    p_tasks = [_fetch(f"{u.rstrip('/')}/metrics") for u in prefill_urls]
    d_tasks = [_fetch(f"{u.rstrip('/')}/metrics") for u in decode_urls]
    e_task = [_fetch(epp_metrics_url)] if epp_metrics_url else []

    results = await asyncio.gather(*p_tasks, *d_tasks, *e_task)
    n_p = len(prefill_urls)
    n_d = len(decode_urls)
    p_res = results[:n_p]
    d_res = results[n_p : n_p + n_d]
    e_res = results[n_p + n_d] if e_task else (False, "")

    p_parsed_per_pod: dict[str, dict[str, Any]] = {}
    p_parsed_list: list[dict[str, Any]] = []
    p_all_ok = True
    for i, (ok, txt) in enumerate(p_res):
        pod_key = f"h{i}"
        if ok:
            pr = parse_vllm_prom(txt, pod_key)
            p_parsed_per_pod[pod_key] = {
                "request_success_total": sum(pr["req_success_by_eng"].values()),
                "prompt_tokens_total": sum(pr["prompt_tok_by_eng"].values()),
                "running_total": sum(pr["running_by_eng"].values()),
                "waiting_total": sum(pr["waiting_by_eng"].values()),
                "kv_mean": (
                    sum(pr["kv_by_eng"].values()) / len(pr["kv_by_eng"])
                    if pr["kv_by_eng"]
                    else 0.0
                ),
                "kv_max": max(pr["kv_by_eng"].values()) if pr["kv_by_eng"] else 0.0,
                "preemptions_total": sum(pr["preempt_by_eng"].values()),
            }
            p_parsed_list.append(pr)
        else:
            p_all_ok = False
            p_parsed_per_pod[pod_key] = {"error": txt}

    d_parsed_per_inst: dict[str, dict[str, Any]] = {}
    d_parsed_list: list[dict[str, Any]] = []
    d_all_ok = True
    for i, (ok, txt) in enumerate(d_res):
        inst_key = f"d{i}"
        if ok:
            dr = parse_vllm_prom(txt, inst_key)
            d_parsed_per_inst[inst_key] = {
                "request_success_total": sum(dr["req_success_by_eng"].values()),
                "generation_tokens_total": sum(dr["gen_tok_by_eng"].values()),
                "running_total": sum(dr["running_by_eng"].values()),
                "waiting_total": sum(dr["waiting_by_eng"].values()),
                "kv_mean": (
                    sum(dr["kv_by_eng"].values()) / len(dr["kv_by_eng"])
                    if dr["kv_by_eng"]
                    else 0.0
                ),
                "kv_max": max(dr["kv_by_eng"].values()) if dr["kv_by_eng"] else 0.0,
                "preemptions_total": sum(dr["preempt_by_eng"].values()),
                "spec_num_drafts": dr["spec_num_drafts"],
                "spec_num_accepted": dr["spec_num_accepted"],
            }
            d_parsed_list.append(dr)
        else:
            d_all_ok = False
            d_parsed_per_inst[inst_key] = {"error": txt}

    epp_parsed = parse_epp_prom(e_res[1]) if e_res[0] else {"error": e_res[1]}

    return {
        "prefill_all_ok": p_all_ok,
        "decode_all_ok": d_all_ok,
        "epp_ok": e_res[0],
        "prefill": merge_vllm_prom(p_parsed_list),
        "prefill_per_pod": p_parsed_per_pod,
        "decode": merge_vllm_prom(d_parsed_list),
        "decode_per_instance": d_parsed_per_inst,
        "epp": epp_parsed,
    }


async def check_all_healthy(
    session: aiohttp.ClientSession, prefill_urls: list[str], decode_urls: list[str]
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for label, urls in [("prefill", prefill_urls), ("decode", decode_urls)]:
        for idx, u in enumerate(urls):
            h_url = f"{u.rstrip('/')}/health"
            try:
                async with session.get(
                    h_url, timeout=aiohttp.ClientTimeout(total=8.0)
                ) as resp:
                    if resp.status != 200:
                        issues.append(f"{label}[{idx}] {h_url} -> HTTP {resp.status}")
            except Exception as exc:
                issues.append(f"{label}[{idx}] {h_url} -> {type(exc).__name__}: {exc}")
    return (len(issues) == 0), issues


async def reset_all_prefix_caches(
    session: aiohttp.ClientSession, prefill_urls: list[str], decode_urls: list[str]
) -> None:
    await asyncio.sleep(1.5)
    for label, urls in [("prefill", prefill_urls), ("decode", decode_urls)]:
        for idx, u in enumerate(urls):
            url = f"{u.rstrip('/')}/reset_prefix_cache"
            ok = False
            last_code = 0
            for _ in range(12):
                try:
                    async with session.post(
                        url, timeout=aiohttp.ClientTimeout(total=15.0)
                    ) as resp:
                        last_code = resp.status
                        if resp.status == 200:
                            ok = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            print(
                f"  [reset_prefix_cache] {label}[{idx}] ({u}): "
                f"{'OK (HTTP 200)' if ok else f'WARN (HTTP {last_code})'}",
                flush=True,
            )
            if not ok:
                raise RuntimeError(
                    f"Failed to reset prefix cache on {label}[{idx}] ({u}): last HTTP {last_code}"
                )
    await asyncio.sleep(1.0)


def build_prompt(
    word_count: int,
    seed: int,
    shared_trunks: list[str] | None = None,
    prefix_words: int = 0,
) -> str:
    rng = random.Random(seed)
    if shared_trunks and prefix_words > 0:
        trunk = shared_trunks[seed % len(shared_trunks)]
        rem = max(0, word_count - prefix_words)
        if rem == 0:
            return trunk
        return trunk + "".join(rng.choices(WORDS, k=rem))
    return "".join(rng.choices(WORDS, k=word_count))


async def send_completion_request(
    session: aiohttp.ClientSession,
    completions_url: str,
    payload: dict[str, Any],
    req_idx: int,
    worker_id: int,
    seed: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    t_first: float | None = None
    chunk_times: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    status_code = 0
    error_msg: str | None = None
    finish_reason: str | None = None
    num_text_chunks = 0

    try:
        async with session.post(completions_url, json=payload) as resp:
            status_code = resp.status
            if status_code != 200:
                body = await resp.text()
                error_msg = f"HTTP {status_code}: {body[:300]}"
            else:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    now = time.perf_counter()
                    choices = chunk.get("choices")
                    if choices and len(choices) > 0:
                        c0 = choices[0]
                        if c0.get("finish_reason"):
                            finish_reason = c0.get("finish_reason")
                        text_val = c0.get("text")
                        if text_val:
                            if t_first is None:
                                t_first = now
                            chunk_times.append(now)
                            num_text_chunks += 1

                    usage = chunk.get("usage")
                    if isinstance(usage, dict) and usage:
                        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                        completion_tokens = int(
                            usage.get("completion_tokens") or completion_tokens
                        )
                        details = usage.get("prompt_tokens_details")
                        if (
                            isinstance(details, dict)
                            and details.get("cached_tokens") is not None
                        ):
                            cached_tokens = int(details["cached_tokens"])
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    t_end = time.perf_counter()
    e2e = t_end - t_start
    if t_first is None and error_msg is None:
        t_first = t_end
    ttft = (t_first - t_start) if t_first is not None else None

    eff_compl = completion_tokens or num_text_chunks
    decode_dur = max(t_end - (t_first if t_first is not None else t_end), 1e-6)
    decode_tps: float | None = None
    tpot_ms: float | None = None
    if error_msg is None and eff_compl >= 1:
        if eff_compl > 1:
            decode_tps = (eff_compl - 1) / decode_dur
        else:
            decode_tps = 1.0 / decode_dur
        tpot_ms = 1000.0 * (t_end - (t_first or t_end)) / max(eff_compl - 1, 1)

    return {
        "req_idx": req_idx,
        "worker_id": worker_id,
        "seed": seed,
        "t_start": t_start,
        "t_first": t_first,
        "t_end": t_end,
        "ttft": ttft,
        "e2e": e2e,
        "decode_dur": decode_dur,
        "decode_tps": decode_tps,
        "tpot_ms": tpot_ms,
        "chunk_times": chunk_times,
        "num_chunks": num_text_chunks,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": eff_compl,
        "finish_reason": finish_reason,
        "status_code": status_code,
        "error": error_msg,
    }


async def calibrate_word_count(
    session: aiohttp.ClientSession,
    completions_url: str,
    model: str,
    target_isl: int,
    run_seed: int,
) -> int:
    word_count = target_isl
    for attempt in range(5):
        seed = run_seed * 1_000_000 + 77_000 + attempt
        prompt_str = build_prompt(word_count, seed)
        payload = {
            "model": model,
            "prompt": prompt_str,
            "max_tokens": 2,
            "min_tokens": 2,
            "ignore_eos": True,
            "temperature": 0.6,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        res = await send_completion_request(
            session, completions_url, payload, req_idx=-1 - attempt, worker_id=-1, seed=seed
        )
        if res["error"] is not None:
            raise RuntimeError(f"Calibration request failed: {res['error']}")
        got = res["prompt_tokens"]
        print(
            f"  [calibrate attempt {attempt + 1}] word_count={word_count} -> "
            f"prompt_tokens={got} (target={target_isl}, cached={res['cached_tokens']})",
            flush=True,
        )
        if got == target_isl:
            return word_count
        word_count = max(1, word_count + (target_isl - got))
    raise RuntimeError(
        f"Could not calibrate exact prompt_tokens={target_isl} after 5 attempts (last word_count={word_count})"
    )


async def run_warmup(
    session: aiohttp.ClientSession,
    completions_url: str,
    model: str,
    word_count: int,
    warmup_concurrency: int,
    warmup_seconds: float,
    warmup_max_tokens: int,
    temperature: float,
    run_seed: int,
) -> None:
    print(
        f"\n[WARMUP] Running {warmup_seconds:.1f}s warmup at c={warmup_concurrency} "
        f"(ISL={word_count}, max_tokens={warmup_max_tokens}) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    t_end = t0 + warmup_seconds
    counter = 0

    async def _w_worker(w_id: int) -> tuple[int, int]:
        nonlocal counter
        ok_cnt = 0
        err_cnt = 0
        while time.perf_counter() < t_end:
            idx = counter
            counter += 1
            seed = run_seed * 1_000_000 + 88_000 + idx
            prompt_str = build_prompt(word_count, seed)
            payload = {
                "model": model,
                "prompt": prompt_str,
                "max_tokens": warmup_max_tokens,
                "min_tokens": warmup_max_tokens,
                "ignore_eos": True,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            res = await send_completion_request(
                session, completions_url, payload, req_idx=idx, worker_id=w_id, seed=seed
            )
            if res["error"] is None:
                ok_cnt += 1
            else:
                err_cnt += 1
        return ok_cnt, err_cnt

    w_res = await asyncio.gather(*(_w_worker(i) for i in range(warmup_concurrency)))
    tot_ok = sum(x[0] for x in w_res)
    tot_err = sum(x[1] for x in w_res)
    print(
        f"[WARMUP] Completed {tot_ok} OK / {tot_err} ERR warmup requests in "
        f"{time.perf_counter() - t0:.2f}s (including drain).",
        flush=True,
    )
    if tot_err > 0:
        raise RuntimeError(f"Warmup encountered {tot_err} failed requests!")


def copy_if_mirror(src: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, mirror_dir / src.name)


async def run_concurrency_level(
    session: aiohttp.ClientSession,
    *,
    run_name: str,
    shape_name: str,
    ep_size: int,
    concurrency: int,
    word_count: int,
    target_isl: int,
    target_osl: int,
    ignore_eos: bool,
    temperature: float,
    cache_hit_ratio: float,
    num_trunks: int,
    steady_seconds: float,
    decode_chips: int,
    total_chips: int,
    completions_url: str,
    model: str,
    prefill_urls: list[str],
    decode_urls: list[str],
    epp_metrics_url: str,
    run_seed: int,
    out_dir: Path,
    raw_out_dir: Path,
    gcs_mirror_dir: Path | None,
    prev_steady_out_tok_s: float | None,
) -> dict[str, Any]:
    level_tag = f"{run_name}_{shape_name}_c{concurrency}"
    print("\n" + "=" * 108, flush=True)
    print(
        f"[START {level_tag}] concurrency={concurrency} | ISL={target_isl} (words={word_count}) | "
        f"OSL={target_osl} (ignore_eos={ignore_eos}) | window={steady_seconds:.1f}s | "
        f"cache_hit_target={cache_hit_ratio*100:.1f}%",
        flush=True,
    )
    print("=" * 108, flush=True)

    await reset_all_prefix_caches(session, prefill_urls, decode_urls)
    healthy_before, issues_before = await check_all_healthy(
        session, prefill_urls, decode_urls
    )
    if not healthy_before:
        raise RuntimeError(f"Pre-run health check failed: {issues_before}")

    shared_trunks: list[str] | None = None
    prefix_words = int(round(word_count * cache_hit_ratio))
    if cache_hit_ratio > 0.0 and prefix_words > 0:
        shared_trunks = [
            "".join(
                random.Random(run_seed * 10_000 + concurrency * 100 + t).choices(
                    WORDS, k=prefix_words
                )
            )
            for t in range(max(1, num_trunks))
        ]

    snap_m0 = await scrape_cluster_snapshot(
        session, prefill_urls, decode_urls, epp_metrics_url
    )
    p_m0 = snap_m0["prefill"]
    d_m0 = snap_m0["decode"]

    ts_path = out_dir / f"{level_tag}_timeseries.jsonl"
    if ts_path.exists():
        ts_path.unlink()
    ts_samples: list[dict[str, Any]] = []
    stop_scraper = asyncio.Event()
    tau_abort_reason: list[str] = []

    t_bench_start = time.perf_counter()
    t_window_end = t_bench_start + steady_seconds

    async def _prom_scraper() -> None:
        last_print_s = -30.0
        while not stop_scraper.is_set():
            try:
                snap = await scrape_cluster_snapshot(
                    session, prefill_urls, decode_urls, epp_metrics_url
                )
                el = round(time.perf_counter() - t_bench_start, 3)
                pm = snap["prefill"]
                dm = snap["decode"]
                d_drafts = max(0.0, dm["spec_num_drafts"] - d_m0["spec_num_drafts"])
                d_acc = max(0.0, dm["spec_num_accepted"] - d_m0["spec_num_accepted"])
                live_tau = round(1.0 + d_acc / d_drafts, 4) if d_drafts > 0 else None

                if (
                    el >= 30.0
                    and d_drafts >= 100.0
                    and live_tau is not None
                    and live_tau < 1.5
                    and not tau_abort_reason
                ):
                    tau_abort_reason.append(
                        f"Live DSpark tau dropped below 1.5 ({live_tau:.4f} at t={el:.1f}s)"
                    )
                    print(f"  [EMERGENCY STOP] {tau_abort_reason[-1]}", flush=True)

                rec = {
                    "elapsed_s": el,
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "in_steady_window": bool(el <= steady_seconds),
                    "prefill_all_ok": snap["prefill_all_ok"],
                    "decode_all_ok": snap["decode_all_ok"],
                    "prefill_running_total": pm["running_total"],
                    "prefill_waiting_total": pm["waiting_total"],
                    "prefill_kv_mean_pct": round(pm["kv_mean"] * 100.0, 4),
                    "prefill_kv_max_pct": round(pm["kv_max"] * 100.0, 4),
                    "prefill_preemptions_delta": pm["preemptions_total"]
                    - p_m0["preemptions_total"],
                    "prefill_per_pod": snap["prefill_per_pod"],
                    "decode_running_total": dm["running_total"],
                    "decode_waiting_total": dm["waiting_total"],
                    "decode_running_by_eng": dm["running_by_eng"],
                    "decode_kv_mean_pct": round(dm["kv_mean"] * 100.0, 4),
                    "decode_kv_max_pct": round(dm["kv_max"] * 100.0, 4),
                    "decode_kv_by_eng_pct": {
                        k: round(v * 100.0, 4) for k, v in dm["kv_by_eng"].items()
                    },
                    "decode_gen_tokens_delta": dm["generation_tokens_total"]
                    - d_m0["generation_tokens_total"],
                    "decode_preemptions_delta": dm["preemptions_total"]
                    - d_m0["preemptions_total"],
                    "dspark_drafts_delta": d_drafts,
                    "dspark_accepted_delta": d_acc,
                    "live_dspark_tau": live_tau,
                    "epp": snap["epp"],
                }
                ts_samples.append(rec)
                with open(ts_path, "a", encoding="utf-8") as fts:
                    fts.write(json.dumps(rec) + "\n")

                if el - last_print_s >= 30.0:
                    d_eng_vals = [
                        int(v) for v in dm["running_by_eng"].values()
                    ]
                    print(
                        f"  [t={el:5.1f}s c={concurrency}] "
                        f"P_run={pm['running_total']:.0f} (wait={pm['waiting_total']:.0f}) "
                        f"P_kv={pm['kv_mean']*100:.2f}% | "
                        f"D_run={dm['running_total']:.0f} (wait={dm['waiting_total']:.0f}, "
                        f"per_dp_min/med/max="
                        f"{min(d_eng_vals) if d_eng_vals else 0}/"
                        f"{statistics.median(d_eng_vals) if d_eng_vals else 0:.1f}/"
                        f"{max(d_eng_vals) if d_eng_vals else 0}) "
                        f"D_kv={dm['kv_mean']*100:.2f}% (pk={dm['kv_max']*100:.2f}%) | "
                        f"tau={live_tau} | preempt(P/D)="
                        f"{rec['prefill_preemptions_delta']:.0f}/{rec['decode_preemptions_delta']:.0f}",
                        flush=True,
                    )
                    last_print_s = el
            except Exception as exc:
                print(f"  [METRICS WARN] {exc}", flush=True)

            try:
                await asyncio.wait_for(stop_scraper.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    scraper_task = asyncio.create_task(_prom_scraper())

    req_counter = 0

    async def _worker(worker_id: int) -> list[dict[str, Any]]:
        nonlocal req_counter
        worker_out: list[dict[str, Any]] = []
        while (time.perf_counter() < t_window_end) and (not tau_abort_reason):
            r_idx = req_counter
            req_counter += 1
            seed = run_seed * 1_000_000 + concurrency * 100_000 + r_idx
            prompt_str = build_prompt(
                word_count, seed, shared_trunks=shared_trunks, prefix_words=prefix_words
            )
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt_str,
                "max_tokens": target_osl,
                "ignore_eos": bool(ignore_eos),
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if ignore_eos:
                payload["min_tokens"] = target_osl
            res = await send_completion_request(
                session,
                completions_url,
                payload,
                req_idx=r_idx,
                worker_id=worker_id,
                seed=seed,
            )
            worker_out.append(res)
        return worker_out

    worker_lists = await asyncio.gather(*(_worker(w) for w in range(concurrency)))
    t_bench_end = time.perf_counter()
    drain_wall_s = t_bench_end - t_bench_start

    stop_scraper.set()
    await scraper_task

    snap_m1 = await scrape_cluster_snapshot(
        session, prefill_urls, decode_urls, epp_metrics_url
    )
    healthy_after, issues_after = await check_all_healthy(
        session, prefill_urls, decode_urls
    )
    p_m1 = snap_m1["prefill"]
    d_m1 = snap_m1["decode"]

    results = [r for lst in worker_lists for r in lst]
    results.sort(key=lambda x: x["req_idx"])
    ok_results = [r for r in results if r["error"] is None]
    err_results = [r for r in results if r["error"] is not None]

    # Exact 300.0s window accounting (matching scripts/bench_empirical_isl_dsv41_rerun.py)
    window_output_tokens = 0.0
    window_input_tokens = 0.0
    window_cached_tokens = 0.0
    in_win_results: list[dict[str, Any]] = []

    for r in ok_results:
        c_times = r["chunk_times"]
        compl = float(r["completion_tokens"] or 0)
        prompt = float(r["prompt_tokens"] or 0)
        cached = float(r["cached_tokens"] or 0)
        if r["t_end"] <= t_window_end:
            window_output_tokens += compl
            window_input_tokens += prompt
            window_cached_tokens += cached
            r["in_window_completed"] = True
            r["window_output_tokens_credited"] = compl
            in_win_results.append(r)
        elif c_times:
            in_win_chunks = sum(1 for ct in c_times if ct <= t_window_end)
            r["in_window_completed"] = False
            if in_win_chunks > 0:
                frac = in_win_chunks / len(c_times)
                credited = compl * frac
                window_output_tokens += credited
                window_input_tokens += prompt
                window_cached_tokens += cached
                r["window_output_tokens_credited"] = round(credited, 4)
            else:
                r["window_output_tokens_credited"] = 0.0
        else:
            r["in_window_completed"] = False
            r["window_output_tokens_credited"] = 0.0

    steady_out_tok_s = window_output_tokens / steady_seconds
    steady_in_tok_s = window_input_tokens / steady_seconds
    steady_total_tok_s = steady_in_tok_s + steady_out_tok_s
    out_tok_s_per_decode_chip = steady_out_tok_s / max(1, decode_chips)
    out_tok_s_per_total_chip = steady_out_tok_s / max(1, total_chips)

    total_prompt_all = sum(r["prompt_tokens"] for r in ok_results)
    total_cached_all = sum(r["cached_tokens"] for r in ok_results)
    total_compl_all = sum(r["completion_tokens"] for r in ok_results)
    realized_cache_hit_pct = (
        (100.0 * total_cached_all / total_prompt_all) if total_prompt_all > 0 else 0.0
    )

    # DSpark speculative decoding metrics (full run + [60s, 290s] steady slice)
    d_drafts_total = max(0.0, d_m1["spec_num_drafts"] - d_m0["spec_num_drafts"])
    d_draft_toks_total = max(
        0.0, d_m1["spec_num_draft_tokens"] - d_m0["spec_num_draft_tokens"]
    )
    d_accepted_total = max(0.0, d_m1["spec_num_accepted"] - d_m0["spec_num_accepted"])
    dspark_tau = (
        (1.0 + d_accepted_total / d_drafts_total) if d_drafts_total > 0 else 1.0
    )
    d_pos_rates: dict[str, float] = {}
    if d_drafts_total > 0:
        for k in sorted(d_m1["spec_per_pos"].keys(), key=lambda x: int(x)):
            delta_k = max(
                0.0,
                d_m1["spec_per_pos"].get(k, 0.0) - d_m0["spec_per_pos"].get(k, 0.0),
            )
            d_pos_rates[k] = round(delta_k / d_drafts_total, 4)

    # Server-side generation_tokens_total rate & steady tau in [60s, 290s]
    samples_60_290 = [s for s in ts_samples if 60.0 <= s["elapsed_s"] <= 290.0]
    server_gen_tok_s_60_290: float | None = None
    dspark_tau_60_290: float | None = None
    if len(samples_60_290) >= 2:
        s_a, s_b = samples_60_290[0], samples_60_290[-1]
        dt_60_290 = s_b["elapsed_s"] - s_a["elapsed_s"]
        if dt_60_290 > 0:
            server_gen_tok_s_60_290 = round(
                (s_b["decode_gen_tokens_delta"] - s_a["decode_gen_tokens_delta"])
                / dt_60_290,
                2,
            )
        dr_60_290 = s_b["dspark_drafts_delta"] - s_a["dspark_drafts_delta"]
        ac_60_290 = s_b["dspark_accepted_delta"] - s_a["dspark_accepted_delta"]
        if dr_60_290 > 0:
            dspark_tau_60_290 = round(1.0 + ac_60_290 / dr_60_290, 4)

    # Steady-state samples [30s, 295s] for per-DP-rank decode batch & KV cache usage %
    samples_30_295 = [s for s in ts_samples if 30.0 <= s["elapsed_s"] <= 295.0]
    if not samples_30_295:
        samples_30_295 = [s for s in ts_samples if s["in_steady_window"]] or ts_samples

    all_dp_rank_obs: list[float] = []
    per_dp_rank_stats: dict[str, dict[str, float]] = {}
    eng_keys = sorted(
        {k for s in samples_30_295 for k in s["decode_running_by_eng"].keys()},
        key=lambda x: (x.split(":")[0], int(x.split(":")[1]) if x.split(":")[1].isdigit() else 0),
    )
    for ek in eng_keys:
        vals = [float(s["decode_running_by_eng"].get(ek, 0.0)) for s in samples_30_295]
        all_dp_rank_obs.extend(vals)
        per_dp_rank_stats[ek] = {
            "mean": round(statistics.mean(vals), 2) if vals else 0.0,
            "median": round(statistics.median(vals), 2) if vals else 0.0,
            "max": round(max(vals), 1) if vals else 0.0,
        }

    decode_tot_vals = [float(s["decode_running_total"]) for s in samples_30_295]
    prefill_tot_vals = [float(s["prefill_running_total"]) for s in samples_30_295]

    decode_kv_steady_mean_pct = (
        round(statistics.mean(s["decode_kv_mean_pct"] for s in samples_30_295), 4)
        if samples_30_295
        else 0.0
    )
    decode_kv_peak_rank_pct = round(
        max((s["decode_kv_max_pct"] for s in ts_samples), default=0.0), 4
    )
    prefill_kv_steady_mean_pct = (
        round(statistics.mean(s["prefill_kv_mean_pct"] for s in samples_30_295), 4)
        if samples_30_295
        else 0.0
    )
    prefill_kv_peak_rank_pct = round(
        max((s["prefill_kv_max_pct"] for s in ts_samples), default=0.0), 4
    )

    prefill_preemptions = p_m1["preemptions_total"] - p_m0["preemptions_total"]
    decode_preemptions = d_m1["preemptions_total"] - d_m0["preemptions_total"]

    # Detect counter reset (indicative of server/pod restart mid-run)
    counter_reset_detected = (
        (d_m1["generation_tokens_total"] < d_m0["generation_tokens_total"])
        or (p_m1["prompt_tokens_total"] < p_m0["prompt_tokens_total"])
        or (not snap_m1["prefill_all_ok"])
        or (not snap_m1["decode_all_ok"])
        or (not healthy_after)
    )

    # Per-Prefill-pod request & prompt token distribution (h0 vs h1)
    prefill_pod_dist: dict[str, Any] = {}
    tot_p_pod_reqs = 0.0
    for pk, p1_stat in snap_m1["prefill_per_pod"].items():
        p0_stat = snap_m0["prefill_per_pod"].get(pk, {})
        if "error" not in p1_stat and "error" not in p0_stat:
            r_delta = max(
                0.0,
                p1_stat.get("request_success_total", 0.0)
                - p0_stat.get("request_success_total", 0.0),
            )
            tok_delta = max(
                0.0,
                p1_stat.get("prompt_tokens_total", 0.0)
                - p0_stat.get("prompt_tokens_total", 0.0),
            )
            tot_p_pod_reqs += r_delta
            prefill_pod_dist[pk] = {
                "requests_delta": round(r_delta, 1),
                "prompt_tokens_delta": round(tok_delta, 1),
            }
    for pk in prefill_pod_dist:
        prefill_pod_dist[pk]["requests_share_pct"] = (
            round(100.0 * prefill_pod_dist[pk]["requests_delta"] / tot_p_pod_reqs, 2)
            if tot_p_pod_reqs > 0
            else 0.0
        )

    # Per-Decode-instance & per-DP-engine request/token distribution
    decode_inst_dist: dict[str, Any] = {}
    for dk, d1_stat in snap_m1["decode_per_instance"].items():
        d0_stat = snap_m0["decode_per_instance"].get(dk, {})
        if "error" not in d1_stat and "error" not in d0_stat:
            decode_inst_dist[dk] = {
                "requests_delta": round(
                    max(
                        0.0,
                        d1_stat.get("request_success_total", 0.0)
                        - d0_stat.get("request_success_total", 0.0),
                    ),
                    1,
                ),
                "generation_tokens_delta": round(
                    max(
                        0.0,
                        d1_stat.get("generation_tokens_total", 0.0)
                        - d0_stat.get("generation_tokens_total", 0.0),
                    ),
                    1,
                ),
            }

    decode_reqs_by_eng = {
        k: round(
            d_m1["req_success_by_eng"].get(k, 0.0)
            - d_m0["req_success_by_eng"].get(k, 0.0),
            1,
        )
        for k in eng_keys
    }
    decode_gen_toks_by_eng = {
        k: round(
            d_m1["gen_tok_by_eng"].get(k, 0.0) - d_m0["gen_tok_by_eng"].get(k, 0.0), 1
        )
        for k in eng_keys
    }

    # Client-side distributions: all_requests vs in_window
    all_tps = [r["decode_tps"] for r in ok_results if r["decode_tps"] is not None]
    win_tps = [r["decode_tps"] for r in in_win_results if r["decode_tps"] is not None]
    all_ttft = [r["ttft"] for r in ok_results if r["ttft"] is not None]
    win_ttft = [r["ttft"] for r in in_win_results if r["ttft"] is not None]
    all_tpot = [r["tpot_ms"] for r in ok_results if r["tpot_ms"] is not None]
    win_tpot = [r["tpot_ms"] for r in in_win_results if r["tpot_ms"] is not None]
    all_e2e = [r["e2e"] for r in ok_results if r["e2e"] is not None]
    win_e2e = [r["e2e"] for r in in_win_results if r["e2e"] is not None]

    all_tps_dist = summarize_dist_tps(all_tps)
    win_tps_dist = summarize_dist_tps(win_tps)
    all_ttft_dist = summarize_dist_lat(all_ttft)
    win_ttft_dist = summarize_dist_lat(win_ttft)
    all_tpot_dist = summarize_dist_lat(all_tpot)
    win_tpot_dist = summarize_dist_lat(win_tpot)
    all_e2e_dist = summarize_dist_lat(all_e2e)
    win_e2e_dist = summarize_dist_lat(win_e2e)

    # Evaluate the 5 Step 3 STOP RULES
    gain_vs_prev_pct: float | None = None
    if prev_steady_out_tok_s is not None and prev_steady_out_tok_s > 0:
        gain_vs_prev_pct = round(
            100.0 * (steady_out_tok_s - prev_steady_out_tok_s) / prev_steady_out_tok_s,
            3,
        )

    stop_reasons: list[str] = []
    if (
        concurrency >= 64
        and gain_vs_prev_pct is not None
        and gain_vs_prev_pct < 5.0
    ):
        stop_reasons.append(
            f"Rule 1: steady_out_tok_s gain {gain_vs_prev_pct:+.2f}% < +5.0% vs previous level ({prev_steady_out_tok_s:,.2f} -> {steady_out_tok_s:,.2f} tok/s)"
        )
    if (all_tps_dist["count"] > 0 and all_tps_dist["p90_low"] < 20.0) or (
        win_tps_dist["count"] > 0 and win_tps_dist["p90_low"] < 20.0
    ):
        stop_reasons.append(
            f"Rule 2: P90_low per-user decode speed < 20.0 tok/s (all={all_tps_dist['p90_low']:.2f}, in_window={win_tps_dist['p90_low']:.2f} tok/s)"
        )
    if (
        decode_preemptions > 0
        or prefill_preemptions > 0
        or counter_reset_detected
    ):
        stop_reasons.append(
            f"Rule 3: KV preemption or pod restart/unhealthy (P_preempt={prefill_preemptions:.0f}, D_preempt={decode_preemptions:.0f}, restart_or_unhealthy={counter_reset_detected}, issues={issues_after})"
        )
    if len(err_results) > 0:
        stop_reasons.append(
            f"Rule 4: failed_requests={len(err_results)} > 0 (sample: {err_results[0]['error']})"
        )
    if dspark_tau < 1.5 or tau_abort_reason:
        stop_reasons.append(
            f"Rule 5: DSpark tau < 1.5 (tau={dspark_tau:.4f}, abort={tau_abort_reason})"
        )

    summary: dict[str, Any] = {
        "run_name": run_name,
        "shape_name": shape_name,
        "ep_per_decode_instance": ep_size,
        "concurrency": concurrency,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": {
            "model": model,
            "target_isl": target_isl,
            "calibrated_word_count": word_count,
            "target_osl": target_osl,
            "ignore_eos": bool(ignore_eos),
            "temperature": temperature,
            "target_cache_hit_pct": round(cache_hit_ratio * 100.0, 2),
            "realized_cache_hit_pct": round(realized_cache_hit_pct, 4),
            "mean_prompt_tokens": round(
                total_prompt_all / len(ok_results), 2
            )
            if ok_results
            else 0.0,
            "mean_completion_tokens": round(
                total_compl_all / len(ok_results), 2
            )
            if ok_results
            else 0.0,
        },
        "window": {
            "steady_window_s": steady_seconds,
            "drain_wall_s": round(drain_wall_s, 3),
            "total_requests_launched": len(results),
            "ok_requests": len(ok_results),
            "in_window_completed_requests": len(in_win_results),
            "straddling_requests": len(ok_results) - len(in_win_results),
            "failed_requests": len(err_results),
            "errors_sample": [r["error"] for r in err_results[:5]],
        },
        "throughput": {
            "steady_out_tok_s": round(steady_out_tok_s, 2),
            "out_tok_s_per_decode_chip": round(out_tok_s_per_decode_chip, 2),
            "out_tok_s_per_total_chip": round(out_tok_s_per_total_chip, 2),
            "decode_chips": decode_chips,
            "total_chips": total_chips,
            "steady_in_tok_s": round(steady_in_tok_s, 2),
            "steady_total_tok_s": round(steady_total_tok_s, 2),
            "server_gen_tok_s_60_290": server_gen_tok_s_60_290,
            "drain_wall_out_tok_s": round(
                total_compl_all / drain_wall_s, 2
            )
            if drain_wall_s > 0
            else 0.0,
            "gain_vs_prev_pct": gain_vs_prev_pct,
        },
        "per_user_decode_tps": {
            "all_requests": all_tps_dist,
            "in_window": win_tps_dist,
        },
        "ttft_s": {
            "all_requests": all_ttft_dist,
            "in_window": win_ttft_dist,
        },
        "tpot_ms": {
            "all_requests": all_tpot_dist,
            "in_window": win_tpot_dist,
        },
        "e2e_s": {
            "all_requests": all_e2e_dist,
            "in_window": win_e2e_dist,
        },
        "dspark": {
            "mean_accept_len_tau": round(dspark_tau, 4),
            "steady_60_290_tau": dspark_tau_60_290,
            "num_drafts": round(d_drafts_total, 1),
            "num_draft_tokens": round(d_draft_toks_total, 1),
            "num_accepted": round(d_accepted_total, 1),
            "per_pos_accept_rates": d_pos_rates,
        },
        "decode_batch_30_295": {
            "per_dp_rank_mean": round(statistics.mean(all_dp_rank_obs), 2)
            if all_dp_rank_obs
            else 0.0,
            "per_dp_rank_median": round(statistics.median(all_dp_rank_obs), 2)
            if all_dp_rank_obs
            else 0.0,
            "per_dp_rank_max": round(max(all_dp_rank_obs), 1)
            if all_dp_rank_obs
            else 0.0,
            "total_running_mean": round(statistics.mean(decode_tot_vals), 2)
            if decode_tot_vals
            else 0.0,
            "total_running_median": round(statistics.median(decode_tot_vals), 2)
            if decode_tot_vals
            else 0.0,
            "total_running_max": round(max(decode_tot_vals), 1)
            if decode_tot_vals
            else 0.0,
            "prefill_running_mean": round(statistics.mean(prefill_tot_vals), 2)
            if prefill_tot_vals
            else 0.0,
            "prefill_running_max": round(max(prefill_tot_vals), 1)
            if prefill_tot_vals
            else 0.0,
            "by_engine": per_dp_rank_stats,
        },
        "kv_cache_usage_pct": {
            "decode_steady_mean_pct": decode_kv_steady_mean_pct,
            "decode_peak_single_rank_pct": decode_kv_peak_rank_pct,
            "prefill_steady_mean_pct": prefill_kv_steady_mean_pct,
            "prefill_peak_single_rank_pct": prefill_kv_peak_rank_pct,
        },
        "preemptions": {
            "prefill": round(prefill_preemptions, 1),
            "decode": round(decode_preemptions, 1),
            "counter_reset_or_unhealthy": counter_reset_detected,
        },
        "routing_distribution": {
            "prefill_pods": prefill_pod_dist,
            "decode_instances": decode_inst_dist,
            "decode_requests_by_engine": decode_reqs_by_eng,
            "decode_gen_tokens_by_engine": decode_gen_toks_by_eng,
            "epp_disagg_decisions_delta": round(
                max(
                    0.0,
                    snap_m1["epp"].get("disagg_decisions_total", 0.0)
                    - snap_m0["epp"].get("disagg_decisions_total", 0.0),
                ),
                1,
            )
            if isinstance(snap_m1["epp"], dict)
            and "error" not in snap_m1["epp"]
            else None,
        },
        "stop_evaluation": {
            "should_stop": bool(len(stop_reasons) > 0),
            "reasons": stop_reasons,
        },
    }

    # Save compact summary JSON + timeseries JSONL + gzipped per-request JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{level_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Strip raw chunk_times list before saving per-request JSON.gz to keep size compact
    for r in results:
        r.pop("chunk_times", None)
    raw_req_path = raw_out_dir / f"{level_tag}_requests.json.gz"
    with gzip.open(raw_req_path, "wt", encoding="utf-8") as gz:
        json.dump(results, gz)

    copy_if_mirror(summary_path, gcs_mirror_dir)
    copy_if_mirror(ts_path, gcs_mirror_dir)
    copy_if_mirror(raw_req_path, gcs_mirror_dir)

    print("-" * 108, flush=True)
    print(
        f"[DONE {level_tag}] Out={steady_out_tok_s:,.2f} tok/s "
        f"({out_tok_s_per_decode_chip:,.2f}/dec-chip, {out_tok_s_per_total_chip:,.2f}/tot-chip, "
        f"gain={f'{gain_vs_prev_pct:+.2f}%' if gain_vs_prev_pct is not None else 'baseline'}) | "
        f"SrvGen(60-290s)={server_gen_tok_s_60_290} tok/s | "
        f"Total(in+out)={steady_total_tok_s:,.2f} tok/s",
        flush=True,
    )
    print(
        f"  Per-User tok/s (In-Win | All): "
        f"{win_tps_dist['median']:.2f}/{win_tps_dist['p90_low']:.2f}/{win_tps_dist['p95_low']:.2f} | "
        f"{all_tps_dist['median']:.2f}/{all_tps_dist['p90_low']:.2f}/{all_tps_dist['p95_low']:.2f} "
        f"(TPOT All P50/P95/P99={all_tpot_dist['p50']:.2f}/{all_tpot_dist['p95']:.2f}/{all_tpot_dist['p99']:.2f} ms)",
        flush=True,
    )
    print(
        f"  TTFT (s) All P50/P95/P99={all_ttft_dist['p50']:.3f}/{all_ttft_dist['p95']:.3f}/{all_ttft_dist['p99']:.3f} | "
        f"In-Win P50/P95/P99={win_ttft_dist['p50']:.3f}/{win_ttft_dist['p95']:.3f}/{win_ttft_dist['p99']:.3f}",
        flush=True,
    )
    print(
        f"  DSpark tau={dspark_tau:.4f} (60-290s={dspark_tau_60_290}) | "
        f"D_batch/DP={summary['decode_batch_30_295']['per_dp_rank_mean']:.2f}/"
        f"{summary['decode_batch_30_295']['per_dp_rank_median']:.1f}/"
        f"{summary['decode_batch_30_295']['per_dp_rank_max']:.1f} "
        f"(tot={summary['decode_batch_30_295']['total_running_mean']:.1f}) | "
        f"KV(D)={decode_kv_steady_mean_pct:.3f}%/{decode_kv_peak_rank_pct:.3f}% "
        f"KV(P)={prefill_kv_steady_mean_pct:.3f}%/{prefill_kv_peak_rank_pct:.3f}% | "
        f"Preempt(P/D)={prefill_preemptions:.0f}/{decode_preemptions:.0f} | "
        f"Fail={len(err_results)}/{len(results)} | CacheHit={realized_cache_hit_pct:.3f}%",
        flush=True,
    )
    print(f"  Prefill Pod Split: {prefill_pod_dist}", flush=True)
    if stop_reasons:
        print(f"  [STOP RULE TRIGGERED] {stop_reasons}", flush=True)
    print("=" * 108, flush=True)
    return summary


async def main_async(args: argparse.Namespace) -> int:
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    prefill_urls = [x.strip() for x in args.prefill_urls.split(",") if x.strip()]
    decode_urls = [x.strip() for x in args.decode_urls.split(",") if x.strip()]
    completions_url = f"{args.router_url.rstrip('/')}/v1/completions"
    out_dir = Path(args.out_dir)
    raw_out_dir = Path(args.raw_out_dir)
    gcs_mirror_dir = Path(args.gcs_mirror_dir) if args.gcs_mirror_dir else None

    connector = aiohttp.TCPConnector(
        limit=2048, limit_per_host=2048, ttl_dns_cache=300, force_close=False
    )
    timeout = aiohttp.ClientTimeout(total=1800.0, sock_connect=30.0, sock_read=600.0)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        ok_h, issues = await check_all_healthy(session, prefill_urls, decode_urls)
        if not ok_h:
            raise RuntimeError(f"Initial health check failed: {issues}")
        print(
            f"[INIT] All {len(prefill_urls)} Prefill + {len(decode_urls)} Decode endpoints healthy.",
            flush=True,
        )

        if args.calibrated_word_count > 0:
            word_count = args.calibrated_word_count
            print(f"[INIT] Using provided calibrated_word_count={word_count}", flush=True)
        else:
            await reset_all_prefix_caches(session, prefill_urls, decode_urls)
            word_count = await calibrate_word_count(
                session,
                completions_url=completions_url,
                model=args.model,
                target_isl=args.target_isl,
                run_seed=args.run_seed,
            )
            print(
                f"[INIT] Calibrated exact word_count={word_count} for target_isl={args.target_isl}.",
                flush=True,
            )

        if args.warmup_seconds > 0:
            await reset_all_prefix_caches(session, prefill_urls, decode_urls)
            await run_warmup(
                session,
                completions_url=completions_url,
                model=args.model,
                word_count=word_count,
                warmup_concurrency=args.warmup_concurrency,
                warmup_seconds=args.warmup_seconds,
                warmup_max_tokens=args.warmup_max_tokens,
                temperature=args.temperature,
                run_seed=args.run_seed,
            )

        prev_out = args.prev_steady_out_tok_s if args.prev_steady_out_tok_s > 0 else None
        all_level_summaries: list[dict[str, Any]] = []

        for c in concurrencies:
            lvl_summary = await run_concurrency_level(
                session,
                run_name=args.run_name,
                shape_name=args.shape_name,
                ep_size=args.ep_size,
                concurrency=c,
                word_count=word_count,
                target_isl=args.target_isl,
                target_osl=args.target_osl,
                ignore_eos=args.ignore_eos,
                temperature=args.temperature,
                cache_hit_ratio=args.cache_hit_ratio,
                num_trunks=args.num_trunks,
                steady_seconds=args.steady_seconds,
                decode_chips=args.decode_chips,
                total_chips=args.total_chips,
                completions_url=completions_url,
                model=args.model,
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                epp_metrics_url=args.epp_metrics_url,
                run_seed=args.run_seed,
                out_dir=out_dir,
                raw_out_dir=raw_out_dir,
                gcs_mirror_dir=gcs_mirror_dir,
                prev_steady_out_tok_s=prev_out,
            )
            all_level_summaries.append(lvl_summary)
            prev_out = lvl_summary["throughput"]["steady_out_tok_s"]

            sweep_prog_path = out_dir / f"{args.run_name}_{args.shape_name}_sweep_summary.json"
            sweep_prog_path.write_text(
                json.dumps(
                    {
                        "run_name": args.run_name,
                        "shape_name": args.shape_name,
                        "ep_per_decode_instance": args.ep_size,
                        "calibrated_word_count": word_count,
                        "completed_concurrencies": [
                            s["concurrency"] for s in all_level_summaries
                        ],
                        "stopped_early": lvl_summary["stop_evaluation"]["should_stop"],
                        "stop_reasons": lvl_summary["stop_evaluation"]["reasons"],
                        "levels": all_level_summaries,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            copy_if_mirror(sweep_prog_path, gcs_mirror_dir)

            if lvl_summary["stop_evaluation"]["should_stop"]:
                print(
                    f"\n[SWEEP COMPLETE] Stopping concurrency escalation after c={c}: "
                    f"{lvl_summary['stop_evaluation']['reasons']}",
                    flush=True,
                )
                break

    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="llm-d Step 3 Shape Sweep Benchmark Harness")
    ap.add_argument("--run-name", default="r1a")
    ap.add_argument("--shape-name", default="s1")
    ap.add_argument("--ep-size", type=int, default=16)
    ap.add_argument(
        "--router-url", default="http://pd-disaggregation-epp.default.svc.cluster.local:80"
    )
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--target-isl", type=int, default=2048)
    ap.add_argument("--calibrated-word-count", type=int, default=0)
    ap.add_argument("--target-osl", type=int, default=1024)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--no-ignore-eos", action="store_false", dest="ignore_eos")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--cache-hit-ratio", type=float, default=0.0)
    ap.add_argument("--num-trunks", type=int, default=32)
    ap.add_argument(
        "--concurrencies", default="32,64,128,256,384,512,768,1024"
    )
    ap.add_argument("--steady-seconds", type=float, default=300.0)
    ap.add_argument("--warmup-concurrency", type=int, default=16)
    ap.add_argument("--warmup-seconds", type=float, default=15.0)
    ap.add_argument("--warmup-max-tokens", type=int, default=64)
    ap.add_argument("--decode-chips", type=int, default=8)
    ap.add_argument("--total-chips", type=int, default=16)
    ap.add_argument(
        "--prefill-urls", default="http://192.168.100.43:8000,http://192.168.100.38:8000"
    )
    ap.add_argument("--decode-urls", default="http://192.168.100.198:9400")
    ap.add_argument(
        "--epp-metrics-url",
        default="http://pd-disaggregation-epp.default.svc.cluster.local:9090/metrics",
    )
    ap.add_argument("--run-seed", type=int, default=101)
    ap.add_argument("--prev-steady-out-tok-s", type=float, default=0.0)
    ap.add_argument("--out-dir", default="/dev/shm/step3_runs/r1a_ep16")
    ap.add_argument("--raw-out-dir", default="/dev/shm/step3_runs/r1a_ep16_raw")
    ap.add_argument(
        "--gcs-mirror-dir",
        default="/models/dsv41-v7x-llmd-20261106/step3_runs/r1a_ep16",
    )
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
