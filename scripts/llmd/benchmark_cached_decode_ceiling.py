#!/usr/bin/env python3
"""Step 3 R3: Decode-Only Ceiling Benchmark Harness (`S1 = 2048:1024` High-Concurrency Sweep).

Pure asyncio + aiohttp (NO torch / transformers imports; strictly enforces PYTHONPATH="").

Design to Isolate Pure Pool B Decode Ceiling:
  1. Resets prefix caches once at R3 startup, then builds a compact pool of `16` distinct `S1`
     trunks (`--num-trunks 16`), each calibrated via vLLM's `/tokenize` endpoint to have
     `>= 2048` shared prefix tokens before the unique request tag (aligned to
     `PREFIX_CACHE_RETENTION_INTERVAL=1024` = `2 x 1024` pinned hybrid KV + Mamba/CSA blocks)
     plus a tiny unique per-request suffix (`~8-10` tokens).
  2. Explicitly pre-warms all `16` trunks across **ALL `8` DP ranks (`X-data-parallel-rank: 0..7`)**
     on **BOTH** `llmd-prefill-poola-h0` (`<private-ip>:8000`) and `llmd-prefill-poola-h1`
     (`<private-ip>:8000`) before starting the sweep (`16 * 2048 = 32,768` tok/rank = `2.2%` of
     each Prefill rank's `1.464M`-token KV cache -> zero evictions, `~99.5%` prefix cache hit in
     `<30 ms`, keeping Prefill `<10%` utilized so Prefill NEVER gates Decode!).
  3. Sweeps `c = 256, 384, 512, 768, 1024, 1280` at `300.0 s` steady window per level with
     `max_tokens=1024, min_tokens=1024, ignore_eos=True` through `llm-d`, saving per-level
     `r3_c{c}.json`, `r3_c{c}_timeseries.jsonl`, `r3_c{c}_requests.json.gz`, and progressive
     `r3_decode_ceiling_summary.json`, and stopping when `gain < +5.0%` at
     `c >= min_concurrency_for_gain_stop` (`1024`) or on any `Rule 2-5` trigger.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

# Strictly isolate from torch / vllm / libtpu PYTHONPATH before importing aiohttp
os.environ["PYTHONPATH"] = ""
assert "torch" not in sys.modules and "transformers" not in sys.modules, (
    "FATAL: torch/transformers must never be imported in run_r3_decode_ceiling.py"
)

import aiohttp  # noqa: E402

VOCAB_WORDS = [
    "quantum", "tensor", "matrix", "pipeline", "compiler", "manifold", "gradient", "register",
    "topology", "scheduler", "bandwidth", "latency", "harmonic", "spectrum", "protocol", "catalyst",
    "synthesis", "dynamics", "geometry", "operator", "invariant", "partition", "coherent", "parallel",
    "dispatch", "horizon", "velocity", "momentum", "symmetry", "algorithm", "precision", "throughput",
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


def summarize_tps(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90_low": 0.0, "p95_low": 0.0, "p90_hi": 0.0}
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(percentile(vals, 50), 4),
        "p90_low": round(percentile(vals, 10), 4),
        "p95_low": round(percentile(vals, 5), 4),
        "p90_hi": round(percentile(vals, 90), 4),
    }


def summarize_lat(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "p50": round(percentile(vals, 50), 4),
        "p90": round(percentile(vals, 90), 4),
        "p95": round(percentile(vals, 95), 4),
        "p99": round(percentile(vals, 99), 4),
    }


async def reset_all_prefix_caches(
    http_session: aiohttp.ClientSession, prefill_urls: list[str], decode_urls: list[str]
) -> None:
    await asyncio.sleep(1.0)
    for label, urls in [("prefill", prefill_urls), ("decode", decode_urls)]:
        for idx, u in enumerate(urls):
            url = f"{u.rstrip('/')}/reset_prefix_cache"
            ok = False
            last_code = 0
            for _ in range(10):
                try:
                    async with http_session.post(
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
                f"  [R3 reset_prefix_cache] {label}[{idx}] ({u}): "
                f"{'OK (HTTP 200)' if ok else f'WARN (HTTP {last_code})'}",
                flush=True,
            )
            if not ok:
                raise RuntimeError(f"Failed /reset_prefix_cache on {u} (HTTP {last_code})")
    await asyncio.sleep(1.0)


async def check_all_healthy(
    http_session: aiohttp.ClientSession, prefill_urls: list[str], decode_urls: list[str]
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for label, urls in [("prefill", prefill_urls), ("decode", decode_urls)]:
        for idx, u in enumerate(urls):
            try:
                async with http_session.get(
                    f"{u.rstrip('/')}/health", timeout=aiohttp.ClientTimeout(total=8.0)
                ) as resp:
                    if resp.status != 200:
                        issues.append(f"{label}[{idx}] {u} -> HTTP {resp.status}")
            except Exception as exc:
                issues.append(f"{label}[{idx}] {u} -> {type(exc).__name__}: {exc}")
    return (len(issues) == 0), issues


async def calibrate_exact_2048_prefix(
    http_session: aiohttp.ClientSession,
    tokenize_base_url: str,
    model_name: str,
    trunk_id: int,
    target_prefix_tokens: int = 2048,
) -> tuple[str, int]:
    """Builds a deterministic technical trunk calibrated via /tokenize so `pfx + '\\n\\nQuery ['` has >= target_prefix_tokens + 2 shared tokens."""
    header = (
        f"System Architecture Reference Document #{trunk_id:03d}: "
        f"Analyze the distributed tensor-parallel and expert-parallel execution schedule below and "
        f"derive a detailed step-by-step performance model.\n\n"
    )
    paragraphs: list[str] = []
    for sec in range(1, 150):
        w0 = VOCAB_WORDS[(trunk_id * 7 + sec * 3) % len(VOCAB_WORDS)]
        w1 = VOCAB_WORDS[(trunk_id * 11 + sec * 5) % len(VOCAB_WORDS)]
        w2 = VOCAB_WORDS[(trunk_id * 13 + sec * 7) % len(VOCAB_WORDS)]
        paragraphs.append(
            f"Section {sec}.{trunk_id}: In stage {sec}, the {w0} subsystem coordinates {w1} buffers "
            f"across the {w2} interconnect while maintaining bounded tail latency under continuous load. "
        )
    full_text = header + "".join(paragraphs)

    tok_url = f"{tokenize_base_url.rstrip('/')}/tokenize"
    # Target 2050 tokens on `cand + "\n\nQuery ["` so the first 2048 tokens (2 x 1024 blocks) are 100% invariant
    desired_shared = target_prefix_tokens + 2
    lo, hi = 1000, len(full_text)
    best = full_text[:10500]
    best_cnt = 0
    for _ in range(18):
        mid = (lo + hi) // 2
        cand = full_text[:mid]
        probe_str = f"{cand}\n\nQuery ["
        async with http_session.post(
            tok_url,
            json={"model": model_name, "prompt": probe_str, "add_special_tokens": True},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"/tokenize failed with HTTP {resp.status}")
            data = await resp.json()
            cnt = int(data.get("count") or len(data.get("tokens") or []))
        if cnt < desired_shared:
            lo = mid + 1
        elif cnt > desired_shared + 2:
            hi = mid - 1
            best = cand
            best_cnt = cnt
        else:
            best = cand
            best_cnt = cnt
            break
    if best_cnt < desired_shared:
        cand = full_text[:lo]
        async with http_session.post(
            tok_url,
            json={"model": model_name, "prompt": f"{cand}\n\nQuery [", "add_special_tokens": True},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()
            best = cand
            best_cnt = int(data.get("count") or len(data.get("tokens") or []))
    return best, best_cnt


async def prewarm_all_prefill_ranks(
    http_session: aiohttp.ClientSession,
    *,
    prefill_urls: list[str],
    router_base_url: str,
    model_name: str,
    prefixes: list[str],
    prefill_dp_size: int = 8,
) -> dict[str, Any]:
    """Pre-warms every trunk across ALL DP ranks (0..7) on EVERY Prefill pod + router path."""
    t0 = time.perf_counter()
    ok_direct = 0
    tot_direct = 0

    for p_url in prefill_urls:
        comp_url = f"{p_url.rstrip('/')}/v1/completions"
        tasks = []
        for dp_r in range(prefill_dp_size):
            for t_id, pfx in enumerate(prefixes):
                tot_direct += 1
                body = {
                    "model": model_name,
                    "prompt": f"{pfx}\n\nQuery [warm-{t_id:02d}-dp{dp_r}]: Derive breakdown.",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
                headers = {"X-data-parallel-rank": str(dp_r)}
                tasks.append(http_session.post(comp_url, json=body, headers=headers))
        for start in range(0, len(tasks), 16):
            batch_coros = tasks[start : start + 16]
            resps = await asyncio.gather(*batch_coros, return_exceptions=True)
            for r in resps:
                if not isinstance(r, Exception):
                    if r.status == 200:
                        ok_direct += 1
                    await r.release()

    # Warm router + EndpointPicker + Decode XLA buckets across 2 passes
    router_comp = f"{router_base_url.rstrip('/')}/v1/completions"
    ok_router = 0
    for rep in range(2):
        r_tasks = []
        for t_id, pfx in enumerate(prefixes):
            body = {
                "model": model_name,
                "prompt": f"{pfx}\n\nQuery [rwarm-{rep}-{t_id:02d}]: Derive breakdown.",
                "max_tokens": 32,
                "min_tokens": 32,
                "ignore_eos": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            }
            r_tasks.append(
                http_session.post(
                    router_comp,
                    json=body,
                    headers={"X-Session-ID": f"r3-trunk-{t_id}"},
                )
            )
        resps = await asyncio.gather(*r_tasks, return_exceptions=True)
        for r in resps:
            if not isinstance(r, Exception):
                if r.status == 200:
                    ok_router += 1
                await r.release()

    elapsed = round(time.perf_counter() - t0, 2)
    print(
        f"[R3 PREWARM] Direct Prefill DP-rank warmup: {ok_direct}/{tot_direct} OK | "
        f"Router+Decode warmup: {ok_router}/{2*len(prefixes)} OK in {elapsed:.2f}s",
        flush=True,
    )
    if ok_direct < tot_direct or ok_router < 2 * len(prefixes):
        raise RuntimeError(
            f"R3 prewarm incomplete: direct={ok_direct}/{tot_direct}, router={ok_router}/{2*len(prefixes)}"
        )
    return {
        "direct_prefill_ok": ok_direct,
        "direct_prefill_total": tot_direct,
        "router_warmup_ok": ok_router,
        "router_warmup_total": 2 * len(prefixes),
        "wall_s": elapsed,
    }


def parse_vllm_prom(text: str, prefix: str) -> dict[str, Any]:
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


async def scrape_all_prom(
    http_session: aiohttp.ClientSession,
    prefill_urls: list[str],
    decode_urls: list[str],
) -> dict[str, Any]:
    async def _one(url: str) -> str:
        async with http_session.get(
            f"{url.rstrip('/')}/metrics", timeout=aiohttp.ClientTimeout(total=8.0)
        ) as resp:
            resp.raise_for_status()
            return await resp.text()

    p_tasks = [_one(u) for u in prefill_urls]
    d_tasks = [_one(u) for u in decode_urls]
    raw = await asyncio.gather(*p_tasks, *d_tasks, return_exceptions=True)
    p_raw = raw[: len(prefill_urls)]
    d_raw = raw[len(prefill_urls) :]

    p_parsed = []
    p_pods: dict[str, Any] = {}
    for i, res in enumerate(p_raw):
        tag = f"h{i}"
        if isinstance(res, Exception):
            p_pods[tag] = {"error": str(res)}
        else:
            parsed = parse_vllm_prom(res, tag)
            p_parsed.append(parsed)
            p_pods[tag] = {
                "url": prefill_urls[i],
                "running": sum(parsed["running_by_eng"].values()),
                "waiting": sum(parsed["waiting_by_eng"].values()),
                "prompt_tokens_total": sum(parsed["prompt_tok_by_eng"].values()),
                "request_success_total": sum(parsed["req_success_by_eng"].values()),
                "prefix_queries_total": parsed["prefix_queries_total"],
                "prefix_hits_total": parsed["prefix_hits_total"],
            }

    d_parsed = []
    d_insts: dict[str, Any] = {}
    for i, res in enumerate(d_raw):
        tag = f"d{i}"
        if isinstance(res, Exception):
            d_insts[tag] = {"error": str(res)}
        else:
            parsed = parse_vllm_prom(res, tag)
            d_parsed.append(parsed)
            d_insts[tag] = {
                "url": decode_urls[i],
                "running": sum(parsed["running_by_eng"].values()),
                "waiting": sum(parsed["waiting_by_eng"].values()),
                "generation_tokens_total": sum(parsed["gen_tok_by_eng"].values()),
                "request_success_total": sum(parsed["req_success_by_eng"].values()),
            }

    return {
        "prefill": merge_vllm_prom(p_parsed),
        "prefill_pods": p_pods,
        "decode": merge_vllm_prom(d_parsed),
        "decode_instances": d_insts,
        "scrape_errors": sum(1 for r in raw if isinstance(r, Exception)),
    }


async def send_completion_stream(
    http_session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    trunk_id: int,
    session_id: int,
    req_idx: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    t_first: float | None = None
    chunk_times: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    num_chunks = 0
    status_code = 0
    error_msg: str | None = None

    headers = {"X-Session-ID": f"r3-trunk-{trunk_id}"}
    try:
        async with http_session.post(url, json=payload, headers=headers) as resp:
            status_code = resp.status
            if status_code != 200:
                err_body = await resp.text()
                error_msg = f"HTTP {status_code}: {err_body[:300]}"
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
                        if c0.get("text"):
                            if t_first is None:
                                t_first = now
                            chunk_times.append(now)
                            num_chunks += 1
                    usage = chunk.get("usage")
                    if isinstance(usage, dict) and usage:
                        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                        details = usage.get("prompt_tokens_details")
                        if isinstance(details, dict) and details.get("cached_tokens") is not None:
                            cached_tokens = int(details["cached_tokens"])
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    t_end = time.perf_counter()
    e2e = t_end - t_start
    if t_first is None and error_msg is None:
        t_first = t_end
    ttft = (t_first - t_start) if t_first is not None else None
    eff_compl = completion_tokens or num_chunks
    decode_dur = max(t_end - (t_first if t_first is not None else t_end), 1e-6)
    decode_tps: float | None = None
    tpot_ms: float | None = None
    if error_msg is None and eff_compl >= 1:
        decode_tps = ((eff_compl - 1) / decode_dur) if eff_compl > 1 else (1.0 / decode_dur)
        tpot_ms = 1000.0 * (t_end - (t_first or t_end)) / max(eff_compl - 1, 1)

    return {
        "req_idx": req_idx,
        "session_id": session_id,
        "trunk_id": trunk_id,
        "t_start": t_start,
        "t_first": t_first,
        "t_end": t_end,
        "ttft": ttft,
        "e2e": e2e,
        "decode_tps": decode_tps,
        "tpot_ms": tpot_ms,
        "chunk_times": chunk_times,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": eff_compl,
        "cached_tokens": cached_tokens,
        "status_code": status_code,
        "error": error_msg,
    }


async def run_r3_level(
    http_session: aiohttp.ClientSession,
    *,
    concurrency_n: int,
    duration_s: float,
    target_url: str,
    prefill_urls: list[str],
    decode_urls: list[str],
    model_name: str,
    prefixes: list[str],
    osl_tokens: int,
    temperature: float,
    top_p: float,
    num_decode_chips: int,
    num_total_chips: int,
    out_dir: Path,
    raw_out_dir: Path,
    prev_steady_out_tok_s: float | None,
    min_concurrency_for_gain_stop: int,
) -> dict[str, Any]:
    level_tag = f"r3_c{concurrency_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    healthy_before, issues_before = await check_all_healthy(http_session, prefill_urls, decode_urls)
    if not healthy_before:
        raise RuntimeError(f"Pre-level R3 health check failed: {issues_before}")

    print("\n" + "=" * 108, flush=True)
    print(
        f"[START {level_tag}] concurrency={concurrency_n} | S1 (~2056:{osl_tokens}, 16 cached 2048-tok trunks) | "
        f"window={duration_s:.1f}s",
        flush=True,
    )
    print("=" * 108, flush=True)

    snap_m0 = await scrape_all_prom(http_session, prefill_urls, decode_urls)
    ts_samples: list[dict[str, Any]] = []
    ts_path = out_dir / f"{level_tag}_timeseries.jsonl"
    stop_poller = asyncio.Event()

    t_bench_start = time.perf_counter()
    t_window_end = t_bench_start + duration_s

    async def _poller() -> None:
        idx = 0
        with ts_path.open("w", encoding="utf-8") as ts_fp:
            while not stop_poller.is_set():
                t_rel = time.perf_counter() - t_bench_start
                try:
                    snap = await scrape_all_prom(http_session, prefill_urls, decode_urls)
                    rec = {
                        "sample_idx": idx,
                        "t_rel_s": round(t_rel, 3),
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "in_steady_window": bool(t_rel <= duration_s),
                        **snap,
                    }
                    ts_samples.append(rec)
                    ts_fp.write(json.dumps(rec) + "\n")
                    ts_fp.flush()
                    if idx % 6 == 0:
                        d = snap["decode"]
                        p = snap["prefill"]
                        r_vals = list(d["running_by_eng"].values())
                        r_med = statistics.median(r_vals) if r_vals else 0.0
                        r_min = min(r_vals) if r_vals else 0.0
                        r_max = max(r_vals) if r_vals else 0.0
                        dd = d["spec_num_drafts"] - snap_m0["decode"]["spec_num_drafts"]
                        da = d["spec_num_accepted"] - snap_m0["decode"]["spec_num_accepted"]
                        live_tau = round(1.0 + da / dd, 3) if dd > 0 else None
                        print(
                            f"  [t={t_rel:6.1f}s c={concurrency_n}] "
                            f"P_run={p['running_total']:.0f} (wait={p['waiting_total']:.0f}) P_kv={p['kv_mean']*100:.2f}% | "
                            f"D_run={d['running_total']:.0f} (wait={d['waiting_total']:.0f}, "
                            f"per_dp_min/med/max={r_min:.0f}/{r_med:.1f}/{r_max:.0f}) "
                            f"D_kv={d['kv_mean']*100:.2f}% (pk={d['kv_max']*100:.2f}%) | "
                            f"tau={live_tau} | "
                            f"preempt(P/D)={p['preemptions_total'] - snap_m0['prefill']['preemptions_total']:.0f}/"
                            f"{d['preemptions_total'] - snap_m0['decode']['preemptions_total']:.0f}",
                            flush=True,
                        )
                    idx += 1
                except Exception as exc:
                    print(f"  [METRICS WARN] {exc}", flush=True)
                try:
                    await asyncio.wait_for(stop_poller.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    pass

    poller_task = asyncio.create_task(_poller())
    dispatch_counter = [0]

    async def worker(s_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        turn = 0
        while time.perf_counter() < t_window_end:
            idx = dispatch_counter[0]
            dispatch_counter[0] += 1
            trunk_id = idx % len(prefixes)
            prompt = f"{prefixes[trunk_id]}\n\nQuery [n{concurrency_n}-s{s_id}-t{turn}]: Derive breakdown."
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": osl_tokens,
                "min_tokens": osl_tokens,
                "ignore_eos": True,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            res = await send_completion_stream(http_session, target_url, payload, trunk_id, s_id, idx)
            out.append(res)
            turn += 1
        return out

    worker_lists = await asyncio.gather(*(worker(i) for i in range(concurrency_n)))
    t_bench_end = time.perf_counter()
    stop_poller.set()
    await poller_task

    snap_m1 = await scrape_all_prom(http_session, prefill_urls, decode_urls)
    healthy_after, issues_after = await check_all_healthy(http_session, prefill_urls, decode_urls)

    wall_clock = t_bench_end - t_bench_start
    results = [r for lst in worker_lists for r in lst]
    results.sort(key=lambda r: r["req_idx"])
    ok_results = [r for r in results if r["error"] is None]
    err_results = [r for r in results if r["error"] is not None]

    window_output_tokens = 0.0
    window_input_tokens = 0.0
    window_completed = 0
    straddling = 0
    for r in ok_results:
        c_times = r.get("chunk_times") or []
        compl = float(r["completion_tokens"] or 0)
        prompt = float(r["prompt_tokens"] or 0)
        if r["t_end"] <= t_window_end:
            window_output_tokens += compl
            window_input_tokens += prompt
            window_completed += 1
        elif c_times:
            in_win = sum(1 for ct in c_times if ct <= t_window_end)
            if in_win > 0:
                straddling += 1
                window_output_tokens += compl * (in_win / len(c_times))
                window_input_tokens += prompt

    win_ok = [r for r in ok_results if r["t_end"] <= t_window_end]
    eval_win = win_ok if len(win_ok) >= max(4, concurrency_n // 4) else ok_results

    steady_out_tok_s = window_output_tokens / duration_s
    steady_in_tok_s = window_input_tokens / duration_s
    steady_total_tok_s = steady_in_tok_s + steady_out_tok_s
    out_per_dec = steady_out_tok_s / max(1, num_decode_chips)
    out_per_tot = steady_out_tok_s / max(1, num_total_chips)
    tot_compl = sum(float(r["completion_tokens"] or 0) for r in ok_results)
    tot_prompt = sum(float(r["prompt_tokens"] or 0) for r in ok_results)
    tot_cached = sum(float(r["cached_tokens"] or 0) for r in ok_results)
    drain_out_tok_s = tot_compl / wall_clock if wall_clock > 0 else 0.0
    api_hit_pct = (100.0 * tot_cached / tot_prompt) if tot_prompt > 0 else 0.0

    gain_vs_prev_pct: float | None = None
    if prev_steady_out_tok_s is not None and prev_steady_out_tok_s > 0:
        gain_vs_prev_pct = 100.0 * (steady_out_tok_s - prev_steady_out_tok_s) / prev_steady_out_tok_s

    all_tps_dist = summarize_tps([r["decode_tps"] for r in ok_results if r["decode_tps"] is not None])
    win_tps_dist = summarize_tps([r["decode_tps"] for r in eval_win if r["decode_tps"] is not None])
    all_ttft_dist = summarize_lat([r["ttft"] for r in ok_results if r["ttft"] is not None])
    win_ttft_dist = summarize_lat([r["ttft"] for r in eval_win if r["ttft"] is not None])
    all_tpot_dist = summarize_lat([r["tpot_ms"] for r in ok_results if r["tpot_ms"] is not None])
    win_tpot_dist = summarize_lat([r["tpot_ms"] for r in eval_win if r["tpot_ms"] is not None])

    d0 = snap_m0["decode"]
    d1 = snap_m1["decode"]
    p0 = snap_m0["prefill"]
    p1 = snap_m1["prefill"]

    delta_drafts = max(0.0, d1["spec_num_drafts"] - d0["spec_num_drafts"])
    delta_draft_tokens = max(0.0, d1["spec_num_draft_tokens"] - d0["spec_num_draft_tokens"])
    delta_accepted = max(0.0, d1["spec_num_accepted"] - d0["spec_num_accepted"])
    tau_full = (1.0 + delta_accepted / delta_drafts) if delta_drafts > 0 else 1.0

    pos_rates: dict[str, float] = {}
    if delta_drafts > 0:
        for k in sorted(d1["spec_per_pos"].keys(), key=lambda x: int(x)):
            dv = max(0.0, d1["spec_per_pos"].get(k, 0.0) - d0["spec_per_pos"].get(k, 0.0))
            pos_rates[f"pos_{k}"] = round(dv / delta_drafts, 4)

    win_60_290 = [
        s for s in ts_samples
        if 60.0 <= s["t_rel_s"] <= min(duration_s - 10.0, 290.0)
        and s.get("scrape_errors", 0) == 0
    ]
    server_gen_tok_s_60_290: float | None = None
    tau_60_290: float | None = None
    if len(win_60_290) >= 2:
        s_a, s_b = win_60_290[0], win_60_290[-1]
        dt_ab = s_b["t_rel_s"] - s_a["t_rel_s"]
        if dt_ab > 5.0:
            dgen = s_b["decode"]["generation_tokens_total"] - s_a["decode"]["generation_tokens_total"]
            server_gen_tok_s_60_290 = round(dgen / dt_ab, 2)
            dd_ab = s_b["decode"]["spec_num_drafts"] - s_a["decode"]["spec_num_drafts"]
            da_ab = s_b["decode"]["spec_num_accepted"] - s_a["decode"]["spec_num_accepted"]
            if dd_ab > 0:
                tau_60_290 = round(1.0 + da_ab / dd_ab, 4)

    win_30_295 = [
        s for s in ts_samples
        if 30.0 <= s["t_rel_s"] <= max(duration_s - 5.0, 30.0)
        and s.get("scrape_errors", 0) == 0
    ]
    eval_ts = win_30_295 if win_30_295 else ts_samples
    per_dp_samples: list[float] = []
    tot_run_samples: list[float] = []
    tot_wait_samples: list[float] = []
    pre_run_samples: list[float] = []
    dec_kv_means: list[float] = []
    pre_kv_means: list[float] = []
    for s in eval_ts:
        rv = list(s["decode"]["running_by_eng"].values())
        per_dp_samples.extend(rv)
        tot_run_samples.append(s["decode"]["running_total"])
        tot_wait_samples.append(s["decode"]["waiting_total"])
        pre_run_samples.append(s["prefill"]["running_total"])
        dec_kv_means.append(s["decode"]["kv_mean"] * 100.0)
        pre_kv_means.append(s["prefill"]["kv_mean"] * 100.0)

    dec_kv_peak_rank = max((s["decode"]["kv_max"] * 100.0 for s in ts_samples), default=0.0)
    pre_kv_peak_rank = max((s["prefill"]["kv_max"] * 100.0 for s in ts_samples), default=0.0)
    dec_preempt = max(0.0, d1["preemptions_total"] - d0["preemptions_total"])
    pre_preempt = max(0.0, p1["preemptions_total"] - p0["preemptions_total"])

    p_hits = max(0.0, p1["prefix_hits_total"] - p0["prefix_hits_total"])
    p_queries = max(0.0, p1["prefix_queries_total"] - p0["prefix_queries_total"])
    prom_hit_pct = (100.0 * p_hits / p_queries) if p_queries > 0 else 0.0

    # Pod splits
    p_pod_dist: dict[str, Any] = {}
    tot_p_req = 0.0
    for tag in sorted(snap_m1["prefill_pods"].keys()):
        b0 = snap_m0["prefill_pods"].get(tag, {})
        b1 = snap_m1["prefill_pods"].get(tag, {})
        if "error" not in b0 and "error" not in b1:
            dr = max(0.0, b1["request_success_total"] - b0["request_success_total"])
            dp = max(0.0, b1["prompt_tokens_total"] - b0["prompt_tokens_total"])
            tot_p_req += dr
            p_pod_dist[tag] = {"requests_delta": round(dr, 1), "prompt_tokens_delta": round(dp, 1)}
    for v in p_pod_dist.values():
        v["requests_share_pct"] = round(100.0 * v["requests_delta"] / tot_p_req, 2) if tot_p_req > 0 else 0.0

    d_inst_dist: dict[str, Any] = {}
    tot_d_req = 0.0
    for tag in sorted(snap_m1["decode_instances"].keys()):
        b0 = snap_m0["decode_instances"].get(tag, {})
        b1 = snap_m1["decode_instances"].get(tag, {})
        if "error" not in b0 and "error" not in b1:
            dr = max(0.0, b1["request_success_total"] - b0["request_success_total"])
            dg = max(0.0, b1["generation_tokens_total"] - b0["generation_tokens_total"])
            tot_d_req += dr
            d_inst_dist[tag] = {"requests_delta": round(dr, 1), "gen_tokens_delta": round(dg, 1)}
    for v in d_inst_dist.values():
        v["requests_share_pct"] = round(100.0 * v["requests_delta"] / tot_d_req, 2) if tot_d_req > 0 else 0.0

    counter_reset = (
        d1["generation_tokens_total"] < d0["generation_tokens_total"]
        or p1["prompt_tokens_total"] < p0["prompt_tokens_total"]
        or (not healthy_after)
    )
    failed_rate_pct = (100.0 * len(err_results) / len(results)) if results else 100.0

    # Stop Rules
    stop_reasons: list[str] = []
    if (
        gain_vs_prev_pct is not None
        and concurrency_n >= min_concurrency_for_gain_stop
        and gain_vs_prev_pct < 5.0
    ):
        stop_reasons.append(
            f"Rule 1 (Throughput Plateau >= c={min_concurrency_for_gain_stop}): "
            f"gain_vs_prev={gain_vs_prev_pct:+.2f}% < +5.0%"
        )
    if all_tps_dist["p90_low"] < 20.0 or win_tps_dist["p90_low"] < 20.0:
        stop_reasons.append(
            f"Rule 2 (Interactivity Floor): p90_low={min(all_tps_dist['p90_low'], win_tps_dist['p90_low']):.2f} < 20.0 tok/s"
        )
    if dec_preempt > 0 or dec_kv_peak_rank >= 95.0:
        stop_reasons.append(
            f"Rule 3 (KV Saturation / Preemption): decode_preemptions={dec_preempt:.0f}, "
            f"decode_kv_peak_rank={dec_kv_peak_rank:.2f}%"
        )
    if failed_rate_pct > 1.0 or counter_reset:
        stop_reasons.append(
            f"Rule 4 (Error / Crash Guard): failed_rate={failed_rate_pct:.2f}%, "
            f"counter_reset_or_unhealthy={counter_reset} ({issues_after})"
        )
    if concurrency_n >= 2048:
        stop_reasons.append(f"Rule 5 (Hard Ceiling): concurrency={concurrency_n} >= 2048")

    summary = {
        "run_name": "r3",
        "shape_name": "s1_cached",
        "ep_per_decode_instance": 8,
        "concurrency": concurrency_n,
        "concurrency_n": concurrency_n,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "window": {
            "steady_window_s": duration_s,
            "wall_incl_drain_s": round(wall_clock, 3),
            "total_requests_launched": len(results),
            "ok_requests_total": len(ok_results),
            "failed_requests": len(err_results),
            "failed_rate_pct": round(failed_rate_pct, 4),
            "completed_in_window": window_completed,
            "straddling_window_end": straddling,
            "errors_sample": [r["error"] for r in err_results[:5]],
        },
        "tokens": {
            "mean_isl": round(tot_prompt / len(ok_results), 2) if ok_results else 0.0,
            "mean_osl": round(tot_compl / len(ok_results), 2) if ok_results else 0.0,
            "api_cache_hit_pct": round(api_hit_pct, 4),
            "prometheus_prefill_hit_pct": round(prom_hit_pct, 4),
        },
        "throughput": {
            "steady_out_tok_s": round(steady_out_tok_s, 2),
            "window_output_tok_per_s": round(steady_out_tok_s, 2),
            "drain_out_tok_s": round(drain_out_tok_s, 2),
            "out_tok_s_per_decode_chip": round(out_per_dec, 2),
            "output_tok_per_s_per_decode_chip": round(out_per_dec, 2),
            "out_tok_s_per_total_chip": round(out_per_tot, 2),
            "output_tok_per_s_per_total_chip": round(out_per_tot, 2),
            "server_gen_tok_s_60_290": server_gen_tok_s_60_290,
            "steady_in_tok_s": round(steady_in_tok_s, 2),
            "steady_total_tok_s": round(steady_total_tok_s, 2),
            "gain_vs_prev_pct": round(gain_vs_prev_pct, 2) if gain_vs_prev_pct is not None else None,
        },
        "per_user_decode_tps": {
            "in_window": win_tps_dist,
            "all_requests": all_tps_dist,
        },
        "ttft_s": {
            "all_requests": all_ttft_dist,
            "in_window": win_ttft_dist,
        },
        "tpot_ms": {
            "all_requests": all_tpot_dist,
            "in_window": win_tpot_dist,
        },
        "dspark": {
            "mean_accept_len_tau": round(tau_full, 4),
            "mean_accept_len_tau_60_290": tau_60_290,
            "num_drafts": round(delta_drafts, 1),
            "num_draft_tokens": round(delta_draft_tokens, 1),
            "num_accepted_tokens": round(delta_accepted, 1),
            "per_pos_accept_rates": pos_rates,
        },
        "decode_batch_30_295": {
            "per_dp_rank_mean": round(statistics.mean(per_dp_samples), 2) if per_dp_samples else 0.0,
            "per_dp_rank_median": round(percentile(per_dp_samples, 50), 2) if per_dp_samples else 0.0,
            "per_dp_rank_max": round(max(per_dp_samples), 2) if per_dp_samples else 0.0,
            "total_running_mean": round(statistics.mean(tot_run_samples), 2) if tot_run_samples else 0.0,
            "total_running_max": round(max(tot_run_samples), 2) if tot_run_samples else 0.0,
            "total_waiting_mean": round(statistics.mean(tot_wait_samples), 2) if tot_wait_samples else 0.0,
            "total_waiting_max": round(max(tot_wait_samples), 2) if tot_wait_samples else 0.0,
            "prefill_running_mean": round(statistics.mean(pre_run_samples), 2) if pre_run_samples else 0.0,
            "prefill_running_max": round(max(pre_run_samples), 2) if pre_run_samples else 0.0,
        },
        "kv_cache_usage_pct": {
            "decode_steady_mean_pct": round(statistics.mean(dec_kv_means), 4) if dec_kv_means else 0.0,
            "decode_peak_single_rank_pct": round(dec_kv_peak_rank, 4),
            "prefill_steady_mean_pct": round(statistics.mean(pre_kv_means), 4) if pre_kv_means else 0.0,
            "prefill_peak_single_rank_pct": round(pre_kv_peak_rank, 4),
        },
        "preemptions": {
            "prefill": round(pre_preempt, 1),
            "decode": round(dec_preempt, 1),
        },
        "routing_distribution": {
            "prefill_pods": p_pod_dist,
            "decode_instances": d_inst_dist,
        },
        "stop_evaluation": {
            "should_stop": bool(len(stop_reasons) > 0),
            "min_concurrency_for_gain_stop": min_concurrency_for_gain_stop,
            "reasons": stop_reasons,
        },
    }

    lvl_json_path = out_dir / f"{level_tag}.json"
    lvl_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for r in results:
        r.pop("chunk_times", None)
    raw_req_path = raw_out_dir / f"{level_tag}_requests.json.gz"
    with gzip.open(raw_req_path, "wt", encoding="utf-8") as gz:
        json.dump(results, gz)

    print("-" * 108, flush=True)
    print(
        f"[DONE {level_tag}] Out={steady_out_tok_s:,.2f} tok/s "
        f"({out_per_dec:,.2f}/dec-chip, {out_per_tot:,.2f}/tot-chip, "
        f"gain={f'{gain_vs_prev_pct:+.2f}%' if gain_vs_prev_pct is not None else 'baseline'}) | "
        f"SrvGen(60-290s)={server_gen_tok_s_60_290} tok/s | "
        f"CacheHit={api_hit_pct:.2f}% (Prom={prom_hit_pct:.2f}%)",
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
    db = summary["decode_batch_30_295"]
    kv = summary["kv_cache_usage_pct"]
    print(
        f"  DSpark tau={tau_full:.4f} (60-290s={tau_60_290}) | "
        f"D_batch/DP={db['per_dp_rank_mean']:.2f}/{db['per_dp_rank_median']:.1f}/{db['per_dp_rank_max']:.1f} "
        f"(D_run={db['total_running_mean']:.1f}/{db['total_running_max']:.0f}, D_wait={db['total_waiting_mean']:.1f}, "
        f"P_run={db['prefill_running_mean']:.2f}) | "
        f"KV(D)={kv['decode_steady_mean_pct']:.3f}%/{kv['decode_peak_single_rank_pct']:.3f}% | "
        f"Preempt(P/D)={pre_preempt:.0f}/{dec_preempt:.0f} | Fail={len(err_results)}/{len(results)}",
        flush=True,
    )
    if stop_reasons:
        print(f"  [STOP RULE TRIGGERED] {stop_reasons}", flush=True)
    print("=" * 108, flush=True)
    return summary


async def main_async(args: argparse.Namespace) -> int:
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    prefill_urls = [x.strip() for x in args.prefill_urls.split(",") if x.strip()]
    decode_urls = [x.strip() for x in args.decode_metrics_urls.split(",") if x.strip()]
    target_url = f"{args.base_url.rstrip('/')}/v1/completions"
    out_dir = Path(args.out_dir)
    raw_out_dir = Path(args.raw_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_connect=30.0, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=2048, limit_per_host=2048, ttl_dns_cache=300, force_close=False)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        ok_h, issues = await check_all_healthy(http_session, prefill_urls, decode_urls)
        if not ok_h:
            raise RuntimeError(f"Initial R3 health check failed: {issues}")
        print(f"[R3 INIT] All {len(prefill_urls)} Prefill + {len(decode_urls)} Decode endpoints healthy.", flush=True)

        await reset_all_prefix_caches(http_session, prefill_urls, decode_urls)

        tok_base = prefill_urls[0]
        print(
            f"[R3 INIT] Calibrating {args.num_trunks} S1 ({args.prefix_tokens}:{args.osl_tokens}) trunks via {tok_base} ...",
            flush=True,
        )
        prefixes: list[str] = []
        trunk_tok_counts: list[int] = []
        for t_id in range(args.num_trunks):
            pfx, cnt = await calibrate_exact_2048_prefix(
                http_session, tok_base, args.model, t_id, args.prefix_tokens
            )
            prefixes.append(pfx)
            trunk_tok_counts.append(cnt)
        print(
            f"[R3 INIT] Calibrated {len(prefixes)} trunks: shared_prefix_tok_counts(min/max)="
            f"{min(trunk_tok_counts)}/{max(trunk_tok_counts)} (all >= {args.prefix_tokens + 2})",
            flush=True,
        )

        prewarm_info = await prewarm_all_prefill_ranks(
            http_session,
            prefill_urls=prefill_urls,
            router_base_url=args.base_url,
            model_name=args.model,
            prefixes=prefixes,
            prefill_dp_size=args.prefill_dp_size,
        )

        levels: list[dict[str, Any]] = []
        prev_out: float | None = None

        for c in concurrencies:
            s = await run_r3_level(
                http_session,
                concurrency_n=c,
                duration_s=args.duration_per_n,
                target_url=target_url,
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                model_name=args.model,
                prefixes=prefixes,
                osl_tokens=args.osl_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_decode_chips=args.num_decode_chips,
                num_total_chips=args.num_total_chips,
                out_dir=out_dir,
                raw_out_dir=raw_out_dir,
                prev_steady_out_tok_s=prev_out,
                min_concurrency_for_gain_stop=args.min_concurrency_for_gain_stop,
            )
            levels.append(s)
            prev_out = s["throughput"]["steady_out_tok_s"]

            peak = max(levels, key=lambda x: x["throughput"]["steady_out_tok_s"])
            out_doc = {
                "experiment": "R3_decode_ceiling_S1_2048_1024",
                "base_url": args.base_url,
                "prefill_urls": prefill_urls,
                "decode_metrics_urls": decode_urls,
                "shape": f"{args.prefix_tokens}:{args.osl_tokens}",
                "num_trunks": args.num_trunks,
                "trunk_shared_token_counts": trunk_tok_counts,
                "num_decode_chips": args.num_decode_chips,
                "num_total_chips": args.num_total_chips,
                "min_concurrency_for_gain_stop": args.min_concurrency_for_gain_stop,
                "prewarm": prewarm_info,
                "completed_concurrencies": [x["concurrency"] for x in levels],
                "stopped_early": s["stop_evaluation"]["should_stop"],
                "stop_reasons": s["stop_evaluation"]["reasons"],
                "peak_concurrency": peak["concurrency"],
                "peak_window_output_tok_per_s": peak["throughput"]["steady_out_tok_s"],
                "peak_output_tok_per_s_per_decode_chip": peak["throughput"]["out_tok_s_per_decode_chip"],
                "levels": levels,
            }
            sweep_json = out_dir / "r3_decode_ceiling_summary.json"
            sweep_json.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
            if args.output_json:
                out_p = Path(args.output_json)
                if out_p.resolve() != sweep_json.resolve():
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")

            if s["stop_evaluation"]["should_stop"]:
                print(
                    f"\n[R3 SWEEP COMPLETE] Stopping concurrency escalation after c={c}: "
                    f"{s['stop_evaluation']['reasons']}",
                    flush=True,
                )
                break

    print(f"\n[DONE] Wrote R3 Decode-Only Ceiling summary to {out_dir / 'r3_decode_ceiling_summary.json'}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 3 R3 Decode-Only Ceiling Benchmark Harness (S1 = 2048:1024)")
    ap.add_argument("--base-url", "--epp-url", dest="base_url", default="http://pd-disaggregation-epp.default.svc.cluster.local:80")
    ap.add_argument(
        "--prefill-urls",
        default="http://<private-ip>:8000,http://<private-ip>:8000",
        help="Comma-separated Prefill vLLM base URLs for multi-rank prewarming and /metrics",
    )
    ap.add_argument(
        "--decode-metrics-urls",
        "--decode-urls",
        dest="decode_metrics_urls",
        default="http://<private-ip>:9400,http://<private-ip>:9400",
        help="Comma-separated Decode vLLM base URLs for /metrics",
    )
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--num-trunks", type=int, default=16)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--osl-tokens", type=int, default=1024)
    ap.add_argument("--prefill-dp-size", type=int, default=8)
    ap.add_argument("--concurrencies", default="256,384,512,768,1024,1280")
    ap.add_argument("--min-concurrency-for-gain-stop", type=int, default=1024)
    ap.add_argument("--duration-per-n", "--steady-seconds", dest="duration_per_n", type=float, default=300.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-decode-chips", type=int, default=8)
    ap.add_argument("--num-total-chips", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out-dir", "--output-dir", dest="out_dir", default="/dev/shm/step3_runs/r3_decode_ceiling")
    ap.add_argument("--raw-out-dir", default="/dev/shm/step3_runs/r3_decode_ceiling_raw")
    ap.add_argument("--output-json", default="")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
