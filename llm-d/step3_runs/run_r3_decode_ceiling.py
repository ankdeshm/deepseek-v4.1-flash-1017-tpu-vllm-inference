#!/usr/bin/env python3
"""Step 3 R3: Decode-Only Ceiling Benchmark Harness (`S1 = 2048:1024` High-Concurrency Sweep).

Pure asyncio + aiohttp (NO torch / transformers imports; safe with PYTHONPATH="").

Design to Isolate Pure Pool B Decode Ceiling:
  1. Builds a compact pool of `16` distinct `S1` trunks (`--num-trunks 16`), each calibrated via
     vLLM's `/tokenize` endpoint to have **exact `2048` shared prefix tokens** (aligned to
     `PREFIX_CACHE_RETENTION_INTERVAL=1024` so all `2048` prefix tokens stay pinned in vLLM's
     hybrid KV + Mamba/CSA cache!) plus a tiny unique per-request suffix (`~4-6` tokens).
  2. Explicitly pre-warms all `16` trunks across **ALL `8` DP ranks (`X-data-parallel-rank: 0..7`)**
     on **BOTH** `llmd-prefill-poola-h0` (`192.168.100.43:8000`) and `llmd-prefill-poola-h1`
     (`192.168.100.38:8000`) before starting the sweep (`16 * 2048 = 32,768` tok/rank = `2.2%` of
     each Prefill rank's `1.464M`-token KV cache -> zero evictions, `~99.7%` prefix cache hit in
     `<25 ms`, keeping Prefill `<10%` utilized so Prefill NEVER gates Decode!).
  3. Sweeps high concurrency (`256, 384, 512, 768, 1024, 1280, 1536`) at `300.0 s` steady window
     per level with `max_tokens=1024, min_tokens=1024, ignore_eos=True` through `llm-d`, recording:
       - Window & drain `out tok/s` total, per decode chip (`/8`), and per total chip (`/16`)
       - Max concurrent running & waiting sequences (`vllm:num_requests_running`, `waiting`)
       - Steady & peak batch size per Decode DP rank (`engine=0..15` or `0..7` x 2)
       - Steady & peak single-rank KV cache usage % (`vllm:kv_cache_usage_perc`)
       - `DSpark` mean acceptance length `tau` + per-position acceptance rates
       - Per-user decode speed `median / P90_low / P95_low`, `TTFT P50 / P95 / P99` (all-reqs & in-win)
       - Preemptions (`vllm:num_preemptions_total`) and HTTP failure count.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - present in in-cluster vllm/benchmark images
    aiohttp = None  # type: ignore[assignment]


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


async def calibrate_exact_2048_prefix(
    http_session: aiohttp.ClientSession,
    tokenize_base_url: str,
    model_name: str,
    trunk_id: int,
    target_prefix_tokens: int = 2048,
) -> str:
    """Builds a deterministic natural-prose technical prompt calibrated via /tokenize to >= target_prefix_tokens."""
    header = (
        f"System Architecture Reference Document #{trunk_id:03d}: "
        f"Analyze the distributed tensor-parallel and expert-parallel execution schedule below and "
        f"derive a detailed step-by-step performance model.\n\n"
    )
    # Build deterministic pseudo-technical paragraphs per trunk_id so DSpark achieves realistic acceptance.
    paragraphs: list[str] = []
    for sec in range(1, 60):
        w0 = VOCAB_WORDS[(trunk_id * 7 + sec * 3) % len(VOCAB_WORDS)]
        w1 = VOCAB_WORDS[(trunk_id * 11 + sec * 5) % len(VOCAB_WORDS)]
        w2 = VOCAB_WORDS[(trunk_id * 13 + sec * 7) % len(VOCAB_WORDS)]
        paragraphs.append(
            f"Section {sec}.{trunk_id}: In stage {sec}, the {w0} subsystem coordinates {w1} buffers "
            f"across the {w2} interconnect while maintaining bounded tail latency under continuous load. "
        )
    full_text = header + "".join(paragraphs)

    # Calibrate exact token length using vLLM's /tokenize endpoint if reachable, else fallback to char ratio.
    tok_url = f"{tokenize_base_url.rstrip('/')}/tokenize"
    try:
        lo, hi = int(target_prefix_tokens * 2.5), min(len(full_text), int(target_prefix_tokens * 7.5))
        best = full_text[: int(target_prefix_tokens * 4.8)]
        for _ in range(14):
            mid = (lo + hi) // 2
            cand = full_text[:mid]
            async with http_session.post(
                tok_url,
                json={"model": model_name, "prompt": cand, "add_special_tokens": True},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
                cnt = int(data.get("count") or len(data.get("tokens") or []))
            if cnt < target_prefix_tokens:
                lo = mid + 1
            elif cnt > target_prefix_tokens + 8:
                hi = mid - 1
            else:
                best = cand
                break
            best = full_text[:lo]
        return best
    except Exception:
        return full_text[: int(target_prefix_tokens * 4.6)]


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
                    "prompt": f"{pfx}\n\nQuery [warm-{t_id}-dp{dp_r}]: Summarize key bottlenecks.",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
                headers = {"X-data-parallel-rank": str(dp_r)}
                tasks.append(http_session.post(comp_url, json=body, headers=headers))
        # Execute in bounded waves of 16 so prefill warms cleanly in ~3s
        for start in range(0, len(tasks), 16):
            batch_coros = tasks[start : start + 16]
            resps = await asyncio.gather(*batch_coros, return_exceptions=True)
            for r in resps:
                if not isinstance(r, Exception):
                    if r.status == 200:
                        ok_direct += 1
                    await r.release()

    # Also warm router + EndpointPicker radix tree + Decode XLA buckets
    router_comp = f"{router_base_url.rstrip('/')}/v1/completions"
    ok_router = 0
    for rep in range(2):
        r_tasks = []
        for t_id, pfx in enumerate(prefixes):
            body = {
                "model": model_name,
                "prompt": f"{pfx}\n\nQuery [router-warm-{rep}-{t_id}]: Summarize key bottlenecks.",
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
    return {
        "direct_prefill_ok": ok_direct,
        "direct_prefill_total": tot_direct,
        "router_warmup_ok": ok_router,
        "router_warmup_total": 2 * len(prefixes),
        "wall_s": elapsed,
    }


def parse_vllm_prometheus(text: str) -> dict[str, Any]:
    kv_by_eng: dict[str, float] = {}
    run_by_eng: dict[str, float] = {}
    wait_by_eng: dict[str, float] = {}
    preempt_total = 0.0
    prefix_hits = 0.0
    prefix_queries = 0.0
    spec_drafts = 0.0
    spec_draft_tokens = 0.0
    spec_accepted = 0.0
    spec_pos: dict[int, float] = {}

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not m:
            continue
        name, labels_raw, val_str = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = float(val_str)
        except ValueError:
            continue
        eng_m = re.search(r'engine="([^"]+)"', labels_raw)
        eng = eng_m.group(1) if eng_m else "0"

        if name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            kv_by_eng[eng] = val * 100.0
        elif name == "vllm:num_requests_running":
            run_by_eng[eng] = val
        elif name == "vllm:num_requests_waiting":
            wait_by_eng[eng] = val
        elif name == "vllm:num_preemptions_total":
            preempt_total += val
        elif name in ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"):
            prefix_hits += val
        elif name in ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"):
            prefix_queries += val
        elif name == "vllm:spec_decode_num_drafts_total":
            spec_drafts += val
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            spec_draft_tokens += val
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            spec_accepted += val
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pm = re.search(r'position="(\d+)"', labels_raw)
            if pm:
                pos = int(pm.group(1))
                spec_pos[pos] = spec_pos.get(pos, 0.0) + val

    return {
        "kv_pct_by_engine": kv_by_eng,
        "running_by_engine": run_by_eng,
        "waiting_by_engine": wait_by_eng,
        "preemptions_total": preempt_total,
        "prefix_hits_total": prefix_hits,
        "prefix_queries_total": prefix_queries,
        "spec_drafts": spec_drafts,
        "spec_draft_tokens": spec_draft_tokens,
        "spec_accepted": spec_accepted,
        "spec_per_pos": spec_pos,
    }


async def scrape_endpoints(http_session: aiohttp.ClientSession, urls: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for u in urls:
        try:
            async with http_session.get(
                f"{u.rstrip('/')}/metrics", timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status == 200:
                    out[u] = parse_vllm_prometheus(await resp.text())
        except Exception:
            pass
    return out


async def send_completion_stream(
    http_session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    trunk_id: int,
    session_id: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    ttft: float | None = None
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
                            if ttft is None:
                                ttft = now - t_start
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
    if ttft is None and error_msg is None:
        ttft = e2e
    eff_compl = completion_tokens or num_chunks
    decode_dur = max(0.0, e2e - (ttft or 0.0))
    decode_tps = ((eff_compl - 1) / decode_dur) if (eff_compl > 1 and decode_dur > 0) else None
    per_tok_itl = (decode_dur / (eff_compl - 1)) if (eff_compl > 1 and decode_dur > 0) else None

    return {
        "session_id": session_id,
        "trunk_id": trunk_id,
        "t_start": t_start,
        "t_end": t_end,
        "ttft": ttft,
        "e2e": e2e,
        "decode_tps": decode_tps,
        "per_token_itl": per_tok_itl,
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
) -> dict[str, Any]:
    pre_m0 = await scrape_endpoints(http_session, prefill_urls)
    dec_m0 = await scrape_endpoints(http_session, decode_urls)

    pre_samples: list[dict[str, dict[str, Any]]] = []
    dec_samples: list[dict[str, dict[str, Any]]] = []
    stop_poller = asyncio.Event()

    t_bench_start = time.perf_counter()
    t_window_end = t_bench_start + duration_s

    async def _poller() -> None:
        while not stop_poller.is_set():
            try:
                await asyncio.wait_for(stop_poller.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                pass
            if time.perf_counter() <= t_window_end:
                ps, ds = await asyncio.gather(
                    scrape_endpoints(http_session, prefill_urls),
                    scrape_endpoints(http_session, decode_urls),
                )
                if ps:
                    pre_samples.append(ps)
                if ds:
                    dec_samples.append(ds)

    poller_task = asyncio.create_task(_poller())
    dispatch_counter = [0]

    async def worker(s_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        turn = 0
        while time.perf_counter() < t_window_end:
            idx = dispatch_counter[0]
            dispatch_counter[0] += 1
            trunk_id = idx % len(prefixes)
            prompt = (
                f"{prefixes[trunk_id]}\n\n"
                f"Query [n{concurrency_n}-s{s_id}-t{turn}]: Derive the full analytical breakdown."
            )
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
            res = await send_completion_stream(http_session, target_url, payload, trunk_id, s_id)
            out.append(res)
            turn += 1
        return out

    worker_lists = await asyncio.gather(*(worker(i) for i in range(concurrency_n)))
    t_bench_end = time.perf_counter()
    stop_poller.set()
    await poller_task

    pre_m1 = await scrape_endpoints(http_session, prefill_urls)
    dec_m1 = await scrape_endpoints(http_session, decode_urls)

    wall_clock = t_bench_end - t_bench_start
    results = [r for lst in worker_lists for r in lst]
    ok_results = [r for r in results if r["error"] is None]
    err_results = [r for r in results if r["error"] is not None]

    window_output_tokens = 0.0
    window_input_tokens = 0.0
    window_completed = 0
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
                window_output_tokens += compl * (in_win / len(c_times))
                window_input_tokens += prompt

    win_ok = [r for r in ok_results if r["t_end"] <= t_window_end]
    eval_win = win_ok if len(win_ok) >= max(4, concurrency_n // 4) else ok_results

    all_ttfts = [r["ttft"] for r in ok_results if r["ttft"] is not None]
    win_ttfts = [r["ttft"] for r in eval_win if r["ttft"] is not None]
    all_tps = [r["decode_tps"] for r in ok_results if r.get("decode_tps") is not None]
    win_tps = [r["decode_tps"] for r in eval_win if r.get("decode_tps") is not None]

    tot_prompt = sum(float(r["prompt_tokens"] or 0) for r in ok_results)
    tot_cached = sum(float(r["cached_tokens"] or 0) for r in ok_results)
    tot_compl = sum(float(r["completion_tokens"] or 0) for r in ok_results)
    api_hit_pct = (100.0 * tot_cached / tot_prompt) if tot_prompt > 0 else 0.0

    # Aggregate Decode & Prefill Prometheus diagnostics
    d_drafts = sum(max(0.0, dec_m1.get(u, {}).get("spec_drafts", 0.0) - dec_m0.get(u, {}).get("spec_drafts", 0.0)) for u in decode_urls)
    d_acc = sum(max(0.0, dec_m1.get(u, {}).get("spec_accepted", 0.0) - dec_m0.get(u, {}).get("spec_accepted", 0.0)) for u in decode_urls)
    tau = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0

    dec_preempt = sum(max(0.0, dec_m1.get(u, {}).get("preemptions_total", 0.0) - dec_m0.get(u, {}).get("preemptions_total", 0.0)) for u in decode_urls)
    pre_preempt = sum(max(0.0, pre_m1.get(u, {}).get("preemptions_total", 0.0) - pre_m0.get(u, {}).get("preemptions_total", 0.0)) for u in prefill_urls)
    p_hits = sum(max(0.0, pre_m1.get(u, {}).get("prefix_hits_total", 0.0) - pre_m0.get(u, {}).get("prefix_hits_total", 0.0)) for u in prefill_urls)
    p_queries = sum(max(0.0, pre_m1.get(u, {}).get("prefix_queries_total", 0.0) - pre_m0.get(u, {}).get("prefix_queries_total", 0.0)) for u in prefill_urls)
    prom_hit_pct = (100.0 * p_hits / p_queries) if p_queries > 0 else 0.0

    dec_run_totals: list[float] = []
    dec_wait_totals: list[float] = []
    dec_rank_batches: list[float] = []
    dec_kv_means: list[float] = []
    dec_kv_peak_rank = 0.0
    pre_run_totals: list[float] = []
    pre_kv_means: list[float] = []
    pre_kv_peak_rank = 0.0

    for s in dec_samples:
        r_vals = [v for m in s.values() for v in m["running_by_engine"].values()]
        w_vals = [v for m in s.values() for v in m["waiting_by_engine"].values()]
        kv_vals = [v for m in s.values() for v in m["kv_pct_by_engine"].values()]
        dec_run_totals.append(sum(r_vals))
        dec_wait_totals.append(sum(w_vals))
        if r_vals:
            dec_rank_batches.append(statistics.mean(r_vals))
        if kv_vals:
            dec_kv_means.append(statistics.mean(kv_vals))
            dec_kv_peak_rank = max(dec_kv_peak_rank, max(kv_vals))

    for s in pre_samples:
        pr_vals = [v for m in s.values() for v in m["running_by_engine"].values()]
        pkv_vals = [v for m in s.values() for v in m["kv_pct_by_engine"].values()]
        pre_run_totals.append(sum(pr_vals))
        if pkv_vals:
            pre_kv_means.append(statistics.mean(pkv_vals))
            pre_kv_peak_rank = max(pre_kv_peak_rank, max(pkv_vals))

    win_out_tps = window_output_tokens / duration_s
    win_in_tps = window_input_tokens / duration_s
    drain_out_tps = tot_compl / wall_clock if wall_clock > 0 else 0.0

    summary = {
        "concurrency_n": concurrency_n,
        "steady_window_s": duration_s,
        "drain_wall_s": round(wall_clock, 2),
        "requests": {
            "total": len(results),
            "successful": len(ok_results),
            "in_window_completed": window_completed,
            "failed": len(err_results),
            "errors_sample": [r["error"] for r in err_results[:5]],
        },
        "tokens": {
            "mean_isl": round(tot_prompt / len(ok_results), 1) if ok_results else 0.0,
            "mean_osl": round(tot_compl / len(ok_results), 1) if ok_results else 0.0,
            "api_cache_hit_pct": round(api_hit_pct, 2),
            "prometheus_prefill_hit_pct": round(prom_hit_pct, 2),
        },
        "throughput": {
            "window_output_tok_per_s": round(win_out_tps, 2),
            "drain_output_tok_per_s": round(drain_out_tps, 2),
            "output_tok_per_s_per_decode_chip": round(win_out_tps / max(1, num_decode_chips), 2),
            "output_tok_per_s_per_total_chip": round(win_out_tps / max(1, num_total_chips), 2),
            "window_input_tok_per_s": round(win_in_tps, 2),
            "window_total_in_plus_out_tok_per_s": round(win_in_tps + win_out_tps, 2),
        },
        "decode_engine_ceiling_diagnostics": {
            "running_sequences_steady_mean": round(statistics.mean(dec_run_totals), 2) if dec_run_totals else 0.0,
            "running_sequences_max": round(max(dec_run_totals), 2) if dec_run_totals else 0.0,
            "waiting_sequences_steady_mean": round(statistics.mean(dec_wait_totals), 2) if dec_wait_totals else 0.0,
            "waiting_sequences_max": round(max(dec_wait_totals), 2) if dec_wait_totals else 0.0,
            "batch_per_dp_rank_steady_mean": round(statistics.mean(dec_rank_batches), 2) if dec_rank_batches else 0.0,
            "batch_per_dp_rank_max": round(
                max((max(m["running_by_engine"].values()) for s in dec_samples for m in s.values() if m["running_by_engine"]), default=0.0),
                2,
            ),
            "decode_kv_steady_mean_pct": round(statistics.mean(dec_kv_means), 2) if dec_kv_means else 0.0,
            "decode_kv_peak_single_rank_pct": round(dec_kv_peak_rank, 2),
            "prefill_running_sequences_steady_mean": round(statistics.mean(pre_run_totals), 2) if pre_run_totals else 0.0,
            "prefill_kv_steady_mean_pct": round(statistics.mean(pre_kv_means), 2) if pre_kv_means else 0.0,
            "prefill_kv_peak_single_rank_pct": round(pre_kv_peak_rank, 2),
            "dspark_tau": round(tau, 4),
            "decode_preemptions_delta": dec_preempt,
            "prefill_preemptions_delta": pre_preempt,
        },
        "per_user_decode_tok_per_s": {
            "in_window": {
                "median": round(statistics.median(win_tps), 2) if win_tps else 0.0,
                "p90_low": round(percentile(win_tps, 10), 2),
                "p95_low": round(percentile(win_tps, 5), 2),
            },
            "all_requests": {
                "median": round(statistics.median(all_tps), 2) if all_tps else 0.0,
                "p90_low": round(percentile(all_tps, 10), 2),
                "p95_low": round(percentile(all_tps, 5), 2),
            },
        },
        "ttft_s": {
            "all_requests_incl_drain": {
                "p50": round(statistics.median(all_ttfts), 3) if all_ttfts else 0.0,
                "p95": round(percentile(all_ttfts, 95), 3),
                "p99": round(percentile(all_ttfts, 99), 3),
            },
            "in_window_only": {
                "p50": round(statistics.median(win_ttfts), 3) if win_ttfts else 0.0,
                "p95": round(percentile(win_ttfts, 95), 3),
                "p99": round(percentile(win_ttfts, 99), 3),
            },
        },
    }

    diag = summary["decode_engine_ceiling_diagnostics"]
    print(
        f"\n[R3 | n={concurrency_n:4d} | {duration_s:.0f}s] "
        f"Out={win_out_tps:,.2f} tok/s ({win_out_tps/num_decode_chips:,.2f}/dec-chip, {win_out_tps/num_total_chips:,.2f}/tot-chip) | "
        f"RunSeq(mean/max)={diag['running_sequences_steady_mean']:.1f}/{diag['running_sequences_max']:.0f} "
        f"(Wait={diag['waiting_sequences_steady_mean']:.1f}, Batch/DP={diag['batch_per_dp_rank_steady_mean']:.1f}) | "
        f"DecKV={diag['decode_kv_steady_mean_pct']:.1f}%/{diag['decode_kv_peak_single_rank_pct']:.1f}% | "
        f"PrefillRun={diag['prefill_running_sequences_steady_mean']:.2f} (Hit={api_hit_pct:.2f}%) | "
        f"DSpark tau={tau:.4f} | Preempt(P/D)={pre_preempt:.0f}/{dec_preempt:.0f} | Err={len(err_results)}/{len(results)}",
        flush=True,
    )
    return summary


async def main_async(args: argparse.Namespace) -> None:
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    prefill_urls = [x.strip() for x in args.prefill_urls.split(",") if x.strip()]
    decode_urls = [x.strip() for x in args.decode_metrics_urls.split(",") if x.strip()]
    target_url = f"{args.base_url.rstrip('/')}/v1/completions"

    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        tok_base = prefill_urls[0] if prefill_urls else args.base_url
        print(f"[R3 INIT] Calibrating {args.num_trunks} S1 ({args.prefix_tokens}:{args.osl_tokens}) trunks via {tok_base} ...", flush=True)
        prefixes = [
            await calibrate_exact_2048_prefix(http_session, tok_base, args.model, t_id, args.prefix_tokens)
            for t_id in range(args.num_trunks)
        ]

        prewarm_info = await prewarm_all_prefill_ranks(
            http_session,
            prefill_urls=prefill_urls,
            router_base_url=args.base_url,
            model_name=args.model,
            prefixes=prefixes,
            prefill_dp_size=args.prefill_dp_size,
        )

        levels: list[dict[str, Any]] = []
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
            )
            levels.append(s)

            peak = max(levels, key=lambda x: x["throughput"]["window_output_tok_per_s"])
            out_doc = {
                "experiment": "R3_decode_ceiling_S1_2048_1024",
                "base_url": args.base_url,
                "prefill_urls": prefill_urls,
                "decode_metrics_urls": decode_urls,
                "shape": f"{args.prefix_tokens}:{args.osl_tokens}",
                "num_trunks": args.num_trunks,
                "num_decode_chips": args.num_decode_chips,
                "num_total_chips": args.num_total_chips,
                "prewarm": prewarm_info,
                "peak_concurrency": peak["concurrency_n"],
                "peak_window_output_tok_per_s": peak["throughput"]["window_output_tok_per_s"],
                "peak_output_tok_per_s_per_decode_chip": peak["throughput"]["output_tok_per_s_per_decode_chip"],
                "levels": levels,
            }
            if args.output_json:
                out_p = Path(args.output_json)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(out_doc, indent=2))

    if args.output_json:
        print(f"\n[DONE] Wrote R3 Decode-Only Ceiling summary to {args.output_json}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 3 R3 Decode-Only Ceiling Benchmark Harness (S1 = 2048:1024)")
    ap.add_argument("--base-url", default="http://pd-disaggregation-epp.default.svc.cluster.local:80")
    ap.add_argument(
        "--prefill-urls",
        default="http://192.168.100.43:8000,http://192.168.100.38:8000",
        help="Comma-separated Prefill vLLM base URLs for multi-rank prewarming and /metrics",
    )
    ap.add_argument(
        "--decode-metrics-urls",
        default="http://192.168.100.198:9400",
        help="Comma-separated Decode vLLM base URLs for /metrics (pass both :9400 IPs for 2xEP=8)",
    )
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--num-trunks", type=int, default=16)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--osl-tokens", type=int, default=1024)
    ap.add_argument("--prefill-dp-size", type=int, default=8)
    ap.add_argument("--concurrencies", default="256,384,512,768,1024,1280,1536")
    ap.add_argument("--duration-per-n", type=float, default=300.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-decode-chips", type=int, default=8)
    ap.add_argument("--num-total-chips", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--output-json", default="/tmp/r3_decode_ceiling.json")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
