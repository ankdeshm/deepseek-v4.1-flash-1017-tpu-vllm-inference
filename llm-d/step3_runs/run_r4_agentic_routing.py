#!/usr/bin/env python3
"""Step 3 R4 (and R6): llm-d Cache-Aware Routing vs Round-Robin Multi-Turn Agentic Benchmark.

Pure asyncio + aiohttp (NO torch / transformers imports; safe with PYTHONPATH="").

Default R4 Specification:
  - Dataset: `/data/requests_mocked.jsonl.xz` (empirical 50:20:12:9:9 ISL distribution,
    exact per-row `(temperature, top_p, top_k)` sampling specs, structured multi-turn + tools).
  - Trunks: `256` distinct trunks total (`128` per prefill instance across 2 Pool A prefill pods),
    partitioned in exact `50:20:12:9:9` ratio (`1-8K: 128, 8-16K: 51, 16-32K: 31, 32-64K: 23, 64K+: 23`).
  - Target cache hit ratio: `0.80` (`--cache-hit-rate 0.80`; pass `0.95` for R6).
  - Natural stop: `ignore_eos=False` (`--no-ignore-eos`).
  - Concurrency sweep: `n = 64, 128, 256` at `600.0 s` steady window per level (`--duration-per-n 600`),
    with **NO cache reset between concurrency levels** (caches are flushed ONLY ONCE at the start of
    each routing condition `(a)` round-robin or `(b)` prefix-aware, followed by router trunk warmup,
    and then `n=64 -> n=128 -> n=256` run continuously against the live warm cache).

Also supports R6 (`--cache-hit-rate 0.95 --duration-per-n 300 --sessions 128,256 --num-trunks 80`).

Captures all Rule 4 metrics:
  - Realized cache hit % from BOTH `usage.prompt_tokens_details.cached_tokens / prompt_tokens` AND
    Prometheus `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` across `h0` + `h1`
    (plus per-pod prefill request & token distribution).
  - TTFT `P50 / P95 / P99` over ALL completed requests (including drain) AND in-window separately.
  - Output tok/s total (window & drain), per decode chip (`/8`), per total chip (`/16`), total in+out tok/s.
  - Per-user stream decode tok/s `median / P90_low / P95_low` (in-window & all-requests).
  - `DSpark` mean acceptance length `tau` + per-position acceptance rates.
  - Steady & peak-rank KV cache usage % (`vllm:kv_cache_usage_perc`), running/waiting batch per DP rank,
    and preemption deltas (`vllm:num_preemptions_total`) on Prefill and Decode.
  - Measured mean/P50/P95 ISL & OSL, tool-call count, and failure count.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import lzma
from pathlib import Path
import re
import statistics
import time
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - present in in-cluster vllm/benchmark images
    aiohttp = None  # type: ignore[assignment]


CHARS_PER_TOKEN = 3.85

# Optional client-side DP-rank affinity (same mechanism as ref_stack/toy_proxy_server.py used by R0 and
# the prior 7,446 / 14,542 runs): X-data-parallel-rank = trunk_id % DP_PIN_RANKS on every request.
# 0 = disabled (vLLM's internal DP load balancer picks the rank inside each prefill/decode instance).
DP_PIN_RANKS = 0
DP_PIN_MAP: dict[int, int] = {}


def build_balanced_dp_map(all_trunks: list[dict[str, Any]], trunks_by_bucket: dict[int, list[dict[str, Any]]], n_ranks: int) -> dict[int, int]:
    """Greedy LPT trunk->rank map weighted by expected token load (bucket share / trunks-in-bucket x ISL)."""
    loads = [0.0] * n_ranks
    weighted = []
    for t in all_trunks:
        b = t["bucket_idx"]
        share = ISL_BUCKETS[b][1] / max(1, len(trunks_by_bucket.get(b, [])))
        weighted.append((share * float(t["target_isl_tokens"]), int(t["trunk_id"])))
    out: dict[int, int] = {}
    for w, tid in sorted(weighted, reverse=True):
        r = min(range(n_ranks), key=lambda i: loads[i])
        out[tid] = r
        loads[r] += w
    return out

ISL_BUCKETS = [
    ("1-8K", 0.50, 5376, 8064),
    ("8-16K", 0.20, 8320, 15872),
    ("16-32K", 0.12, 16384, 31744),
    ("32-64K", 0.09, 32768, 63488),
    ("64K+", 0.09, 65536, 92160),
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


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(json.dumps(p, ensure_ascii=False))
        return "\n".join(parts)
    return str(content or "")


def _msg_char_len(m: dict[str, Any]) -> int:
    c = len(_extract_text(m.get("content"))) + len(str(m.get("reasoning_content") or ""))
    if m.get("tool_calls"):
        c += len(json.dumps(m["tool_calls"], ensure_ascii=False))
    return max(16, c + 16)


def resolve_dataset_path(path: str) -> Path:
    candidates = [
        Path(path),
        Path("/data/requests_mocked.jsonl.xz"),
        Path("/tmp/requests_mocked.jsonl"),
        Path("/models/dsv41-stack/requests_mocked.jsonl"),
        Path(__file__).resolve().parents[2] / "data" / "requests_mocked.jsonl.xz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate dataset at {path} or fallback paths {[str(x) for x in candidates]}")


def load_dataset(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = resolve_dataset_path(path)
    if p.suffix == ".xz":
        with lzma.open(p, "rt", encoding="utf-8") as f:
            raw_lines = [ln.strip() for ln in f if ln.strip()]
    else:
        with open(p, "r", encoding="utf-8") as f:
            raw_lines = [ln.strip() for ln in f if ln.strip()]

    raw_reqs: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for line in raw_lines:
        obj = json.loads(line)
        r = obj.get("body", obj)
        if isinstance(r, str):
            r = json.loads(r)
        raw_reqs.append(r)
        sp: dict[str, Any] = {}
        if "temperature" in r and r["temperature"] is not None:
            sp["temperature"] = r["temperature"]
        if "top_p" in r and r["top_p"] is not None:
            sp["top_p"] = r["top_p"]
        if "top_k" in r and r["top_k"] is not None:
            sp["top_k"] = r["top_k"]
        specs.append(sp)

    tool_bounded = [r for r in raw_reqs if r.get("tools") and int(r.get("max_tokens") or 0) <= 32768]
    if tool_bounded:
        templates = tool_bounded
    else:
        tool_any = [r for r in raw_reqs if r.get("tools")]
        templates = tool_any if tool_any else (raw_reqs[1:] if len(raw_reqs) > 1 else raw_reqs)
    return templates, specs


def _bucket_target_tokens(b_idx: int) -> int:
    _, _, lo_tok, hi_tok = ISL_BUCKETS[b_idx]
    return (((lo_tok + hi_tok) // 2) // 128) * 128


def allocate_bucket_counts(total_trunks: int) -> list[int]:
    n_b = len(ISL_BUCKETS)
    assert total_trunks >= n_b, f"total_trunks={total_trunks} must be >= {n_b}"
    if total_trunks == n_b:
        return [1] * n_b
    raw = [total_trunks * w for _, w, _, _ in ISL_BUCKETS]
    floors = [max(1, int(x)) for x in raw]
    while sum(floors) > total_trunks:
        idx = max(
            (i for i in range(n_b) if floors[i] > 1),
            key=lambda i: (floors[i] - raw[i], -ISL_BUCKETS[i][1]),
        )
        floors[idx] -= 1
    leftover = total_trunks - sum(floors)
    order = sorted(
        range(n_b),
        key=lambda i: (raw[i] - floors[i], ISL_BUCKETS[i][1]),
        reverse=True,
    )
    for i in order[:leftover]:
        floors[i] += 1
    return floors


def build_global_trunk_pool(num_trunks: int) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Builds `num_trunks` (e.g. 256) distinct trunks partitioned in exact 50:20:12:9:9 proportion."""
    target_counts = allocate_bucket_counts(num_trunks)
    assigned = [0] * len(ISL_BUCKETS)
    all_trunks: list[dict[str, Any]] = []
    by_bucket: dict[int, list[dict[str, Any]]] = {b: [] for b in range(len(ISL_BUCKETS))}

    for trunk_id in range(num_trunks):
        candidates = [b for b in range(len(ISL_BUCKETS)) if assigned[b] < target_counts[b]]
        b_chosen = max(
            candidates,
            key=lambda b: ((trunk_id + 1) * ISL_BUCKETS[b][1] - assigned[b], ISL_BUCKETS[b][1]),
        )
        assigned[b_chosen] += 1
        sp = {
            "trunk_id": trunk_id,
            "bucket_idx": b_chosen,
            "bucket": ISL_BUCKETS[b_chosen][0],
            "target_isl_tokens": _bucket_target_tokens(b_chosen),
        }
        all_trunks.append(sp)
        by_bucket[b_chosen].append(sp)
    return all_trunks, by_bucket


def build_trunk_chat_payload(
    orig_req: dict[str, Any],
    *,
    model_name: str,
    trunk_namespace: str,
    session_id: int,
    trunk_id: int,
    bucket_idx: int,
    target_isl_tokens: int,
    cache_hit_ratio: float,
    uid: str,
    max_tokens: int,
    bucket_orig_max_tokens: int,
    ignore_eos: bool,
    is_warmup: bool = False,
    sampling_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds standard OpenAI /v1/chat/completions payload (no non-standard fields like dp_rank/rid)."""
    eff_target_tok = 7040 if (cache_hit_ratio >= 0.90 and bucket_idx == 0) else target_isl_tokens
    total_chars = max(1024, int(eff_target_tok * CHARS_PER_TOKEN))
    # Account for 1024-token hybrid KV retention block quantization:
    quant_offset = 550.0 if cache_hit_ratio >= 0.90 else 512.0
    max_eff = 0.985 if cache_hit_ratio >= 0.90 else 0.92
    eff_ratio = min(max_eff, (eff_target_tok * cache_hit_ratio + quant_offset) / max(1, eff_target_tok))
    prefix_budget = int(total_chars * eff_ratio)
    suffix_budget = max(256, total_chars - prefix_budget)

    trunk_tag = f"[trunk_key={trunk_namespace}_k{trunk_id:04d}_b{bucket_idx}] "
    raw_tools = copy.deepcopy(orig_req.get("tools") or [])

    max_tools_chars = max(0, prefix_budget - 1024)
    kept_tools: list[dict[str, Any]] = []
    tools_chars = 0
    for t in raw_tools:
        tc_len = len(json.dumps(t, ensure_ascii=False)) + 2
        if tools_chars + tc_len <= max_tools_chars or (not kept_tools and tc_len <= max_tools_chars):
            kept_tools.append(t)
            tools_chars += tc_len
        else:
            break

    orig_msgs = copy.deepcopy(orig_req.get("messages") or [])
    if kept_tools:
        if "function" in kept_tools[0] and isinstance(kept_tools[0]["function"], dict):
            desc = str(kept_tools[0]["function"].get("description") or "")
            kept_tools[0]["function"]["description"] = trunk_tag + desc
    if orig_msgs and isinstance(orig_msgs[0].get("content"), str):
        orig_msgs[0]["content"] = trunk_tag + orig_msgs[0]["content"]
    elif not kept_tools:
        orig_msgs.insert(0, {"role": "system", "content": trunk_tag})

    rem_prefix = max(256, prefix_budget - tools_chars)
    prefix_messages: list[dict[str, Any]] = []
    cur_p = 0
    rep = 0
    while cur_p < rem_prefix:
        for m in orig_msgs:
            m_copy = copy.deepcopy(m)
            if isinstance(m_copy.get("content"), list):
                m_copy["content"] = _extract_text(m_copy["content"])
            mlen = _msg_char_len(m_copy)
            if cur_p + mlen <= rem_prefix:
                prefix_messages.append(m_copy)
                cur_p += mlen
            else:
                need = rem_prefix - cur_p
                if need >= 32:
                    txt = _extract_text(m_copy.get("content"))
                    if len(txt) < need:
                        pad_unit = txt + "\n" + json.dumps(m_copy, ensure_ascii=False)
                        txt = pad_unit * ((need // max(1, len(pad_unit))) + 2)
                    m_slice: dict[str, Any] = {
                        "role": m_copy.get("role", "user"),
                        "content": txt[:need],
                    }
                    if m_slice["role"] == "tool" and m_copy.get("tool_call_id"):
                        m_slice["tool_call_id"] = m_copy["tool_call_id"]
                    prefix_messages.append(m_slice)
                    cur_p += need
                cur_p = rem_prefix
                break
        rep += 1
        if rep > 50:
            break

    turn_tag = f"[turn_uid=s{session_id:04d}_k{trunk_id:04d}_{uid}]\n"
    tail_texts: list[str] = []
    cur_s = len(turn_tag)
    while cur_s < suffix_budget:
        for m in reversed(orig_msgs):
            txt = _extract_text(m.get("content"))
            if txt.strip():
                tail_texts.append(txt)
                cur_s += len(txt) + 2
                if cur_s >= suffix_budget:
                    break
        if not tail_texts:
            tail_texts.append("Continue the analysis and call the appropriate tool if needed.")
            break
    suffix_str = (turn_tag + "\n".join(tail_texts))[:suffix_budget]
    suffix_messages = [{"role": "user", "content": suffix_str}]

    if max_tokens > 0:
        req_max_tokens = max_tokens
    elif target_isl_tokens < 8192:
        req_max_tokens = 1024
    else:
        req_max_tokens = bucket_orig_max_tokens
    req_ignore_eos = ignore_eos
    if is_warmup:
        req_max_tokens = 1
        req_ignore_eos = False

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": prefix_messages + suffix_messages,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "high",
            "clear_thinking": False,
        },
        "max_tokens": req_max_tokens,
        "ignore_eos": req_ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if is_warmup:
        payload["temperature"] = 0.0
    elif sampling_spec is not None:
        if "temperature" in sampling_spec:
            payload["temperature"] = sampling_spec["temperature"]
        if "top_p" in sampling_spec:
            payload["top_p"] = sampling_spec["top_p"]
        if "top_k" in sampling_spec:
            payload["top_k"] = sampling_spec["top_k"]
    else:
        if "temperature" in orig_req and orig_req["temperature"] is not None:
            payload["temperature"] = orig_req["temperature"]
        if "top_p" in orig_req and orig_req["top_p"] is not None:
            payload["top_p"] = orig_req["top_p"]
    if kept_tools:
        payload["tools"] = kept_tools
        if orig_req.get("tool_choice") is not None:
            payload["tool_choice"] = orig_req["tool_choice"]
    return payload


async def send_chat_request(
    http_session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    meta: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    ttft: float | None = None
    chunk_times: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    status_code = 0
    error_msg: str | None = None
    num_content_chunks = 0
    finish_reason: str | None = None
    has_tool_calls = False

    # Hyphen-only headers (Envoy rejects headers containing underscores).
    headers = {
        "X-Session-ID": f"trunk-{meta.get('trunk_id', 0)}",
    }
    if DP_PIN_RANKS > 0:
        tid = int(meta.get("trunk_id", 0))
        headers["X-data-parallel-rank"] = str(DP_PIN_MAP.get(tid, tid % DP_PIN_RANKS))
    if extra_headers:
        headers.update(extra_headers)

    try:
        async with http_session.post(url, json=payload, headers=headers) as resp:
            status_code = resp.status
            if status_code != 200:
                err_body = await resp.text()
                error_msg = f"HTTP {status_code}: {err_body[:350]}"
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
                        delta = c0.get("delta", {})
                        text_val = c0.get("text", "")
                        if delta.get("tool_calls"):
                            has_tool_calls = True
                        has_token = bool(
                            delta.get("content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or delta.get("tool_calls")
                            or text_val
                        )
                        if has_token:
                            if ttft is None:
                                ttft = now - t_start
                            chunk_times.append(now)
                            num_content_chunks += 1

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

    eff_compl = completion_tokens or num_content_chunks
    decode_dur = max(0.0, e2e - (ttft or 0.0))
    per_token_itl: float | None = None
    decode_tps: float | None = None
    if eff_compl > 1 and decode_dur > 0:
        per_token_itl = decode_dur / (eff_compl - 1)
        decode_tps = (eff_compl - 1) / decode_dur

    return {
        **meta,
        "t_start": t_start,
        "t_end": t_end,
        "ttft": ttft,
        "e2e": e2e,
        "decode_dur": decode_dur,
        "per_token_itl": per_token_itl,
        "decode_tps": decode_tps,
        "chunk_times": chunk_times,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": eff_compl,
        "cached_tokens": cached_tokens,
        "finish_reason": finish_reason,
        "has_tool_calls": has_tool_calls,
        "status_code": status_code,
        "error": error_msg,
    }


def parse_vllm_prometheus(text: str) -> dict[str, Any]:
    """Extracts prefix cache, KV usage, running/waiting batch, preemption, and DSpark counters from /metrics."""
    kv_by_eng: dict[str, float] = {}
    run_by_eng: dict[str, float] = {}
    wait_by_eng: dict[str, float] = {}
    preempt_total = 0.0
    prefix_hits = 0.0
    prefix_queries = 0.0
    prompt_tok_total = 0.0
    gen_tok_total = 0.0
    req_success_total = 0.0
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
        elif name == "vllm:prompt_tokens_total":
            prompt_tok_total += val
        elif name == "vllm:generation_tokens_total":
            gen_tok_total += val
        elif name == "vllm:request_success_total":
            req_success_total += val
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
        "prompt_tokens_total": prompt_tok_total,
        "generation_tokens_total": gen_tok_total,
        "request_success_total": req_success_total,
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


async def flush_endpoints(
    http_session: aiohttp.ClientSession, prefill_urls: list[str], decode_urls: list[str]
) -> None:
    await asyncio.sleep(2.0)
    for role_label, urls in [("prefill", prefill_urls), ("decode", decode_urls)]:
        for idx, u in enumerate(urls):
            ok = False
            last_code = 0
            for _ in range(10):
                for ep in ("/reset_prefix_cache", "/flush_cache"):
                    try:
                        async with http_session.post(
                            f"{u.rstrip('/')}{ep}", timeout=aiohttp.ClientTimeout(total=15)
                        ) as resp:
                            last_code = resp.status
                            if resp.status == 200:
                                ok = True
                                break
                    except Exception:
                        pass
                if ok:
                    break
                await asyncio.sleep(1.0)
            print(f"  [flush] {role_label}[{idx}] ({u}): {'OK (HTTP 200)' if ok else f'WARN ({last_code})'}", flush=True)
    await asyncio.sleep(1.0)


def diff_endpoint_metrics(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    samples: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    urls = sorted(set(before.keys()) | set(after.keys()))
    per_endpoint: dict[str, Any] = {}
    tot_hits = 0.0
    tot_queries = 0.0
    tot_preemptions = 0.0
    tot_drafts = 0.0
    tot_draft_tokens = 0.0
    tot_accepted = 0.0
    tot_pos: dict[int, float] = {}

    steady_rank_means: list[float] = []
    peak_single_rank_kv = 0.0
    steady_running_totals: list[float] = []
    peak_running_total = 0.0

    for s in samples:
        ep_kv_vals: list[float] = []
        ep_run_sum = 0.0
        for u, m in s.items():
            kv_vals = list(m["kv_pct_by_engine"].values())
            if kv_vals:
                ep_kv_vals.extend(kv_vals)
                peak_single_rank_kv = max(peak_single_rank_kv, max(kv_vals))
            ep_run_sum += sum(m["running_by_engine"].values())
        if ep_kv_vals:
            steady_rank_means.append(statistics.mean(ep_kv_vals))
        steady_running_totals.append(ep_run_sum)
        peak_running_total = max(peak_running_total, ep_run_sum)

    for u in urls:
        b = before.get(u, {})
        a = after.get(u, {})
        d_hits = max(0.0, a.get("prefix_hits_total", 0.0) - b.get("prefix_hits_total", 0.0))
        d_queries = max(0.0, a.get("prefix_queries_total", 0.0) - b.get("prefix_queries_total", 0.0))
        d_preempt = max(0.0, a.get("preemptions_total", 0.0) - b.get("preemptions_total", 0.0))
        d_prompt = max(0.0, a.get("prompt_tokens_total", 0.0) - b.get("prompt_tokens_total", 0.0))
        d_gen = max(0.0, a.get("generation_tokens_total", 0.0) - b.get("generation_tokens_total", 0.0))
        d_reqs = max(0.0, a.get("request_success_total", 0.0) - b.get("request_success_total", 0.0))
        d_drafts = max(0.0, a.get("spec_drafts", 0.0) - b.get("spec_drafts", 0.0))
        d_dtok = max(0.0, a.get("spec_draft_tokens", 0.0) - b.get("spec_draft_tokens", 0.0))
        d_acc = max(0.0, a.get("spec_accepted", 0.0) - b.get("spec_accepted", 0.0))

        tot_hits += d_hits
        tot_queries += d_queries
        tot_preemptions += d_preempt
        tot_drafts += d_drafts
        tot_draft_tokens += d_dtok
        tot_accepted += d_acc
        for pos, v in a.get("spec_per_pos", {}).items():
            tot_pos[pos] = tot_pos.get(pos, 0.0) + max(0.0, v - b.get("spec_per_pos", {}).get(pos, 0.0))

        ep_kv_samples = [
            statistics.mean(s[u]["kv_pct_by_engine"].values())
            for s in samples
            if u in s and s[u]["kv_pct_by_engine"]
        ]
        ep_kv_peak = max(
            (max(s[u]["kv_pct_by_engine"].values()) for s in samples if u in s and s[u]["kv_pct_by_engine"]),
            default=0.0,
        )
        ep_run_samples = [sum(s[u]["running_by_engine"].values()) for s in samples if u in s]

        per_endpoint[u] = {
            "requests_completed_delta": d_reqs,
            "prompt_tokens_delta": d_prompt,
            "generation_tokens_delta": d_gen,
            "prefix_hits_delta": d_hits,
            "prefix_queries_delta": d_queries,
            "prometheus_prefix_hit_pct": round(100.0 * d_hits / d_queries, 2) if d_queries > 0 else 0.0,
            "preemptions_delta": d_preempt,
            "kv_usage_steady_mean_pct": round(statistics.mean(ep_kv_samples), 2) if ep_kv_samples else 0.0,
            "kv_usage_peak_rank_pct": round(ep_kv_peak, 2),
            "running_batch_steady_mean": round(statistics.mean(ep_run_samples), 2) if ep_run_samples else 0.0,
            "running_batch_peak": round(max(ep_run_samples), 2) if ep_run_samples else 0.0,
        }

    tau = (1.0 + tot_accepted / tot_drafts) if tot_drafts > 0 else 1.0
    per_pos_rates = [round(tot_pos.get(k, 0.0) / tot_drafts, 4) for k in sorted(tot_pos)] if tot_drafts > 0 else []

    return {
        "prometheus_prefix_hits_delta": tot_hits,
        "prometheus_prefix_queries_delta": tot_queries,
        "prometheus_prefix_cache_hit_pct": round(100.0 * tot_hits / tot_queries, 2) if tot_queries > 0 else 0.0,
        "preemptions_delta": tot_preemptions,
        "kv_usage_steady_mean_pct": round(statistics.mean(steady_rank_means), 2) if steady_rank_means else 0.0,
        "kv_usage_peak_single_rank_pct": round(peak_single_rank_kv, 2),
        "running_batch_steady_mean": round(statistics.mean(steady_running_totals), 2) if steady_running_totals else 0.0,
        "running_batch_peak": round(peak_running_total, 2),
        "dspark": {
            "num_drafts": tot_drafts,
            "num_draft_tokens": tot_draft_tokens,
            "num_accepted": tot_accepted,
            "mean_accept_len_tau": round(tau, 4),
            "per_pos_accept_rates": per_pos_rates,
        },
        "per_endpoint": per_endpoint,
    }


async def warm_trunks_through_router(
    http_session: aiohttp.ClientSession,
    *,
    target_url: str,
    model_name: str,
    trunk_namespace: str,
    all_trunks: list[dict[str, Any]],
    raw_reqs: list[dict[str, Any]],
    cache_hit_ratio: float,
    warmup_concurrency: int,
    warmup_repeats: int,
) -> dict[str, Any]:
    """Warms all 256 trunks through the llm-d router in descending ISL bucket order."""
    ordered = sorted(all_trunks, key=lambda sp: (-sp["bucket_idx"], sp["trunk_id"]))
    sem = asyncio.Semaphore(max(1, warmup_concurrency))
    t0 = time.perf_counter()

    async def _warm_one(sp: dict[str, Any], rep_idx: int) -> dict[str, Any]:
        trunk_id = sp["trunk_id"]
        b_idx = sp["bucket_idx"]
        orig_req = raw_reqs[trunk_id % len(raw_reqs)]
        b_max_tok = int(orig_req.get("max_tokens") or 8192)
        payload = build_trunk_chat_payload(
            orig_req,
            model_name=model_name,
            trunk_namespace=trunk_namespace,
            session_id=trunk_id,
            trunk_id=trunk_id,
            bucket_idx=b_idx,
            target_isl_tokens=sp["target_isl_tokens"],
            cache_hit_ratio=cache_hit_ratio,
            uid=f"warm_r{rep_idx}_k{trunk_id}",
            max_tokens=1,
            bucket_orig_max_tokens=b_max_tok,
            ignore_eos=False,
            is_warmup=True,
        )
        async with sem:
            for att in range(2):
                res = await send_chat_request(
                    http_session,
                    target_url,
                    payload,
                    {"session_id": trunk_id, "trunk_id": trunk_id, "bucket": sp["bucket"], "turn": -1},
                )
                if res["error"] is None:
                    return res
                await asyncio.sleep(1.0)
            return res

    all_warm_results: list[dict[str, Any]] = []
    for rep in range(max(1, warmup_repeats)):
        batch = await asyncio.gather(*(_warm_one(sp, rep) for sp in ordered))
        all_warm_results.extend(batch)

    ok_cnt = sum(1 for r in all_warm_results if r["error"] is None)
    tot_prompt = sum(r["prompt_tokens"] for r in all_warm_results if r["error"] is None)
    elapsed = round(time.perf_counter() - t0, 2)
    print(
        f"[WARMUP] Completed {ok_cnt}/{len(all_warm_results)} warmup requests across {len(all_trunks)} trunks "
        f"({tot_prompt:,} prompt tok) in {elapsed:.2f}s",
        flush=True,
    )
    return {
        "num_trunks": len(all_trunks),
        "warmup_repeats": warmup_repeats,
        "warmup_requests_ok": ok_cnt,
        "warmup_requests_total": len(all_warm_results),
        "warmup_prompt_tokens": tot_prompt,
        "warmup_wall_s": elapsed,
    }


async def run_concurrency_level(
    http_session: aiohttp.ClientSession,
    *,
    routing_label: str,
    num_sessions: int,
    duration_s: float,
    target_url: str,
    prefill_urls: list[str],
    decode_urls: list[str],
    model_name: str,
    trunk_namespace: str,
    all_trunks: list[dict[str, Any]],
    trunks_by_bucket: dict[int, list[dict[str, Any]]],
    raw_reqs: list[dict[str, Any]],
    all_sampling_specs: list[dict[str, Any]],
    cache_hit_ratio: float,
    max_tokens: int,
    ignore_eos: bool,
    num_decode_chips: int,
    num_total_chips: int,
    save_requests_path: str,
) -> dict[str, Any]:
    pre_m0 = await scrape_endpoints(http_session, prefill_urls)
    dec_m0 = await scrape_endpoints(http_session, decode_urls)

    pre_samples: list[dict[str, dict[str, Any]]] = []
    dec_samples: list[dict[str, dict[str, Any]]] = []
    stop_poller = asyncio.Event()

    async def _metrics_poller() -> None:
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

    dispatched_counts = [0] * len(ISL_BUCKETS)
    bucket_cursors = [0] * len(ISL_BUCKETS)

    def sample_next_trunk() -> tuple[int, dict[str, Any], int]:
        d_tot = sum(dispatched_counts)
        b_chosen = max(
            range(len(ISL_BUCKETS)),
            key=lambda b_idx: (
                (d_tot + 1) * ISL_BUCKETS[b_idx][1] - dispatched_counts[b_idx],
                ISL_BUCKETS[b_idx][1],
            ),
        )
        dispatched_counts[b_chosen] += 1
        b_list = trunks_by_bucket[b_chosen]
        cur = bucket_cursors[b_chosen]
        sp = b_list[cur % len(b_list)]
        bucket_cursors[b_chosen] = cur + 1
        return b_chosen, sp, d_tot

    t_bench_start = time.perf_counter()
    t_window_end = t_bench_start + duration_s
    poller_task = asyncio.create_task(_metrics_poller())

    async def session_worker(s_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        turn = 0
        while time.perf_counter() < t_window_end:
            b_idx, sp, req_idx = sample_next_trunk()
            trunk_id = sp["trunk_id"]
            samp_spec = all_sampling_specs[req_idx % len(all_sampling_specs)]
            orig_req = raw_reqs[trunk_id % len(raw_reqs)]
            b_orig_max = int(orig_req.get("max_tokens") or 8192)
            uid = f"{routing_label}_n{num_sessions}_s{s_id}_t{turn}_{int(time.perf_counter()*1000)%1000000}"

            payload = build_trunk_chat_payload(
                orig_req,
                model_name=model_name,
                trunk_namespace=trunk_namespace,
                session_id=s_id,
                trunk_id=trunk_id,
                bucket_idx=b_idx,
                target_isl_tokens=sp["target_isl_tokens"],
                cache_hit_ratio=cache_hit_ratio,
                uid=uid,
                max_tokens=max_tokens,
                bucket_orig_max_tokens=b_orig_max,
                ignore_eos=ignore_eos,
                is_warmup=False,
                sampling_spec=samp_spec,
            )
            res = await send_chat_request(
                http_session,
                target_url,
                payload,
                {
                    "routing_mode": routing_label,
                    "session_id": s_id,
                    "trunk_id": trunk_id,
                    "bucket": sp["bucket"],
                    "target_isl": sp["target_isl_tokens"],
                    "req_max_tokens": payload["max_tokens"],
                    "req_temperature": str(samp_spec.get("temperature", "<omitted>")),
                    "req_top_p": str(samp_spec.get("top_p", "<omitted>")),
                    "turn": turn,
                },
            )
            out.append(res)
            turn += 1
        return out

    worker_lists = await asyncio.gather(*(session_worker(i) for i in range(num_sessions)))
    t_bench_end = time.perf_counter()
    stop_poller.set()
    await poller_task

    pre_m1 = await scrape_endpoints(http_session, prefill_urls)
    dec_m1 = await scrape_endpoints(http_session, decode_urls)

    prefill_prom = diff_endpoint_metrics(pre_m0, pre_m1, pre_samples)
    decode_prom = diff_endpoint_metrics(dec_m0, dec_m1, dec_samples)

    wall_clock = t_bench_end - t_bench_start
    results = [r for lst in worker_lists for r in lst]
    ok_results = [r for r in results if r["error"] is None]
    err_results = [r for r in results if r["error"] is not None]
    distinct_trunks_used = len({r["trunk_id"] for r in results})

    window_output_tokens = 0.0
    window_input_tokens = 0.0
    window_uncached_tokens = 0.0
    window_completed_reqs = 0
    for r in ok_results:
        c_times = r.get("chunk_times") or []
        compl = float(r["completion_tokens"] or 0)
        prompt = float(r["prompt_tokens"] or 0)
        cached = float(r["cached_tokens"] or 0)
        uncached = max(0.0, prompt - cached)
        if r["t_end"] <= t_window_end:
            window_output_tokens += compl
            window_input_tokens += prompt
            window_uncached_tokens += uncached
            window_completed_reqs += 1
        elif c_times:
            in_win_chunks = sum(1 for ct in c_times if ct <= t_window_end)
            if in_win_chunks > 0:
                frac = in_win_chunks / len(c_times)
                window_output_tokens += compl * frac
                window_input_tokens += prompt
                window_uncached_tokens += uncached

    win_ok = [r for r in ok_results if r["t_end"] <= t_window_end]
    eval_win = win_ok if len(win_ok) >= max(4, num_sessions // 2) else ok_results

    all_ttfts = [r["ttft"] for r in ok_results if r["ttft"] is not None]
    win_ttfts = [r["ttft"] for r in eval_win if r["ttft"] is not None]
    all_tps = [r["decode_tps"] for r in ok_results if r.get("decode_tps") is not None]
    win_tps = [r["decode_tps"] for r in eval_win if r.get("decode_tps") is not None]
    all_itls = [r["per_token_itl"] for r in ok_results if r.get("per_token_itl") is not None]

    all_prompts = [float(r["prompt_tokens"] or 0) for r in ok_results]
    all_compls = [float(r["completion_tokens"] or 0) for r in ok_results]
    tot_prompt = sum(all_prompts)
    tot_cached = sum(float(r["cached_tokens"] or 0) for r in ok_results)
    tot_uncached = max(0.0, tot_prompt - tot_cached)
    tot_compl = sum(all_compls)
    api_cache_hit_pct = (100.0 * tot_cached / tot_prompt) if tot_prompt > 0 else 0.0
    tool_call_reqs = sum(1 for r in ok_results if r.get("has_tool_calls") or r.get("finish_reason") == "tool_calls")

    win_out_tps = window_output_tokens / duration_s
    win_in_tps = window_input_tokens / duration_s
    win_uncached_tps = window_uncached_tokens / duration_s
    drain_out_tps = tot_compl / wall_clock if wall_clock > 0 else 0.0

    bucket_trunk_counts = {
        ISL_BUCKETS[b][0]: len(trunks_by_bucket[b]) for b in range(len(ISL_BUCKETS))
    }
    bucket_breakdown: dict[str, Any] = {}
    for b_name, _, _, _ in ISL_BUCKETS:
        b_res = [r for r in ok_results if r["bucket"] == b_name]
        b_ttfts = [r["ttft"] for r in b_res if r["ttft"] is not None]
        b_tps = [r["decode_tps"] for r in b_res if r.get("decode_tps") is not None]
        b_p = sum(r["prompt_tokens"] for r in b_res)
        b_c = sum(r["cached_tokens"] for r in b_res)
        b_o = sum(r["completion_tokens"] for r in b_res)
        bucket_breakdown[b_name] = {
            "count": len(b_res),
            "trunks_configured": bucket_trunk_counts[b_name],
            "mean_isl": round(b_p / len(b_res), 1) if b_res else 0.0,
            "mean_osl": round(b_o / len(b_res), 1) if b_res else 0.0,
            "api_cache_hit_pct": round(100.0 * b_c / b_p, 2) if b_p > 0 else 0.0,
            "ttft_p50_s": round(statistics.median(b_ttfts), 3) if b_ttfts else 0.0,
            "ttft_p95_s": round(percentile(b_ttfts, 95), 3),
            "ttft_p99_s": round(percentile(b_ttfts, 99), 3),
            "decode_tps_median": round(statistics.median(b_tps), 2) if b_tps else 0.0,
            "decode_tps_p90_low": round(percentile(b_tps, 10), 2),
            "decode_tps_p95_low": round(percentile(b_tps, 5), 2),
        }

    for r in results:
        r.pop("chunk_times", None)
    if save_requests_path:
        Path(save_requests_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_requests_path, "w", encoding="utf-8") as fq:
            json.dump(results, fq, indent=2)

    all_p90_low = percentile(all_tps, 10)
    all_ttft_p95 = percentile(all_ttfts, 95)
    gate_compliant = bool(ok_results and all_p90_low >= 50.0 and all_ttft_p95 <= 3.0)

    summary = {
        "routing_mode": routing_label,
        "concurrency_n": num_sessions,
        "steady_window_s": duration_s,
        "drain_wall_s": round(wall_clock, 2),
        "num_trunks_configured": len(all_trunks),
        "distinct_trunks_used": distinct_trunks_used,
        "target_cache_hit_ratio": cache_hit_ratio,
        "ignore_eos": ignore_eos,
        "num_decode_chips": num_decode_chips,
        "num_total_chips": num_total_chips,
        "requests": {
            "total": len(results),
            "successful": len(ok_results),
            "in_window_completed": window_completed_reqs,
            "tool_calls": tool_call_reqs,
            "failed": len(err_results),
            "errors_sample": [r["error"] for r in err_results[:5]],
        },
        "cache_hit": {
            "api_prompt_details_hit_pct": round(api_cache_hit_pct, 2),
            "prometheus_prefill_hit_pct": prefill_prom["prometheus_prefix_cache_hit_pct"],
            "prometheus_hits_delta": prefill_prom["prometheus_prefix_hits_delta"],
            "prometheus_queries_delta": prefill_prom["prometheus_prefix_queries_delta"],
        },
        "tokens": {
            "mean_isl": round(tot_prompt / len(ok_results), 1) if ok_results else 0.0,
            "p50_isl": round(statistics.median(all_prompts), 1) if all_prompts else 0.0,
            "p95_isl": round(percentile(all_prompts, 95), 1),
            "mean_cached_tokens": round(tot_cached / len(ok_results), 1) if ok_results else 0.0,
            "mean_uncached_tokens": round(tot_uncached / len(ok_results), 1) if ok_results else 0.0,
            "mean_osl": round(tot_compl / len(ok_results), 1) if ok_results else 0.0,
            "p50_osl": round(statistics.median(all_compls), 1) if all_compls else 0.0,
            "p95_osl": round(percentile(all_compls, 95), 1),
        },
        "throughput": {
            "window_output_tok_per_s": round(win_out_tps, 2),
            "drain_output_tok_per_s": round(drain_out_tps, 2),
            "output_tok_per_s_per_decode_chip": round(win_out_tps / max(1, num_decode_chips), 2),
            "output_tok_per_s_per_total_chip": round(win_out_tps / max(1, num_total_chips), 2),
            "window_input_tok_per_s": round(win_in_tps, 2),
            "window_uncached_prefill_tok_per_s": round(win_uncached_tps, 2),
            "window_total_in_plus_out_tok_per_s": round(win_in_tps + win_out_tps, 2),
        },
        "per_user_decode_tok_per_s": {
            "in_window": {
                "median": round(statistics.median(win_tps), 2) if win_tps else 0.0,
                "p90_low": round(percentile(win_tps, 10), 2),
                "p95_low": round(percentile(win_tps, 5), 2),
                "p90_high": round(percentile(win_tps, 90), 2),
            },
            "all_requests": {
                "median": round(statistics.median(all_tps), 2) if all_tps else 0.0,
                "p90_low": round(all_p90_low, 2),
                "p95_low": round(percentile(all_tps, 5), 2),
                "p90_high": round(percentile(all_tps, 90), 2),
                "median_itl_ms": round(statistics.median(all_itls) * 1000.0, 2) if all_itls else 0.0,
            },
        },
        "ttft_s": {
            "all_requests_incl_drain": {
                "mean": round(statistics.mean(all_ttfts), 3) if all_ttfts else 0.0,
                "p50": round(statistics.median(all_ttfts), 3) if all_ttfts else 0.0,
                "p90": round(percentile(all_ttfts, 90), 3),
                "p95": round(all_ttft_p95, 3),
                "p99": round(percentile(all_ttfts, 99), 3),
            },
            "in_window_only": {
                "mean": round(statistics.mean(win_ttfts), 3) if win_ttfts else 0.0,
                "p50": round(statistics.median(win_ttfts), 3) if win_ttfts else 0.0,
                "p90": round(percentile(win_ttfts, 90), 3),
                "p95": round(percentile(win_ttfts, 95), 3),
                "p99": round(percentile(win_ttfts, 99), 3),
            },
        },
        "dspark": decode_prom["dspark"],
        "kv_cache_and_preemptions": {
            "prefill_kv_steady_mean_pct": prefill_prom["kv_usage_steady_mean_pct"],
            "prefill_kv_peak_rank_pct": prefill_prom["kv_usage_peak_single_rank_pct"],
            "decode_kv_steady_mean_pct": decode_prom["kv_usage_steady_mean_pct"],
            "decode_kv_peak_rank_pct": decode_prom["kv_usage_peak_single_rank_pct"],
            "prefill_preemptions_delta": prefill_prom["preemptions_delta"],
            "decode_preemptions_delta": decode_prom["preemptions_delta"],
            "decode_running_batch_steady_mean": decode_prom["running_batch_steady_mean"],
            "decode_running_batch_peak": decode_prom["running_batch_peak"],
        },
        "prefill_endpoints_breakdown": prefill_prom["per_endpoint"],
        "decode_endpoints_breakdown": decode_prom["per_endpoint"],
        "buckets": bucket_breakdown,
        "gate_compliant": gate_compliant,
    }

    print(
        f"\n[{routing_label} | n={num_sessions} | {duration_s:.0f}s] "
        f"Out={win_out_tps:,.2f} tok/s ({win_out_tps/num_decode_chips:,.2f}/dec-chip, {win_out_tps/num_total_chips:,.2f}/tot-chip) | "
        f"Hit(API/Prom)={api_cache_hit_pct:.2f}%/{prefill_prom['prometheus_prefix_cache_hit_pct']:.2f}% | "
        f"TTFT(All P50/P95/P99)={summary['ttft_s']['all_requests_incl_drain']['p50']:.3f}/"
        f"{summary['ttft_s']['all_requests_incl_drain']['p95']:.3f}/"
        f"{summary['ttft_s']['all_requests_incl_drain']['p99']:.3f}s | "
        f"UserTPS(Med/P90_low/P95_low)={summary['per_user_decode_tok_per_s']['all_requests']['median']:.2f}/"
        f"{summary['per_user_decode_tok_per_s']['all_requests']['p90_low']:.2f}/"
        f"{summary['per_user_decode_tok_per_s']['all_requests']['p95_low']:.2f} | "
        f"DSpark tau={decode_prom['dspark']['mean_accept_len_tau']:.4f} | "
        f"Err={len(err_results)}/{len(results)}",
        flush=True,
    )
    return summary


async def main_async(args: argparse.Namespace) -> None:
    global DP_PIN_RANKS
    DP_PIN_RANKS = int(getattr(args, "dp_pin_ranks", 0) or 0)
    concurrencies = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
    prefill_urls = [x.strip() for x in args.prefill_metrics_urls.split(",") if x.strip()]
    decode_urls = [x.strip() for x in args.decode_metrics_urls.split(",") if x.strip()]
    target_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    raw_reqs, all_sampling_specs = load_dataset(args.dataset)
    all_trunks, trunks_by_bucket = build_global_trunk_pool(args.num_trunks)
    global DP_PIN_MAP
    if DP_PIN_RANKS > 0 and getattr(args, "dp_pin_mode", "balanced") == "balanced":
        DP_PIN_MAP = build_balanced_dp_map(all_trunks, trunks_by_bucket, DP_PIN_RANKS)
        print(f"[R4/R6] DP pinning: balanced trunk->rank map over {DP_PIN_RANKS} ranks (rank trunk counts="
              f"{[list(DP_PIN_MAP.values()).count(r) for r in range(DP_PIN_RANKS)]})", flush=True)
    elif DP_PIN_RANKS > 0:
        print(f"[R4/R6] DP pinning: trunk_id % {DP_PIN_RANKS}", flush=True)
    bucket_counts = {ISL_BUCKETS[b][0]: len(trunks_by_bucket[b]) for b in range(len(ISL_BUCKETS))}
    print(
        f"[R4/R6 INIT] routing_mode={args.routing_label} | trunks={len(all_trunks)} {bucket_counts} | "
        f"target_hit={args.cache_hit_rate:.2f} | sessions={concurrencies} | window={args.duration_per_n:.0f}s/n | "
        f"ignore_eos={args.ignore_eos} | reset_between_n=False",
        flush=True,
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)
    trunk_namespace = f"{args.routing_label}_tr{len(all_trunks)}"

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        if not args.no_initial_flush:
            print(f"[R4/R6] Initial single cache reset before '{args.routing_label}' warmup & sweep...", flush=True)
            await flush_endpoints(http_session, prefill_urls, decode_urls)

        warmup_summary = await warm_trunks_through_router(
            http_session,
            target_url=target_url,
            model_name=args.model,
            trunk_namespace=trunk_namespace,
            all_trunks=all_trunks,
            raw_reqs=raw_reqs,
            cache_hit_ratio=args.cache_hit_rate,
            warmup_concurrency=args.warmup_concurrency,
            warmup_repeats=args.warmup_repeats,
        )

        level_summaries: list[dict[str, Any]] = []
        for n in concurrencies:
            req_path = ""
            if args.save_requests_dir:
                req_path = str(Path(args.save_requests_dir) / f"requests_{args.routing_label}_n{n}.json")
            # CRITICAL R4 INVARIANT: NO cache reset and NO re-warmup between n=64 -> n=128 -> n=256!
            s = await run_concurrency_level(
                http_session,
                routing_label=args.routing_label,
                num_sessions=n,
                duration_s=args.duration_per_n,
                target_url=target_url,
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                model_name=args.model,
                trunk_namespace=trunk_namespace,
                all_trunks=all_trunks,
                trunks_by_bucket=trunks_by_bucket,
                raw_reqs=raw_reqs,
                all_sampling_specs=all_sampling_specs,
                cache_hit_ratio=args.cache_hit_rate,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
                num_decode_chips=args.num_decode_chips,
                num_total_chips=args.num_total_chips,
                save_requests_path=req_path,
            )
            level_summaries.append(s)

            out_doc = {
                "experiment": args.experiment_name,
                "routing_mode": args.routing_label,
                "base_url": args.base_url,
                "prefill_metrics_urls": prefill_urls,
                "decode_metrics_urls": decode_urls,
                "num_trunks": len(all_trunks),
                "bucket_trunk_counts": bucket_counts,
                "target_cache_hit_ratio": args.cache_hit_rate,
                "reset_cache_between_levels": False,
                "dp_pin_ranks": DP_PIN_RANKS,
                "dp_pin_mode": (getattr(args, "dp_pin_mode", "balanced") if DP_PIN_RANKS > 0 else "off"),
                "warmup": warmup_summary,
                "levels": level_summaries,
            }
            if args.output_json:
                out_p = Path(args.output_json)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(out_doc, indent=2))

    if args.output_json:
        print(f"\n[DONE] Wrote R4/R6 summary to {args.output_json}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 3 R4 (Cache-Aware vs Round-Robin) & R6 Agentic Routing Harness")
    ap.add_argument("--experiment-name", default="R4_agentic_routing")
    ap.add_argument("--routing-label", default="prefix_aware", help="e.g. roundrobin or prefix_aware")
    ap.add_argument("--base-url", default="http://pd-disaggregation-epp.default.svc.cluster.local:80")
    ap.add_argument(
        "--prefill-metrics-urls",
        default="http://192.168.100.43:8000,http://192.168.100.38:8000",
        help="Comma-separated Prefill vLLM base URLs for /reset_prefix_cache and /metrics",
    )
    ap.add_argument(
        "--decode-metrics-urls",
        default="http://192.168.100.198:9400",
        help="Comma-separated Decode vLLM base URLs for /metrics (pass both :9400 IPs for 2xEP=8)",
    )
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--dataset", default="/data/requests_mocked.jsonl.xz")
    ap.add_argument("--num-trunks", type=int, default=256, help="256 distinct trunks (128/prefill pod) for R4")
    ap.add_argument("--sessions", default="64,128,256", help="Comma-separated concurrency levels")
    ap.add_argument("--duration-per-n", type=float, default=600.0, help="600s steady window per n for R4 (300s for R6)")
    ap.add_argument("--cache-hit-rate", type=float, default=0.80, help="0.80 for R4; 0.95 for R6")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 preserves empirical per-bucket max_tokens")
    ap.add_argument("--ignore-eos", action="store_true", default=False)
    ap.add_argument("--no-ignore-eos", action="store_false", dest="ignore_eos")
    ap.add_argument("--warmup-concurrency", type=int, default=16)
    ap.add_argument("--warmup-repeats", type=int, default=1, help="Number of warmup passes over the 256 trunks")
    ap.add_argument("--no-initial-flush", action="store_true", default=False)
    ap.add_argument("--num-decode-chips", type=int, default=8)
    ap.add_argument("--num-total-chips", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--save-requests-dir", default="")
    ap.add_argument("--output-json", default="/tmp/r4_summary.json")
    ap.add_argument("--dp-pin-mode", default="balanced", choices=["balanced", "mod"],
                    help="balanced = greedy load-weighted trunk->rank map; mod = trunk_id %% N (used by R4 conditions c/d)")
    ap.add_argument("--dp-pin-ranks", type=int, default=0,
                    help="0=off; N>0 adds X-data-parallel-rank=trunk_id%%N (client DP affinity, as in toy_proxy_server.py)")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
