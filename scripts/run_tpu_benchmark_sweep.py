#!/usr/bin/env python3
"""TPU Benchmark Suite Automated Serving Benchmark & Prefix-Cache Sweep Driver.

Runs against an OpenAI-compatible vLLM endpoint:
  1. Health & /v1/models auto-discovery + sanity generation check
  2. Concurrency sweep (1, 4, 8, 16, 32) using InferenceX / vllm bench
  3. Context-length sweep (2K, 8K, 16K, 32K, 64K up to MAX_INPUT_LEN)
  4. 80% Prefix-Cache Hit Rate workload simulation (async OpenAI streaming)
Writes JSON + Markdown reports to OUTPUT_DIR (e.g. GCSFuse mount on gs://YOUR_GCS_BUCKET).
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


def wait_for_server(base_url: str, timeout_s: int = 86400) -> str:
    start = time.time()
    print(f"[tpu-bench] Waiting up to {timeout_s}s for {base_url}/health ...", flush=True)
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(15)
    else:
        raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout_s}s")

    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        model_id = data["data"][0]["id"]
    print(f"[tpu-bench] Server READY! Auto-detected served model: {model_id}", flush=True)
    return model_id


def run_sanity_check(base_url: str, model_id: str) -> dict:
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Explain Mixture-of-Experts routing on TPUs in two concise sentences."}
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or str(msg)
    usage = body.get("usage", {})
    print(f"[tpu-bench] Sanity check OK in {elapsed:.2f}s (usage={usage}): {text[:240]}", flush=True)
    return {"elapsed_s": round(elapsed, 3), "usage": usage, "sample_output": text[:500]}


async def _stream_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "ignore_eos": True,
        "stream": True,
    }
    async with sem:
        t_start = time.perf_counter()
        t_first = None
        chunks = 0
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=900)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return {"ok": False, "error": f"HTTP {resp.status}: {err[:200]}"}
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                        choices = obj.get("choices", [])
                        if choices and (choices[0].get("text") or choices[0].get("delta")):
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks += 1
                    except Exception:
                        continue
            t_end = time.perf_counter()
            if t_first is None:
                t_first = t_end
            ttft_ms = (t_first - t_start) * 1000.0
            e2e_s = t_end - t_start
            decode_s = max(t_end - t_first, 1e-6)
            out_toks = max(chunks, max_tokens)
            tpot_ms = (decode_s * 1000.0) / max(out_toks - 1, 1)
            stream_tps = (out_toks - 1) / decode_s if out_toks > 1 else out_toks / e2e_s
            e2e_tps = out_toks / max(e2e_s, 1e-6)
            return {
                "ok": True,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "e2e_s": e2e_s,
                "out_tokens": out_toks,
                "stream_tps": stream_tps,
                "e2e_tps": e2e_tps,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def _build_word_block(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = [
        "tensor", "matrix", "router", "expert", "prefill", "decode", "latency",
        "bandwidth", "compiler", "kernel", "attention", "sequence", "cache",
        "pipeline", "parallel", "cluster", "topology", "memory", "compute", "token",
    ]
    # ~1.3 tokens per word + space depending on tokenizer; 0.78 words/token is close to 1:1
    n_words = max(16, int(approx_tokens * 0.78))
    return " ".join(rng.choice(vocab) for _ in range(n_words))


async def run_streaming_benchmark(
    base_url: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    num_prompts: int,
    prefix_hit_ratio: float = 0.0,
) -> dict:
    url = f"{base_url}/v1/completions"
    shared_len = int(input_tokens * prefix_hit_ratio)
    unique_len = max(32, input_tokens - shared_len)
    shared_prefix = _build_word_block(shared_len, seed=42) if shared_len > 0 else ""

    prompts = []
    for i in range(num_prompts):
        uniq = _build_word_block(unique_len, seed=1000 + i)
        prompts.append((shared_prefix + " " + uniq).strip())

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 64))
    async with aiohttp.ClientSession(connector=conn) as session:
        # Warmup shared prefix if testing prefix caching
        if prefix_hit_ratio > 0:
            await _stream_one_request(session, url, model_id, prompts[0], 16, sem)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_stream_one_request(session, url, model_id, p, output_tokens, sem) for p in prompts]
        )
        wall_s = time.perf_counter() - t0

    ok_res = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok_res)
    if not ok_res:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "concurrency": concurrency,
            "num_prompts": num_prompts,
            "prefix_hit_ratio": prefix_hit_ratio,
            "successful": 0,
            "failed": failed,
            "errors": [r.get("error") for r in results[:3]],
        }

    def pct(vals: list[float], p: float) -> float:
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
        return s[idx]

    ttfts = [r["ttft_ms"] for r in ok_res]
    tpots = [r["tpot_ms"] for r in ok_res]
    stream_tpss = [r["stream_tps"] for r in ok_res]
    e2e_tpss = [r["e2e_tps"] for r in ok_res]
    total_out = sum(r["out_tokens"] for r in ok_res)

    summary = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "prefix_hit_ratio": prefix_hit_ratio,
        "successful": len(ok_res),
        "failed": failed,
        "wall_s": round(wall_s, 2),
        "total_output_tps": round(total_out / max(wall_s, 1e-6), 2),
        "mean_stream_tps": round(sum(stream_tpss) / len(stream_tpss), 2),
        "p50_stream_tps": round(pct(stream_tpss, 50), 2),
        "p10_stream_tps": round(pct(stream_tpss, 10), 2),
        "mean_e2e_tps": round(sum(e2e_tpss) / len(e2e_tpss), 2),
        "mean_ttft_ms": round(sum(ttfts) / len(ttfts), 1),
        "p50_ttft_ms": round(pct(ttfts, 50), 1),
        "p90_ttft_ms": round(pct(ttfts, 90), 1),
        "p99_ttft_ms": round(pct(ttfts, 99), 1),
        "mean_tpot_ms": round(sum(tpots) / len(tpots), 2),
        "p90_tpot_ms": round(pct(tpots, 90), 2),
    }
    print(f"[tpu-bench] RESULT: {json.dumps(summary)}", flush=True)
    return summary


def maybe_run_inferencex(
    host: str,
    port: int,
    model_id: str,
    tokenizer: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    out_dir: Path,
) -> dict | None:
    script = Path("/workspace/inferencex/utils/bench_serving/benchmark_serving.py")
    if not script.exists():
        return None
    result_file = out_dir / f"inferencex_in{input_len}_out{output_len}_c{concurrency}.json"
    cmd = [
        sys.executable,
        str(script),
        "--backend=vllm",
        "--request-rate=inf",
        "--percentile-metrics=ttft,tpot,itl,e2el",
        f"--host={host}",
        f"--port={port}",
        f"--model={model_id}",
        f"--tokenizer={tokenizer}",
        "--dataset-name=random",
        f"--random-input-len={input_len}",
        f"--random-output-len={output_len}",
        "--random-range-ratio=0.8",
        f"--num-prompts={num_prompts}",
        f"--max-concurrency={concurrency}",
        "--ignore-eos",
        "--save-result",
        f"--result-filename={result_file}",
    ]
    print(f"[tpu-bench] Running InferenceX: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(f"[tpu-bench] InferenceX warning (rc={proc.returncode}): {proc.stderr[-600:]}", flush=True)
        return None
    if result_file.exists():
        return json.loads(result_file.read_text())
    return None


def render_markdown(meta: dict, sanity: dict, rows: list[dict]) -> str:
    lines = [
        f"# Cloud TPU vLLM Benchmark Report — `{meta['slice_name']}`",
        "",
        f"- **Timestamp (UTC)**: `{meta['timestamp_utc']}`",
        f"- **TPU Slice / Config**: `{meta['slice_name']}` (`{meta['config_tag']}`)",
        f"- **Served Model**: `{meta['model_id']}`",
        f"- **Sanity Check**: `{sanity['elapsed_s']}s` — `{sanity['sample_output'][:180]}...`",
        "",
        "## Sweep Results",
        "",
        "| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("successful", 0) == 0:
            lines.append(
                f"| {r.get('suite','-')} | {r['input_tokens']} | {r['output_tokens']} | "
                f"{int(r['prefix_hit_ratio']*100)}% | {r['concurrency']} | ERR | ERR | ERR | - | - | - | 0/{r['failed']} |"
            )
        else:
            lines.append(
                f"| {r.get('suite','-')} | {r['input_tokens']} | {r['output_tokens']} | "
                f"{int(r['prefix_hit_ratio']*100)}% | {r['concurrency']} | "
                f"**{r['total_output_tps']}** | **{r['mean_stream_tps']}** | {r['mean_e2e_tps']} | "
                f"{r['mean_ttft_ms']} | {r['p90_ttft_ms']} | {r['mean_tpot_ms']} | "
                f"{r['successful']}/{r['failed']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "8000")))
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", ""))
    parser.add_argument("--slice-name", default=os.environ.get("SLICE_NAME", "tpu-slice"))
    parser.add_argument("--config-tag", default=os.environ.get("CONFIG_TAG", "baseline"))
    parser.add_argument("--max-input-len", type=int, default=int(os.environ.get("MAX_INPUT_LEN", "8192")))
    parser.add_argument("--output-len", type=int, default=int(os.environ.get("OUTPUT_LEN", "512")))
    parser.add_argument("--concurrencies", default=os.environ.get("CONCURRENCIES", "1,4,8,16,32"))
    parser.add_argument("--context-lengths", default=os.environ.get("CONTEXT_LENGTHS", "2048,8192"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/models/benchmark-results"))
    parser.add_argument("--run-inferencex", action="store_true", default=True)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    model_id = wait_for_server(base_url)
    tokenizer = args.tokenizer or model_id

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_id.strip("/").replace("/", "__")
    out_dir = Path(args.output_dir) / f"{args.slice_name}__{safe_model}__{args.config_tag}__{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity = run_sanity_check(base_url, model_id)
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    ctx_lens = [int(x) for x in args.context_lengths.split(",") if x.strip() and int(x) <= args.max_input_len]

    all_rows = []
    std_in = min(4096, args.max_input_len)

    # 1. Concurrency sweep (0% prefix hit)
    for c in concs:
        n_prompts = max(c * 2, 8)
        row = asyncio.run(
            run_streaming_benchmark(base_url, model_id, std_in, args.output_len, c, n_prompts, prefix_hit_ratio=0.0)
        )
        row["suite"] = "concurrency_0pct_cache"
        all_rows.append(row)

    # 2. 80% Prefix-Cache Hit Sweep
    for c in [x for x in concs if x in (1, 4, 8, 16, 32, 64)]:
        n_prompts = max(c * 2, 8)
        row = asyncio.run(
            run_streaming_benchmark(base_url, model_id, std_in, args.output_len, c, n_prompts, prefix_hit_ratio=0.8)
        )
        row["suite"] = "cached_80pct_prefix"
        all_rows.append(row)

    # 2B. 95% Prefix-Cache Hit Sweep
    for c in [x for x in concs if x in (1, 4, 8, 16, 32, 64)]:
        n_prompts = max(c * 2, 8)
        row = asyncio.run(
            run_streaming_benchmark(base_url, model_id, std_in, args.output_len, c, n_prompts, prefix_hit_ratio=0.95)
        )
        row["suite"] = "cached_95pct_prefix"
        all_rows.append(row)

    # 3. Context-length sweep at concurrency=4
    for ctx in ctx_lens:
        row = asyncio.run(
            run_streaming_benchmark(base_url, model_id, ctx, args.output_len, min(4, max(concs)), 8, prefix_hit_ratio=0.0)
        )
        row["suite"] = "context_sweep"
        all_rows.append(row)

    # 4. Official InferenceX cross-check at concurrency=8
    infx = None
    if args.run_inferencex:
        infx = maybe_run_inferencex(
            args.host, args.port, model_id, tokenizer, std_in, args.output_len, min(8, max(concs)), 16, out_dir
        )

    meta = {
        "timestamp_utc": ts,
        "slice_name": args.slice_name,
        "config_tag": args.config_tag,
        "model_id": model_id,
        "tokenizer": tokenizer,
        "max_input_len": args.max_input_len,
        "output_len": args.output_len,
    }
    payload = {"meta": meta, "sanity": sanity, "results": all_rows, "inferencex_crosscheck": infx}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md = render_markdown(meta, sanity, all_rows)
    (out_dir / "REPORT.md").write_text(md)
    print("\n================ FINAL BENCHMARK REPORT ================\n", flush=True)
    print(md, flush=True)
    print(f"[tpu-bench] Saved artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
