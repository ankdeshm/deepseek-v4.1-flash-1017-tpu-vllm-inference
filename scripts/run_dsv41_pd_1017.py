#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 Disaggregated 1P1D (8 chips) & 2P2D (16 chips) TPU v7x Runner.

Coordinates 4 Indexed Job pods across a 4-host TPU v7x slice (`tpu7x-standard-4t` x 4 = 16 chips / 32 TensorCores):
  - Index 0: Prefill A (`ROLE=prefill, DP=8, TP=1, EP=8, CED_PREFILL=1, MNBT=8192`) + 1P1D Proxy (:8000) + 2P2D Gateway (:8002) + Benchmark Driver
  - Index 1: Decode A  (`ROLE=decode,  DP=8, TP=1, EP=8, ENABLE_DSPARK=1 K=5, MNBT=384`)
  - Index 2: Prefill B (`ROLE=prefill, DP=8, TP=1, EP=8, CED_PREFILL=1, MNBT=8192`) + 1P1D Proxy (:8000)
  - Index 3: Decode B  (`ROLE=decode,  DP=8, TP=1, EP=8, ENABLE_DSPARK=1 K=5, MNBT=384`)

Each host runs independently as a single-host `2x2x1` (`8` TensorCores) via `mock_metadata.py` +
`LD_PRELOAD=/workspace/metadata_hook.so` (with `VBAR_CONTROL_SERVICE_URL` explicitly unset so GKE's
slice vbar socket never blocks single-host init) and transfers KV blocks over `TPURaidenConnector` Stage 3 DCN.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request


GATEWAY_2P2D_PY = r'''#!/usr/bin/env python3
"""2P2D Gateway Proxy across two independent 1P1D pairs (16 TPU v7x chips / 32 TensorCores)."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import itertools
import json
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None, keepalive_expiry=4.0)
    app.state.pairs = []
    for i, url in enumerate(GLOBAL_ARGS.pair_urls):
        app.state.pairs.append({
            "id": i,
            "url": url.rstrip("/"),
            "client": httpx.AsyncClient(timeout=None, base_url=url.rstrip("/"), limits=limits),
            "inflight": 0,
        })
    app.state.decode_metrics_urls = [u.rstrip("/") for u in GLOBAL_ARGS.decode_metrics_urls]
    app.state.metrics_client = httpx.AsyncClient(timeout=15.0, limits=limits)
    app.state.rr = itertools.count()
    yield
    for p in app.state.pairs:
        await p["client"].aclose()
    await app.state.metrics_client.aclose()


app = FastAPI(lifespan=lifespan)


def _pick_pair(app: FastAPI, req_data: dict) -> tuple[dict, dict]:
    raw_dp = req_data.get("dp_rank", None)
    if raw_dp is not None:
        dp_int = int(raw_dp)
        pair_idx = (dp_int // 8) % len(app.state.pairs)
        req_data = dict(req_data)
        req_data["dp_rank"] = dp_int % 8
        return app.state.pairs[pair_idx], req_data
    rr = next(app.state.rr)
    n_p = len(app.state.pairs)
    chosen = min(range(n_p), key=lambda idx: (app.state.pairs[idx]["inflight"], (idx - rr) % n_p))
    return app.state.pairs[chosen], req_data


async def _proxy_post(api: str, request: Request):
    req_data = await request.json()
    pair, fwd_data = _pick_pair(request.app, req_data)
    headers = {"Content-Type": "application/json"}
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]
    is_stream = bool(fwd_data.get("stream", False))
    pair["inflight"] += 1
    if not is_stream:
        try:
            resp = await pair["client"].post(api, json=fwd_data, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        finally:
            pair["inflight"] -= 1

    async def _gen():
        try:
            async with pair["client"].stream("POST", api, json=fwd_data, headers=headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        finally:
            pair["inflight"] -= 1

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_post("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_post("/v1/chat/completions", request)


@app.post("/flush_cache")
@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    async def _one(p):
        try:
            r = await p["client"].post("/reset_prefix_cache", timeout=20.0)
            return {"pair": p["id"], "status_code": r.status_code}
        except Exception as exc:
            return {"pair": p["id"], "error": str(exc)}

    res = await asyncio.gather(*(_one(p) for p in request.app.state.pairs))
    return {"status": "ok", "results": res}


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck(request: Request):
    for p in request.app.state.pairs:
        try:
            r = await p["client"].get("/healthcheck", timeout=5.0)
            if r.status_code != 200:
                return Response(content=json.dumps({"status": "unhealthy", "pair": p["id"]}), status_code=503)
        except Exception as exc:
            return Response(content=json.dumps({"status": "unhealthy", "pair": p["id"], "err": str(exc)}), status_code=503)
    return {"status": "ok", "pairs": len(request.app.state.pairs), "chips": 8 * len(request.app.state.pairs)}


@app.get("/metrics")
async def metrics(request: Request):
    async def _fetch(u: str) -> str:
        try:
            r = await request.app.state.metrics_client.get(f"{u}/metrics", timeout=10.0)
            return r.text if r.status_code == 200 else ""
        except Exception:
            return ""

    parts = await asyncio.gather(*(_fetch(u) for u in request.app.state.decode_metrics_urls))
    return PlainTextResponse("\n".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--pair-urls", nargs="+", required=True)
    ap.add_argument("--decode-metrics-urls", nargs="+", required=True)
    GLOBAL_ARGS = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=GLOBAL_ARGS.host, port=GLOBAL_ARGS.port)
'''


def log(idx: int, msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}][pod-{idx}] {msg}", flush=True)


def vm_metadata(path: str, default: str = "") -> str:
    url = f"http://169.254.169.254/computeMetadata/v1/{path.lstrip('/')}"
    try:
        req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode("utf-8").strip()
    except Exception:
        return default


def vm_access_token() -> str:
    raw = vm_metadata("instance/service-accounts/default/token")
    if raw:
        return json.loads(raw)["access_token"]
    raise RuntimeError("Could not obtain GCE VM service-account OAuth token")


def gcs_download_file(bucket: str, obj_name: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tok = vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj_name, safe='')}?alt=media"
    for att in range(5):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 16 << 20)
            return
        except Exception as exc:
            if att == 4:
                raise
            time.sleep(2 * (att + 1))
            tok = vm_access_token()


def check_http(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def run_fixed_shape_probe(base_url: str, model_name: str, tok_path: Path, isl: int, osl: int, conc: int, n_reqs: int) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    para_ids = tok.encode(
        "Distributed Mixture-of-Experts language models leverage multi-head latent attention, "
        "block-wise low-precision quantization, and speculative multi-token verification across "
        "high-bandwidth accelerator interconnects to maximize per-stream token throughput. ",
        add_special_tokens=False,
    ).ids
    prompts = []
    for i in range(n_reqs):
        hdr = tok.encode(
            f"<｜begin▁of▁sentence｜><｜User｜>[req_nonce={i}_{conc}_{isl}] Architectural analysis notes:\n",
            add_special_tokens=False,
        ).ids
        tail = tok.encode(
            "\nProvide a detailed step-by-step derivation of throughput and latency scaling.<｜Assistant｜></think>",
            add_special_tokens=False,
        ).ids
        mid_len = max(1, isl - len(hdr) - len(tail))
        mid = (para_ids * (mid_len // max(1, len(para_ids)) + 2))[:mid_len]
        prompts.append(hdr + mid + tail)

    def _one(p_ids: list[int]) -> dict:
        payload = {
            "model": model_name,
            "prompt": p_ids,
            "max_tokens": osl,
            "min_tokens": osl,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft = None
        t_last = t0
        chunks = 0
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        break
                    obj = json.loads(line[6:])
                    if obj.get("choices") and obj["choices"][0].get("text"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        t_last = now
                        chunks += 1
            e2e = t_last - t0
            dec_s = max(1e-6, e2e - (ttft or 0.0))
            return {
                "ok": ttft is not None,
                "ttft_ms": (ttft or 0.0) * 1000.0,
                "e2e_s": e2e,
                "tpot_ms": dec_s * 1000.0 / max(1, osl - 1),
                "stream_tps": max(1, osl - 1) / dec_s,
                "chunks": chunks,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # 2-request warmup
    with cf.ThreadPoolExecutor(max_workers=min(2, conc)) as ex:
        list(ex.map(_one, prompts[: min(2, len(prompts))]))

    t_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(_one, prompts))
    wall_s = time.perf_counter() - t_start
    oks = [r for r in res if r.get("ok")]
    out_toks = len(oks) * osl
    in_toks = len(oks) * isl
    return {
        "isl": isl,
        "osl": osl,
        "concurrency": conc,
        "num_prompts": n_reqs,
        "completed": len(oks),
        "failed": len(res) - len(oks),
        "wall_s": round(wall_s, 3),
        "output_tok_s": round(out_toks / max(1e-6, wall_s), 2),
        "total_tok_s": round((in_toks + out_toks) / max(1e-6, wall_s), 2),
        "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in oks), 2) if oks else None,
        "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in oks), 2) if oks else None,
        "mean_tpot_ms": round(statistics.mean(r["tpot_ms"] for r in oks), 2) if oks else None,
        "median_tpot_ms": round(statistics.median(r["tpot_ms"] for r in oks), 2) if oks else None,
        "mean_stream_tps": round(statistics.mean(r["stream_tps"] for r in oks), 2) if oks else None,
        "median_stream_tps": round(statistics.median(r["stream_tps"] for r in oks), 2) if oks else None,
    }


def write_live_report(out_dir: Path, state: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (out_dir / "summary.json").write_text(json.dumps(state, indent=2))
    lines = [
        f"# DeepSeek-V4.1-Flash-1017 (`1P1D` 8-Chip & `2P2D` 16-Chip) TPU v7x Results — `{state['run_tag']}`",
        "",
        f"- **Updated (UTC):** `{state['updated_utc']}`",
        f"- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`MXFP4` MoE + `CED=1` on Prefill + `DSpark K=5` on Decode + `TPURaidenConnector` Stage 3)",
        f"- **Nodes (`2x2x4` = 4x `2x2x1` hosts, 16 `v7x` chips / 32 TensorCores):** `{json.dumps(state.get('nodes', {}))}`",
        f"- **Cold Boot Time:** `{state.get('boot_s', '-')} s`",
        "",
    ]
    if "pd_gates_1p1d" in state:
        g = state["pd_gates_1p1d"]
        lines += [
            "## 1. `1P1D` (8 Chips / 16 TensorCores) Correctness & Micro-Latency Gates (`pd_gates.py`)",
            f"- **Hard Gates Passed (`2+2 x16` & `64-token greedy equality`):** `{g.get('hard_gates_passed')}`",
            f"- **USA 50-States (`PD` / `Local`):** `{g.get('gate_usa', {}).get('pd_states')}` / `{g.get('gate_usa', {}).get('local_states')}`",
            f"- **4K/512 Streaming Probe (`c=8`):** `{json.dumps({k: v for k, v in (g.get('bench_pd') or {}).items() if k != 'per_request'})}`",
            f"- **4K/512 Streaming Probe (`c=64`):** `{json.dumps({k: v for k, v in (g.get('bench_pd_c64') or {}).items() if k != 'per_request'})}`",
            "",
        ]
    for sec_key, sec_title in [
        ("empirical_1p1d_80pct", "2. `1P1D` (8 Chips) Empirical 50:20:12:9:9 ISL Sweep (~80% Prefix Cache)"),
        ("empirical_1p1d_95pct", "3. `1P1D` (8 Chips) Empirical 50:20:12:9:9 ISL Sweep (~95% Prefix Cache)"),
        ("empirical_2p2d_80pct", "4. `2P2D` (16 Chips / Full Slice) Empirical 50:20:12:9:9 ISL Sweep (~80% Prefix Cache)"),
        ("empirical_2p2d_95pct", "5. `2P2D` (16 Chips / Full Slice) Empirical 50:20:12:9:9 ISL Sweep (~95% Prefix Cache)"),
    ]:
        rows = state.get(sec_key)
        if rows:
            lines += [
                f"## {sec_title}",
                "",
                "| Sessions (`n`) | Trunks | Mean ISL | Cache Hit (%) | Mean OSL | Win Out tok/s | Drain Out tok/s | Win In tok/s | Med TPS/stream | Med ITL (ms) | Med TTFT (s) | DSpark Accept Len | OK/Total |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in rows:
                tok = r.get("tokens", {})
                tp = r.get("throughput", {})
                cs = r.get("client_decode_tps_per_stream", {})
                itl = r.get("per_token_itl_ms", {})
                ttft = r.get("ttft_s", {})
                sd = r.get("spec_decode") or {}
                acc_len = sd.get("mean_acceptance_length") or "-"
                lines.append(
                    f"| `{r['num_sessions']}` | `{r['num_trunks_configured']}` | "
                    f"{tok.get('mean_prompt_tokens', 0):,.0f} | {tok.get('observed_cache_hit_pct', 0):.1f}% | "
                    f"{tok.get('mean_completion_tokens', 0):,.1f} | **{tp.get('window_output_tok_per_s', 0):,.1f}** | "
                    f"{tp.get('output_tok_per_s', 0):,.1f} | {tp.get('window_input_tok_per_s', 0):,.1f} | "
                    f"**{cs.get('median', 0):.2f}** | {itl.get('median', 0):.2f} | "
                    f"{ttft.get('median', 0):.2f} | {acc_len} | "
                    f"`{r['successful_requests']}/{r['total_requests']}` |"
                )
            lines.append("")
    if state.get("fixed_shape_probes"):
        lines += [
            "## 6. Fixed-Shape `InferenceX` (`4096:1024`) & Cold-Cache (`1024:512`) Probes (`1P1D` vs `2P2D`)",
            "",
            "| Topology | Chips | Shape (`ISL:OSL`) | Concurrency | Output tok/s | Total tok/s | Med Stream tok/s | Med TPOT (ms) | Med TTFT (ms) | OK/Total |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in state["fixed_shape_probes"]:
            lines.append(
                f"| `{p['topology']}` | `{p['chips']}` | `{p['isl']}:{p['osl']}` | `{p['concurrency']}` | "
                f"**{p['output_tok_s']:,.2f}** | {p['total_tok_s']:,.2f} | "
                f"**{p['median_stream_tps']}** | {p['median_tpot_ms']} | {p['median_ttft_ms']} | "
                f"`{p['completed']}/{p['num_prompts']}` |"
            )
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", "YOUR_GCS_BUCKET"))
    ap.add_argument("--stack-prefix", default="dsv41-stack")
    ap.add_argument("--model-prefix", default="models/deepseek-ai/DeepSeek-V4.1-Flash-1017/")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--output-dir", default="/models/benchmark-results")
    ap.add_argument("--duration-80", type=float, default=120.0)
    ap.add_argument("--duration-95", type=float, default=90.0)
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    role = "prefill" if idx in (0, 2) else "decode"
    pair_name = "pairA" if idx in (0, 1) else "pairB"
    run_id = f"{pair_name}_{args.run_tag}"

    out_root = Path(args.output_dir) / f"v7x-16-dsv41-1017-pd__{args.run_tag}"
    coord_dir = out_root / "coord"
    coord_dir.mkdir(parents=True, exist_ok=True)
    if idx == 0:
        # Clear any stale coordination markers if Job restarted from scratch
        for f in coord_dir.glob("*.json"):
            f.unlink(missing_ok=True)
    else:
        time.sleep(3)

    # 1. Discover real host IP and hostname BEFORE starting mock_metadata.py
    host_ip = vm_metadata("instance/network-interfaces/0/ip") or socket.gethostbyname(socket.gethostname())
    hostname = vm_metadata("instance/hostname") or socket.gethostname()
    node_info = {"idx": idx, "role": role, "pair": pair_name, "run_id": run_id, "ip": host_ip, "hostname": hostname}
    (coord_dir / f"node_{idx}.json").write_text(json.dumps(node_info, indent=2))
    log(idx, f"Registered node_info: {node_info}")

    t_wait = time.time()
    while sum(1 for i in range(4) if (coord_dir / f"node_{i}.json").exists()) < 4:
        if time.time() - t_wait > 600:
            raise TimeoutError("Timed out waiting for all 4 pods to register in coord_dir")
        time.sleep(2)
    nodes = {i: json.loads((coord_dir / f"node_{i}.json").read_text()) for i in range(4)}
    log(idx, f"All 4 nodes registered: { {i: nodes[i]['ip'] for i in range(4)} }")

    # 2. Clean stale processes & stage stack + checkpoint into /dev/shm
    subprocess.run(
        ["bash", "-lc", "for p in $(pgrep -f 'in_pod_serve.sh|vllm serve|VLLM::|EngineCore|toy_proxy_server|mock_metadata.py|gateway_2p2d.py'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; sleep 2"],
        check=False,
    )
    subprocess.run(["bash", "-lc", "mount -o remount,size=850G /dev/shm || true"], check=False)
    subprocess.run(
        ["bash", "-lc", "rm -rf /dev/shm/pd_* /dev/shm/psm_* /dev/shm/sem.* /dev/shm/dsv41_* /tmp/stack.tar && mkdir -p /workspace /dev/shm/tmp"],
        check=True,
    )

    log(idx, "Downloading stack_27be32f509754e88.tar and P/D scripts from regional GCS...")
    gcs_download_file(args.gcs_bucket, f"{args.stack_prefix}/stack_27be32f509754e88.tar", Path("/tmp/stack.tar"))
    for fn in [
        "in_pod_setup.sh",
        "in_pod_serve.sh",
        "pd_gates.py",
        "bench_empirical_isl_dsv41.py",
        "toy_proxy_server.py",
        "launch_dsv41_disagg_role.sh",
        "requests_mocked.jsonl",
    ]:
        gcs_download_file(args.gcs_bucket, f"{args.stack_prefix}/{fn}", Path(f"/tmp/{fn}"))

    stage_sh = r"""
set -euo pipefail
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar && rm -f /tmp/stack.tar
cd /workspace
(gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl) \
  || (apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null && \
      gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl)
cp -f /tmp/toy_proxy_server.py /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py
cp -f /tmp/launch_dsv41_disagg_role.sh /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh
chmod +x /workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh /tmp/in_pod_setup.sh /tmp/in_pod_serve.sh
git config --global --add safe.directory '*' || true
"""
    subprocess.run(["bash", "-lc", stage_sh], check=True)

    # CRITICAL: Remove VBAR_CONTROL_SERVICE_URL export in launch_dsv41_disagg_role.sh so 2x2x4 slice vbar never blocks 2x2x1 single-host init!
    launch_p = Path("/workspace/vllm-torchtpu/examples/disagg/launch_dsv41_disagg_role.sh")
    launch_txt = launch_p.read_text()
    launch_txt = launch_txt.replace(
        '[ -S /var/run/tpu-plugin/vbar.sock ] \\\n  && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"',
        '# Unset VBAR_CONTROL_SERVICE_URL on multi-host GKE nodepools running independent 2x2x1 hosts\nunset VBAR_CONTROL_SERVICE_URL',
    )
    launch_p.write_text(launch_txt)

    # Write /tmp/dsv41_dl.py + /tmp/dsv41_env.sh pointing at our regional us-central1 GCS bucket
    tok = vm_access_token()
    dl_py = r"""
import os, sys, json, time, glob, urllib.request, urllib.parse, concurrent.futures
BUCKET, PREFIX = os.environ['DSV41_GCS_BUCKET'], os.environ['DSV41_GCS_PREFIX']
DST = '/dev/shm/DeepSeek-V4.1-Flash'
CHUNK, BUF = 2 << 30, 16 << 20
os.makedirs(DST, exist_ok=True)
marker = f'{DST}/.done_all'
if os.path.exists(marker) and len(glob.glob(f'{DST}/model-*-of-00048.safetensors')) == 48:
    print('CKPT_ALREADY_READY', flush=True); sys.exit(0)
def _vm_token():
    req = urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
                                 headers={'Metadata-Flavor': 'Google'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())['access_token']
token = _vm_token()
items, pt = [], None
while True:
    url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=' + urllib.parse.quote(PREFIX)
    if pt: url += '&pageToken=' + urllib.parse.quote(pt)
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'}), timeout=30).read())
    items += d.get('items', []); pt = d.get('nextPageToken')
    if not pt: break
blobs = [(it['name'][len(PREFIX):], int(it['size'])) for it in items
         if not it['name'].endswith(('.pdf', '.png', '.jpeg', '.patch', '.md', '/', '.tail'))]
def dl(a):
    rel, dst, s, e = a
    for r in range(6):
        try:
            tk = _vm_token() if r > 0 else token
            u = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(PREFIX + rel, safe="")}?alt=media'
            req = urllib.request.Request(u, headers={'Authorization': f'Bearer {tk}', 'Range': f'bytes={s}-{e-1}'})
            fd = os.open(dst, os.O_WRONLY)
            try:
                pos = s
                with urllib.request.urlopen(req, timeout=180) as resp:
                    while pos < e:
                        b = resp.read(min(BUF, e - pos))
                        if not b: break
                        os.pwrite(fd, b, pos); pos += len(b)
            finally:
                os.close(fd)
            return e - s
        except Exception:
            if r == 5: raise
            time.sleep(2 * (r + 1))
t0 = time.time(); tasks = []; total = 0
for rel, sz in blobs:
    if not rel: continue
    dst = os.path.join(DST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == sz: continue
    fd = os.open(dst, os.O_CREAT | os.O_WRONLY, 0o644); os.ftruncate(fd, sz); os.close(fd)
    total += sz
    for s in range(0, max(1, sz), CHUNK): tasks.append((rel, dst, s, min(s + CHUNK, sz)))
print(f'Downloading {total/1e9:.1f} GB in {len(tasks)} chunks from gs://{BUCKET}/{PREFIX} ...', flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex: list(ex.map(dl, tasks))
open(marker, 'w').close()
print(f'CKPT_READY in {time.time()-t0:.1f}s', flush=True)
"""
    Path("/tmp/dsv41_dl.py").write_text(dl_py)
    Path("/tmp/dsv41_env.sh").write_text(
        f"export GCS_TOKEN='{tok}' DSV41_GCS_BUCKET='{args.gcs_bucket}' "
        f"DSV41_GCS_PREFIX='{args.model_prefix}' DSV41_SHARDS='all'\n"
    )
    pd_env = {
        "ADVERTISE_HOST": host_ip,
        "RUN_ID": run_id,
        "CED_PREFILL": "1",
        "MAX_NUM_BATCHED_TOKENS": "8192" if role == "prefill" else "384",
        "ENABLE_DSPARK": "0" if role == "prefill" else "1",
        "GPU_MEMORY_UTILIZATION": "0.85",
    }
    Path("/tmp/pd_env.sh").write_text("".join(f"export {k}='{v}'\n" for k, v in pd_env.items()))

    log(idx, "Running in_pod_setup.sh (pip wheel install + 510.3 GB GCS tmpfs download)...")
    t_setup = time.time()
    subprocess.run(["bash", "-lc", "bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup.log 2>&1"], check=True)
    setup_tail = Path("/dev/shm/pd_setup.log").read_text()[-600:]
    if "PD_SETUP_OK" not in setup_tail:
        raise RuntimeError(f"in_pod_setup.sh failed:\n{setup_tail}")
    log(idx, f"Setup completed in {time.time() - t_setup:.1f}s:\n{setup_tail.strip()}")

    # Keep a lightweight copy of tokenizer.json before PD_DELETE_CKPT removes *.safetensors
    tok_backup = Path("/tmp/dsv41_tokenizer.json")
    if Path("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json").exists():
        shutil.copy2("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json", tok_backup)

    # 3. Launch vLLM role server (`in_pod_serve.sh prefill|decode`)
    srv_port = 8400 if role == "prefill" else 9400
    log(idx, f"Launching {role} server ({pair_name}, port={srv_port}, cold boot ~20-25 min)...")
    t_boot = time.time()
    srv_proc = subprocess.Popen(["bash", "/tmp/in_pod_serve.sh", role])
    srv_log = Path(f"/dev/shm/pd_{role}.log")
    seen_lines = 0
    progress_keys = (
        "[PD-LAUNCH]",
        "Memory statistics",
        "TPU KV cache size",
        "Raiden DSv4 pool",
        "Application startup complete",
        "POST_WARMUP_HBM",
        "deleted checkpoint",
        "Error",
        "Traceback",
    )
    fail_keys = ("Engine core initialization failed", "RuntimeBufferAllocationFailure", "EngineCore failed", "[PD-SERVE] server exited")

    while True:
        if srv_proc.poll() is not None:
            tail = srv_log.read_text(errors="replace")[-4000:] if srv_log.exists() else ""
            raise RuntimeError(f"{role} server exited prematurely (rc={srv_proc.returncode}):\n{tail}")
        if srv_log.exists():
            all_ln = srv_log.read_text(errors="replace").splitlines()
            for ln in all_ln[seen_lines:]:
                if any(k in ln for k in progress_keys):
                    log(idx, f"[{role}] {ln[:350]}")
                if any(fk in ln for fk in fail_keys):
                    tail = "\n".join(all_ln[-80:])
                    (out_root / f"pod{idx}_{role}_boot_fail.log").write_text(tail)
                    raise RuntimeError(f"{role} failed during boot:\n{ln}\n{tail}")
            seen_lines = len(all_ln)
        if check_http(f"http://127.0.0.1:{srv_port}/health"):
            boot_s = round(time.time() - t_boot, 1)
            log(idx, f"{role.upper()} SERVER HEALTHY on :{srv_port} after {boot_s}s!")
            (coord_dir / f"healthy_{idx}.json").write_text(json.dumps({"idx": idx, "role": role, "boot_s": boot_s}))
            break
        if time.time() - t_boot > 5400:
            raise TimeoutError(f"{role} boot exceeded 5400s")
        time.sleep(15)

    # Wait for ALL 4 servers to be healthy
    while sum(1 for i in range(4) if (coord_dir / f"healthy_{i}.json").exists()) < 4:
        if srv_proc.poll() is not None:
            raise RuntimeError(f"{role} died while waiting for peer servers")
        time.sleep(5)
    boot_times = {i: json.loads((coord_dir / f"healthy_{i}.json").read_text())["boot_s"] for i in range(4)}
    max_boot_s = max(boot_times.values())
    log(idx, f"ALL 4 SERVERS HEALTHY! boot_times={boot_times}")

    # 4. On Prefill pods (idx 0 and idx 2), start local 1P1D toy_proxy_server.py on :8000
    proxy_proc = None
    if idx in (0, 2):
        dec_peer_ip = nodes[idx + 1]["ip"]
        proxy_cmd = [
            "python3",
            "/workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--prefiller-hosts", "127.0.0.1",
            "--prefiller-ports", "8400",
            "--prefiller-dp-size", "8",
            "--decoder-hosts", dec_peer_ip,
            "--decoder-ports", "9400",
            "--decoder-dp-size", "8",
        ]
        proxy_log_f = open("/dev/shm/pd_proxy.log", "w")
        proxy_proc = subprocess.Popen(proxy_cmd, stdout=proxy_log_f, stderr=subprocess.STDOUT)
        for _ in range(30):
            if check_http("http://127.0.0.1:8000/healthcheck"):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"1P1D proxy on pod {idx} failed healthcheck")
        (coord_dir / f"proxy_{idx}.json").write_text(json.dumps({"idx": idx, "port": 8000}))
        log(idx, f"1P1D toy_proxy_server HEALTHY on :8000 -> Decode {dec_peer_ip}:9400")

    # Worker pods (1, 2, 3) wait until Pod 0 writes done.json, then copy logs and exit cleanly
    if idx != 0:
        log(idx, "Serving traffic while pod-0 executes 1P1D & 2P2D benchmark suites...")
        while not (coord_dir / "done.json").exists():
            if srv_proc.poll() is not None:
                raise RuntimeError(f"Server on pod-{idx} exited unexpectedly during benchmark!")
            time.sleep(10)
        log(idx, "Benchmark complete signal received! Harvesting logs...")
        for fn in [f"pd_{role}.log", "pd_setup.log", "pd_proxy.log"]:
            p = Path(f"/dev/shm/{fn}")
            if p.exists():
                shutil.copy2(p, out_root / f"pod{idx}_{fn}")
        srv_proc.terminate()
        if proxy_proc:
            proxy_proc.terminate()
        return 0

    # --- Pod 0 Benchmark Driver ---
    while not (coord_dir / "proxy_2.json").exists():
        time.sleep(2)

    # Start 2P2D Gateway Proxy on :8002 across Pair A (127.0.0.1:8000) and Pair B (nodes[2]['ip']:8000)
    Path("/tmp/gateway_2p2d.py").write_text(GATEWAY_2P2D_PY)
    gw_log_f = open("/dev/shm/pd_gateway_2p2d.log", "w")
    gw_proc = subprocess.Popen(
        [
            "python3",
            "/tmp/gateway_2p2d.py",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--pair-urls", "http://127.0.0.1:8000", f"http://{nodes[2]['ip']}:8000",
            "--decode-metrics-urls", f"http://{nodes[1]['ip']}:9400", f"http://{nodes[3]['ip']}:9400",
        ],
        stdout=gw_log_f,
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        if check_http("http://127.0.0.1:8002/health"):
            break
        time.sleep(2)
    else:
        raise RuntimeError("2P2D Gateway Proxy on :8002 failed healthcheck")
    log(0, "2P2D Gateway Proxy HEALTHY on :8002 (16 chips / 32 TensorCores ready)!")

    state: dict = {
        "run_tag": args.run_tag,
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "nodes": nodes,
        "boot_s": max_boot_s,
        "boot_times_per_pod": boot_times,
        "fixed_shape_probes": [],
    }
    write_live_report(out_root, state)

    try:
        # Suite 1: 1P1D Correctness & Micro-Latency Gates (pd_gates.py)
        log(0, "=== SUITE 1/5: Running 1P1D pd_gates.py correctness & 4K/512 c=8,c=64 probes ===")
        gates_log = out_root / "pd_gates_1p1d.log"
        with open(gates_log, "w") as gf:
            subprocess.run(
                [
                    "python3", "-u", "/tmp/pd_gates.py",
                    "--proxy", "http://127.0.0.1:8000",
                    "--decode", f"http://{nodes[1]['ip']}:9400",
                    "--tokenizer", str(tok_backup if tok_backup.exists() else "/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json"),
                    "--out", "/dev/shm/pd_gates.json",
                ],
                stdout=gf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if Path("/dev/shm/pd_gates.json").exists():
            shutil.copy2("/dev/shm/pd_gates.json", out_root / "pd_gates_1p1d.json")
            state["pd_gates_1p1d"] = json.loads(Path("/dev/shm/pd_gates.json").read_text())
            write_live_report(out_root, state)
            log(0, f"pd_gates completed: hard_gates_passed={state['pd_gates_1p1d'].get('hard_gates_passed')}")

        # Suite 2: 1P1D (8 chips) Empirical 50:20:12:9:9 ISL Sweep @ 80% Prefix Cache
        log(0, "=== SUITE 2/5: Running 1P1D (8 chips) Empirical ISL Sweep (80% cache hit, n=1,8,16,32,64,128) ===")
        emp_1p1d_80_json = out_root / "empirical_1p1d_80pct.json"
        with open(out_root / "empirical_1p1d_80pct.log", "w") as lf:
            subprocess.run(
                [
                    "python3", "-u", "/tmp/bench_empirical_isl_dsv41.py",
                    "--base-url", "http://127.0.0.1:8000",
                    "--prefill-url", "http://127.0.0.1:8400",
                    "--decode-url", f"http://{nodes[1]['ip']}:9400",
                    "--dataset", "/tmp/requests_mocked.jsonl",
                    "--sessions", "1,8,16,32,64,128",
                    "--dp-size", "8",
                    "--num-chips", "8",
                    "--max-pinned-prefix-slots", "16",
                    "--warmup-concurrency", "8",
                    "--warmup-pass-s", "10",
                    "--duration-per-n", str(args.duration_80),
                    "--cache-hit-rate", "0.80",
                    "--flush-each-n",
                    "--output-json", str(emp_1p1d_80_json),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if emp_1p1d_80_json.exists():
            state["empirical_1p1d_80pct"] = json.loads(emp_1p1d_80_json.read_text())
            write_live_report(out_root, state)

        # Suite 3: 2P2D (16 chips / Full Slice) Empirical 50:20:12:9:9 ISL Sweep @ 80% Prefix Cache
        log(0, "=== SUITE 3/5: Running 2P2D (16 chips) Empirical ISL Sweep (80% cache hit, n=16,32,64,128,256) ===")
        emp_2p2d_80_json = out_root / "empirical_2p2d_80pct.json"
        with open(out_root / "empirical_2p2d_80pct.log", "w") as lf:
            subprocess.run(
                [
                    "python3", "-u", "/tmp/bench_empirical_isl_dsv41.py",
                    "--base-url", "http://127.0.0.1:8002",
                    "--prefill-url", "http://127.0.0.1:8002",
                    "--decode-url", "http://127.0.0.1:8002",
                    "--dataset", "/tmp/requests_mocked.jsonl",
                    "--sessions", "16,32,64,128,256",
                    "--dp-size", "16",
                    "--num-chips", "16",
                    "--max-pinned-prefix-slots", "32",
                    "--warmup-concurrency", "16",
                    "--warmup-pass-s", "10",
                    "--duration-per-n", str(args.duration_80),
                    "--cache-hit-rate", "0.80",
                    "--flush-each-n",
                    "--output-json", str(emp_2p2d_80_json),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if emp_2p2d_80_json.exists():
            state["empirical_2p2d_80pct"] = json.loads(emp_2p2d_80_json.read_text())
            write_live_report(out_root, state)

        # Suite 4: 1P1D & 2P2D Empirical 50:20:12:9:9 ISL Sweep @ 95% Prefix Cache
        log(0, "=== SUITE 4/5: Running 1P1D & 2P2D Empirical ISL Sweeps @ 95% Prefix Cache ===")
        emp_1p1d_95_json = out_root / "empirical_1p1d_95pct.json"
        with open(out_root / "empirical_1p1d_95pct.log", "w") as lf:
            subprocess.run(
                [
                    "python3", "-u", "/tmp/bench_empirical_isl_dsv41.py",
                    "--base-url", "http://127.0.0.1:8000",
                    "--prefill-url", "http://127.0.0.1:8400",
                    "--decode-url", f"http://{nodes[1]['ip']}:9400",
                    "--dataset", "/tmp/requests_mocked.jsonl",
                    "--sessions", "8,32,64,128",
                    "--dp-size", "8",
                    "--num-chips", "8",
                    "--max-pinned-prefix-slots", "16",
                    "--warmup-concurrency", "8",
                    "--warmup-pass-s", "0",
                    "--duration-per-n", str(args.duration_95),
                    "--cache-hit-rate", "0.95",
                    "--flush-each-n",
                    "--output-json", str(emp_1p1d_95_json),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if emp_1p1d_95_json.exists():
            state["empirical_1p1d_95pct"] = json.loads(emp_1p1d_95_json.read_text())
            write_live_report(out_root, state)

        emp_2p2d_95_json = out_root / "empirical_2p2d_95pct.json"
        with open(out_root / "empirical_2p2d_95pct.log", "w") as lf:
            subprocess.run(
                [
                    "python3", "-u", "/tmp/bench_empirical_isl_dsv41.py",
                    "--base-url", "http://127.0.0.1:8002",
                    "--prefill-url", "http://127.0.0.1:8002",
                    "--decode-url", "http://127.0.0.1:8002",
                    "--dataset", "/tmp/requests_mocked.jsonl",
                    "--sessions", "16,64,128,256",
                    "--dp-size", "16",
                    "--num-chips", "16",
                    "--max-pinned-prefix-slots", "32",
                    "--warmup-concurrency", "16",
                    "--warmup-pass-s", "0",
                    "--duration-per-n", str(args.duration_95),
                    "--cache-hit-rate", "0.95",
                    "--flush-each-n",
                    "--output-json", str(emp_2p2d_95_json),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if emp_2p2d_95_json.exists():
            state["empirical_2p2d_95pct"] = json.loads(emp_2p2d_95_json.read_text())
            write_live_report(out_root, state)

        # Suite 5: Fixed-Shape InferenceX (4096:1024) & Cold-Cache (1024:512) Comparison Probes
        if tok_backup.exists():
            log(0, "=== SUITE 5/5: Running Fixed-Shape InferenceX (4096:1024) & Cold-Cache (1024:512) Probes ===")
            probe_specs = [
                ("1P1D", 8, "http://127.0.0.1:8000", 1024, 512, 1, 4),
                ("1P1D", 8, "http://127.0.0.1:8000", 1024, 512, 8, 16),
                ("1P1D", 8, "http://127.0.0.1:8000", 4096, 1024, 8, 16),
                ("1P1D", 8, "http://127.0.0.1:8000", 4096, 1024, 64, 64),
                ("2P2D", 16, "http://127.0.0.1:8002", 1024, 512, 16, 32),
                ("2P2D", 16, "http://127.0.0.1:8002", 4096, 1024, 16, 32),
                ("2P2D", 16, "http://127.0.0.1:8002", 4096, 1024, 128, 128),
            ]
            for topo, chips, burl, isl, osl, conc, n_reqs in probe_specs:
                try:
                    urllib.request.urlopen(urllib.request.Request(f"{burl}/reset_prefix_cache", method="POST"), timeout=15).read()
                except Exception:
                    pass
                r = run_fixed_shape_probe(burl, "deepseek-ai/DeepSeek-V4.1-Flash", tok_backup, isl, osl, conc, n_reqs)
                r["topology"] = topo
                r["chips"] = chips
                state["fixed_shape_probes"].append(r)
                write_live_report(out_root, state)
                log(0, f"Probe {topo} ({chips}c) {isl}:{osl} c={conc} -> out={r['output_tok_s']} tok/s, stream={r['median_stream_tps']} tok/s")

    finally:
        (coord_dir / "done.json").write_text(json.dumps({"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}))
        for fn in ["pd_prefill.log", "pd_setup.log", "pd_proxy.log", "pd_gateway_2p2d.log"]:
            p = Path(f"/dev/shm/{fn}")
            if p.exists():
                shutil.copy2(p, out_root / f"pod0_{fn}")
        write_live_report(out_root, state)
        log(0, f"ALL SUITES COMPLETE! Results saved to {out_root}")
        time.sleep(15)
        gw_proc.terminate()
        if proxy_proc:
            proxy_proc.terminate()
        srv_proc.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
