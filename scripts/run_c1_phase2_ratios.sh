#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Atlas Cloud TPU v7x Rerun — Phase 2 Adjusted DeepSeek-V4.1 Ratio Progression
# Executes on Pool B host0 (atlas-dsv41-c1-d0, P = 192.168.100.14:8400):
#   1. 1P2D (12 total chips, 100% Pool B): P=host0, D=[host1, host2]
#   2. 1P3D (16 total chips, 100% Pool B): P=host0, D=[host1, host2, host3]
#   3. 1P4D (20 total chips, Pool B + 1 host Pool A): P=host0, D=[host1..3, PoolA_D4]
# For EACH ratio, runs TWO cache modes with (48, 288) decode caps + bucket 288:
#   (a) 128 distinct trunks per prefill node at 80% target hit
#   (b) 40 shared trunks at 95% target hit (comparability with A1-b)
# ==============================================================================

P0_IP="127.0.0.1"
D1_IP="192.168.100.19"   # Pool B host1 (atlas-dsv41-c1-d1)
D2_IP="192.168.100.16"   # Pool B host2 (atlas-dsv41-c1-d2)
D3_IP="192.168.100.9"    # Pool B host3 (atlas-dsv41-c1-d3)
D4_IP="192.168.100.43"   # Pool A host0 (atlas-dsv41-p0 repurposed as D4)

DATASET="/data/requests_mocked.jsonl"
HARNESS="/workspace/bench_empirical_isl_dsv41_rerun.py"
OUT_DIR="/tmp/c1_phase2"
mkdir -p "${OUT_DIR}"

test -s "${DATASET}" || { echo "FATAL: missing ${DATASET}"; exit 1; }
test -s "${HARNESS}" || { echo "FATAL: empty or missing ${HARNESS}"; exit 1; }

wait_health() {
  local target="$1"
  echo "[C1-PHASE2] $(date -u +%FT%TZ) Waiting for http://${target}/health == 200 ..."
  until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${target}/health" || true)" = "200" ]; do
    sleep 8
  done
  echo "[C1-PHASE2] $(date -u +%FT%TZ) Healthy: http://${target}/health == 200"
}

start_proxy() {
  local ratio_tag="$1"
  shift
  local d_hosts=("$@")
  local d_ports=()
  for _ in "${d_hosts[@]}"; do
    d_ports+=("9400")
  done

  python3 -c '
import glob, os, signal
me = {os.getpid(), os.getppid()}
for p in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        pid = int(p.split("/")[2])
        if pid in me or pid == 1:
            continue
        cmd = open(p, "rb").read().decode("utf-8", "ignore")
        if "toy_proxy_server.py" in cmd:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
'
  sleep 2

  local proxy_log="/dev/shm/pd_proxy_${ratio_tag}.log"
  : > "${proxy_log}"
  nohup python3 -u /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "${P0_IP}" --prefiller-ports 8400 --prefiller-dp-size 8 \
    --decoder-hosts "${d_hosts[@]}" \
    --decoder-ports "${d_ports[@]}" --decoder-dp-size 8 \
    > "${proxy_log}" 2>&1 </dev/null &

  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/healthcheck || true)" = "200" ]; do
    sleep 2
  done
  echo "[C1-PHASE2] $(date -u +%FT%TZ) [${ratio_tag}] Proxy ready: $(curl -s http://127.0.0.1:8000/healthcheck)"
}

smoke_test_proxy() {
  local ratio_tag="$1"
  echo "[C1-PHASE2] $(date -u +%FT%TZ) [${ratio_tag}] Running live Raiden KV smoke test through :8000 ..."
  local code
  code=$(curl -s -o "${OUT_DIR}/smoke_${ratio_tag}.json" -w '%{http_code}' \
    http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0.0}')
  if [ "${code}" != "200" ]; then
    echo "FATAL: smoke test HTTP ${code}: $(cat "${OUT_DIR}/smoke_${ratio_tag}.json")"
    exit 1
  fi
  echo "[C1-PHASE2] $(date -u +%FT%TZ) [${ratio_tag}] Smoke test HTTP 200 OK"
}

run_ratio() {
  local ratio_tag="$1"       # 1p2d | 1p3d | 1p4d
  local total_chips="$2"     # 12   | 16   | 20
  local decode_urls="$3"
  local sess_80="$4"
  local sess_95="$5"
  local warmup_s="$6"

  echo "================================================================================"
  echo "[C1-PHASE2] $(date -u +%FT%TZ) STARTING RATIO ${ratio_tag^^} (${total_chips} TOTAL CHIPS)"
  echo "  Decode URLs : ${decode_urls}"
  echo "  Mode (a)    : 128 distinct trunks @ 80% target hit | sessions=${sess_80}"
  echo "  Mode (b)    : 40 shared trunks @ 95% target hit    | sessions=${sess_95}"
  echo "================================================================================"

  # Mode (a): 128 distinct trunks per prefill node at 80% target hit
  python3 -u "${HARNESS}" \
    --base-url http://127.0.0.1:8000 \
    --prefill-url "http://${P0_IP}:8400" \
    --decode-url "${decode_urls}" \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset "${DATASET}" \
    --sessions "${sess_80}" \
    --duration-per-n 150 \
    --warmup-pass-s "${warmup_s}" \
    --cache-hit-rate 0.80 \
    --trunks-above-n16 128 \
    --max-tokens 0 \
    --no-ignore-eos \
    --dp-size 8 \
    --num-chips "${total_chips}" \
    --output-json "${OUT_DIR}/C1_${ratio_tag}_80pct_128trunks.json" \
    2>&1 | tee "${OUT_DIR}/C1_${ratio_tag}_80pct_128trunks.log"

  # Mode (b): 40 shared trunks at 95% target hit (comparability with A1-b)
  python3 -u "${HARNESS}" \
    --base-url http://127.0.0.1:8000 \
    --prefill-url "http://${P0_IP}:8400" \
    --decode-url "${decode_urls}" \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset "${DATASET}" \
    --sessions "${sess_95}" \
    --duration-per-n 150 \
    --warmup-pass-s 0 \
    --cache-hit-rate 0.95 \
    --trunks-above-n16 40 \
    --max-tokens 0 \
    --no-ignore-eos \
    --dp-size 8 \
    --num-chips "${total_chips}" \
    --output-json "${OUT_DIR}/C1_${ratio_tag}_95pct_40trunks.json" \
    2>&1 | tee "${OUT_DIR}/C1_${ratio_tag}_95pct_40trunks.log"

  echo "[C1-PHASE2] $(date -u +%FT%TZ) COMPLETED RATIO ${ratio_tag^^}"
}

# --- 1. 1P2D (12 chips, Pool B only: host0=P, hosts1-2=D) ---
wait_health "${P0_IP}:8400"
wait_health "${D1_IP}:9400"
wait_health "${D2_IP}:9400"
start_proxy "1p2d" "${D1_IP}" "${D2_IP}"
smoke_test_proxy "1p2d"
run_ratio "1p2d" 12 "http://${D1_IP}:9400,http://${D2_IP}:9400" \
  "64,96,128,192,256,384,512" \
  "128,192,256,320,384,512,640" \
  10

# --- 2. 1P3D (16 chips, Pool B only: host0=P, hosts1-3=D) ---
wait_health "${D3_IP}:9400"
start_proxy "1p3d" "${D1_IP}" "${D2_IP}" "${D3_IP}"
smoke_test_proxy "1p3d"
run_ratio "1p3d" 16 "http://${D1_IP}:9400,http://${D2_IP}:9400,http://${D3_IP}:9400" \
  "64,96,128,192,256,384,512,768" \
  "192,256,384,448,512,768,960" \
  5

# --- 3. 1P4D (20 chips, Pool B 1P+3D + Pool A D4) ---
wait_health "${D4_IP}:9400"
start_proxy "1p4d" "${D1_IP}" "${D2_IP}" "${D3_IP}" "${D4_IP}"
smoke_test_proxy "1p4d"
run_ratio "1p4d" 20 "http://${D1_IP}:9400,http://${D2_IP}:9400,http://${D3_IP}:9400,http://${D4_IP}:9400" \
  "64,96,128,192,256,384,512,768,1024" \
  "256,384,512,640,768,1024,1280" \
  5

echo "[C1-PHASE2] $(date -u +%FT%TZ) ALL_RATIOS_1P2D_1P3D_1P4D_DONE"
