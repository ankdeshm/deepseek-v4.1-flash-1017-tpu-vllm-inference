#!/usr/bin/env python3
"""Monitors r0-prefill-poola-h0 and r0-decode-poola-h1 until all 3 R0 endpoints are healthy."""

import subprocess
import sys
import time


def pod_phase(pod: str) -> str:
    out = subprocess.check_output(
        ["kubectl", "get", "pod", pod, "-o", "jsonpath={.status.phase}"],
        text=True,
    ).strip()
    return out


def pod_logs(pod: str) -> str:
    return subprocess.check_output(
        ["kubectl", "logs", pod, "-c", "runner"],
        text=True,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    seen = {"r0-prefill-poola-h0": 0, "r0-decode-poola-h1": 0}
    t0 = time.time()
    while True:
        for pod in ("r0-prefill-poola-h0", "r0-decode-poola-h1"):
            ph = pod_phase(pod)
            logs = pod_logs(pod)
            lines = logs.splitlines()
            for ln in lines[seen[pod]:]:
                print(f"[{pod}] {ln}", flush=True)
            seen[pod] = len(lines)
            if ph in ("Failed", "Succeeded"):
                print(f"ERROR: {pod} entered terminal phase {ph} before benchmark!", flush=True)
                return 1
            if pod == "r0-prefill-poola-h0" and "ALL THREE ENDPOINTS HEALTHY" in logs:
                print(f"\n[WAIT-READY] Stack healthy in {time.time() - t0:.1f}s!", flush=True)
                return 0
        if time.time() - t0 > 2400:
            print("ERROR: Timed out waiting for R0 stack readiness", flush=True)
            return 2
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
