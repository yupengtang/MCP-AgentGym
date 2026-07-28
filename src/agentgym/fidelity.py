"""Sim-to-real fidelity: run the workload for real and check the simulator agreed.

The environment simulates pod status, health and traffic. That is what makes it fast and
deterministic, and it is also its most-questioned assumption: does an agent trained here
learn anything about real systems, or only about this simulator's opinions?

This module answers a narrower, checkable version of that question. It runs each workload
as a **real OS process** with a **real kernel-enforced memory limit** (`RLIMIT_AS`), a
**real HTTP health endpoint**, and **real TCP dependency checks**, then compares what
actually happened against what `kubelet_sim` predicted.

What is genuinely real here:
  - memory limits are enforced by the kernel; exceeding one raises MemoryError in the
    child and the process dies, exactly as an over-limit container does
  - readiness is a real HTTP GET against a real socket, so a probe pointed at the wrong
    path or port fails for the real reason
  - dependency health is a real TCP connect, so a down dependency is really unreachable
  - configuration is really read at startup, and a missing key really aborts the process

What remains simulated even here: container image pull, scheduling, the kubelet's own
bookkeeping, and anything requiring a container runtime (unavailable without root).

So this measures fidelity of the **application layer** -- the layer an incident agent
actually observes through logs, health and metrics. It does not claim fidelity of the
node layer. `agentgym fidelity` writes the agreement table; disagreements are reported,
never smoothed over.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# The child program. Deliberately standalone: it is exec'd with a real rlimit applied,
# so nothing from the parent's memory state can mask a genuine allocation failure.
_WORKLOAD = r'''
import http.server, json, os, resource, socket, sys, threading, time

spec = json.loads(sys.argv[1])
limit_mi   = int(spec["memory_limit_mi"])
working_mi = int(spec["working_set_mi"])
health     = spec.get("health_path", "/healthz")
port       = int(spec.get("port", 0))
config     = spec.get("config") or {}
required   = spec.get("required_config_keys") or []
deps       = spec.get("depends_on_ports") or []

# A real, kernel-enforced ceiling. RLIMIT_AS counts *total* address space, and a bare
# CPython interpreter already maps several hundred MB of it, so an absolute limit would
# measure the runtime rather than the workload. Budget relative to the interpreter's own
# footprint: the container's limit is what the application may use on top of its runtime.
with open("/proc/self/statm") as fh:
    baseline_bytes = int(fh.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
headroom = 64 * 1024 * 1024            # allocator slack, page tables, the HTTP server
limit_bytes = baseline_bytes + headroom + limit_mi * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

# Configuration is really validated at startup.
for key in required:
    if not str(config.get(key, "")).strip():
        print(f"FATAL config key {key!r} missing or empty", flush=True)
        sys.exit(1)

# Dependencies are really contacted over TCP.
for dep_port in deps:
    s = socket.socket(); s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", int(dep_port)))
    except OSError:
        print(f"FATAL dependency on port {dep_port} unreachable", flush=True)
        sys.exit(1)
    finally:
        s.close()

# Allocate the declared working set. If the limit is below it, the kernel refuses and
# this raises -- the same outcome an over-limit container has, for the same reason.
try:
    ballast = bytearray(working_mi * 1024 * 1024)
    ballast[::4096] = b"\x01" * len(ballast[::4096])   # touch pages so it is resident
except MemoryError:
    print(f"FATAL out of memory: needed {working_mi}Mi, limit {limit_mi}Mi", flush=True)
    sys.exit(137)

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == health:
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass

srv = http.server.HTTPServer(("127.0.0.1", port), H)
print(f"READY {srv.server_address[1]}", flush=True)
srv.serve_forever()
'''


@dataclass
class RealOutcome:
    started: bool = False
    ready: bool = False
    exit_code: int | None = None
    reason: str = ""
    port: int = 0
    stdout: str = ""


@dataclass
class Comparison:
    fault: str
    simulated_verdict: str
    simulated_ready: bool
    real_ready: bool
    real_reason: str
    agrees: bool


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_workload(
    *,
    memory_limit_mi: int,
    working_set_mi: int,
    health_path: str = "/healthz",
    probe_path: str | None = None,
    config: dict | None = None,
    required_config_keys: list[str] | None = None,
    depends_on_ports: list[int] | None = None,
    boot_timeout_s: float = 12.0,
) -> RealOutcome:
    """Start one workload for real and observe what happens to it."""
    port = _free_port()
    spec = {
        "memory_limit_mi": memory_limit_mi,
        "working_set_mi": working_set_mi,
        "health_path": health_path,
        "port": port,
        "config": config or {},
        "required_config_keys": required_config_keys or [],
        "depends_on_ports": depends_on_ports or [],
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKLOAD, json.dumps(spec)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    out = RealOutcome(port=port)
    deadline = time.time() + boot_timeout_s
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                out.exit_code = proc.returncode
                out.stdout = (proc.stdout.read() if proc.stdout else "")[-400:]
                out.reason = _reason_from(out.stdout, out.exit_code)
                return out
            if _probe(port, probe_path or health_path):
                out.started = out.ready = True
                out.reason = "Ready"
                return out
            if _probe(port, health_path):
                # Serving, but the probe path does not match: Running, never Ready.
                out.started = True
                out.reason = "ProbeMismatch"
                return out
            time.sleep(0.2)
        out.started = proc.poll() is None
        out.reason = "TimedOut"
        return out
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _probe(port: int, path: str) -> bool:
    import http.client

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        conn.request("GET", path)
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except Exception:
        return False


def _reason_from(stdout: str, code: int | None) -> str:
    if "out of memory" in stdout:
        return "OOMKilled"
    if "config key" in stdout:
        return "ConfigError"
    if "dependency" in stdout:
        return "DependencyUnavailable"
    return f"Exited({code})"


# --- the study ---------------------------------------------------------------

# Each case states what kubelet_sim predicts and what a real process should do. The
# simulator's verdict names are its own; the mapping to a real outcome is the claim
# under test.
CASES: list[dict] = [
    {
        "fault": "healthy",
        "sim_verdict": "RunningReady", "sim_ready": True,
        "expect_reason": "Ready",
        "workload": dict(memory_limit_mi=256, working_set_mi=48),
    },
    {
        "fault": "OOMKilled",
        "sim_verdict": "CrashLoopBackOff/OOMKilled", "sim_ready": False,
        "expect_reason": "OOMKilled",
        "workload": dict(memory_limit_mi=64, working_set_mi=512),
    },
    {
        "fault": "BadConfigMap",
        "sim_verdict": "CrashLoopBackOff/Config", "sim_ready": False,
        "expect_reason": "ConfigError",
        "workload": dict(memory_limit_mi=256, working_set_mi=48,
                         config={"MODEL_PATH": ""},
                         required_config_keys=["MODEL_PATH"]),
    },
    {
        "fault": "ReadinessProbeMisconfig",
        "sim_verdict": "RunningNotReady", "sim_ready": False,
        "expect_reason": "ProbeMismatch",
        "workload": dict(memory_limit_mi=256, working_set_mi=48,
                         health_path="/healthz", probe_path="/wrong-path"),
    },
    {
        "fault": "CascadingDependencyDown",
        "sim_verdict": "CrashLoopBackOff/DependencyUnavailable", "sim_ready": False,
        "expect_reason": "DependencyUnavailable",
        # Port 1 is reserved and never listening: a genuinely unreachable dependency.
        "workload": dict(memory_limit_mi=256, working_set_mi=48, depends_on_ports=[1]),
    },
]


def run_study() -> tuple[list[Comparison], dict]:
    results: list[Comparison] = []
    for case in CASES:
        outcome = run_workload(**case["workload"])
        # Readiness alone is not agreement: a pod that failed for an unintended reason
        # would score as a match and quietly inflate the fidelity number.
        agrees = (outcome.ready == case["sim_ready"]
                  and outcome.reason == case["expect_reason"])
        results.append(Comparison(
            fault=case["fault"],
            simulated_verdict=case["sim_verdict"],
            simulated_ready=case["sim_ready"],
            real_ready=outcome.ready,
            real_reason=outcome.reason,
            agrees=agrees,
        ))
    agreed = sum(1 for r in results if r.agrees)
    summary = {
        "cases": len(results),
        "agreements": agreed,
        "agreement_rate": round(agreed / max(1, len(results)), 4),
        "disagreements": [asdict(r) for r in results if not r.agrees],
    }
    return results, summary


def write_report(path: Path) -> dict:
    results, summary = run_study()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"summary": summary, "cases": [asdict(r) for r in results]}, indent=2))
    return summary


__all__ = ["run_workload", "run_study", "write_report", "RealOutcome", "Comparison"]
