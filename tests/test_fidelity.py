"""Sim-to-real fidelity of the application layer.

The simulator's most-questioned assumption is that its pod outcomes resemble reality.
This runs each workload as a real OS process under a real kernel memory limit, with a
real HTTP health endpoint and real TCP dependency checks, and requires the simulator to
have predicted both the readiness *and the reason*.
"""

from __future__ import annotations

from agentgym.fidelity import CASES, run_study, run_workload


def test_healthy_workload_really_serves():
    out = run_workload(memory_limit_mi=256, working_set_mi=48)
    assert out.ready and out.reason == "Ready"


def test_limit_below_working_set_really_fails_to_allocate():
    """Not a simulated verdict: the kernel refuses the allocation."""
    out = run_workload(memory_limit_mi=32, working_set_mi=512)
    assert not out.ready
    assert out.reason == "OOMKilled"
    assert out.exit_code == 137


def test_probe_on_the_wrong_path_really_never_becomes_ready():
    out = run_workload(memory_limit_mi=256, working_set_mi=48,
                       health_path="/healthz", probe_path="/wrong")
    assert out.started and not out.ready
    assert out.reason == "ProbeMismatch"


def test_unreachable_dependency_really_aborts_startup():
    out = run_workload(memory_limit_mi=256, working_set_mi=48, depends_on_ports=[1])
    assert not out.ready and out.reason == "DependencyUnavailable"


def test_simulator_agrees_with_reality_on_every_case():
    results, summary = run_study()
    assert summary["cases"] == len(CASES)
    assert summary["agreement_rate"] == 1.0, (
        "simulator and real process disagreed: "
        f"{summary['disagreements']}"
    )


def test_agreement_requires_the_right_reason_not_just_readiness():
    """Guards the fidelity number itself. An earlier version compared readiness only,
    and scored ReadinessProbeMisconfig as agreeing while the process had actually died
    of memory exhaustion -- a false agreement that inflates the headline."""
    assert all("expect_reason" in c for c in CASES)
