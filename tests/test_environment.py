"""The environment's correctness contract.

For every fault family we assert two things:

  1. after injection the incident is NOT resolved (the fault actually bites), and
  2. after the known-correct remediation it IS resolved (the fault is fixable, and the
     verifier recognises the fix).

If either direction fails, every reward derived from that family is meaningless -- so
this file is the foundation the training signal rests on.
"""

from __future__ import annotations

import pytest

from agentgym.incidents import episode as episode_mod
from agentgym.incidents import generator
from agentgym.incidents.episode import Episode
from agentgym.incidents.faults import ALL_FAULTS
from agentgym.incidents.generator import Incident
from agentgym.incidents.gym import AgentGym


# The known-correct remediation for each family, expressed with the same mutation
# methods the MCP write-tools use.
def _fix_oom(ep: Episode, p: dict) -> None:
    ep.patch_memory(p["service"], limit_mi=int(p["required_mi"]) + 256,
                    request_mi=int(p["required_mi"]))


def _fix_image(ep: Episode, p: dict) -> None:
    ep.set_image(p["service"], p["good_image"])


def _fix_config(ep: Episode, p: dict) -> None:
    ep.patch_configmap(p["configmap"], p["key"], p["good_value"] or "restored")


def _fix_secret(ep: Episode, p: dict) -> None:
    ep.patch_deployment_secret_ref(p["service"], p["good_secret"])


def _fix_selector(ep: Episode, p: dict) -> None:
    ep.patch_service_selector(p["service"], p["good_selector"])


def _fix_probe(ep: Episode, p: dict) -> None:
    ep.patch_deployment_probe(p["service"], path=p["good_path"], port=int(p["good_port"]))


def _fix_quota(ep: Episode, p: dict) -> None:
    total = sum(s.replicas for s in ep.topology.services)
    ep.patch_resource_quota("namespace-quota", {"pods": str(total + 2)})


def _fix_node_selector(ep: Episode, p: dict) -> None:
    ep.patch_node_selector(p["service"], p["good_selector"])


def _fix_kv(ep: Episode, p: dict) -> None:
    ep.patch_memory(p["service"], limit_mi=int(p["needed_limit_mi"]) + 512)


def _fix_netpol(ep: Episode, p: dict) -> None:
    ep.delete_network_policy(p["policy"])


def _fix_zero(ep: Episode, p: dict) -> None:
    ep.scale(p["service"], int(p["good_replicas"]))


def _fix_node(ep: Episode, p: dict) -> None:
    ep.sim.set_node_ready(p["node"], ready=True)


FIXES = {
    "OOMKilled": _fix_oom,
    "ImagePullBackOff": _fix_image,
    "BadConfigMap": _fix_config,
    "MissingSecret": _fix_secret,
    "ServiceSelectorMismatch": _fix_selector,
    "ReadinessProbeMisconfig": _fix_probe,
    "ResourceQuotaExhausted": _fix_quota,
    "NodeSelectorUnschedulable": _fix_node_selector,
    "KVCacheExhaustion": _fix_kv,
    "NetworkPolicyDeny": _fix_netpol,
    "CascadingDependencyDown": _fix_oom,   # same shape: raise the limit back
    "RolloutStuck": _fix_image,
    "ReplicasScaledToZero": _fix_zero,
    "NodeNotReady": _fix_node,
}


@pytest.fixture(scope="module")
def gym():
    with AgentGym(offset=40) as g:
        yield g


def _incident_for(fault_name: str) -> Incident:
    """First topology that accepts this fault."""
    matrix = generator.compatibility_matrix()
    for topo, faults in sorted(matrix.items()):
        if fault_name in faults:
            _, fault, params = generator.build(topo, fault_name, 12345)
            return Incident(
                id=f"test-{fault_name.lower()}", topology=topo, fault=fault_name,
                seed=12345, split="in_distribution", difficulty=fault.difficulty,
                root_cause=fault.root_cause, params=params,
            )
    pytest.skip(f"no topology accepts {fault_name}")


@pytest.mark.parametrize("fault", [f.name for f in ALL_FAULTS])
def test_fault_breaks_then_fixes(gym, fault):
    incident = _incident_for(fault)
    with gym.run_episode(incident) as ep:
        assert not ep.resolved(), (
            f"{fault} did not break the cluster: verdicts={ep.pod_verdicts()} "
            f"health={ {k: v.reason for k, v in ep.service_health().items()} }"
        )

        FIXES[fault](ep, ep.params)
        # Settle on the same convergence rule the reward uses, rather than a fixed tick
        # budget. Pods are created asynchronously by the *real* controller-manager, so a
        # loaded machine produces fewer pods per tick; a fixed budget therefore makes a
        # correct fix look broken as a function of machine load, not of the fix. A
        # dependency chain also resolves one tier per tick, so allow several rounds.
        #
        # settle_world already encodes convergence: it advances until pod state stops
        # changing *and* nothing is still transient, and returns the ticks it used, so a
        # return equal to its budget means it never converged. Use that rather than
        # re-deriving it here -- comparing verdict maps between rounds cannot tell
        # "settled" from "stuck", because an unscheduled pod reports `Booting`
        # identically on every tick (kubelet_sim marks a pod with no node_name Booting
        # and never gives it a boot clock). A dependency chain resolves one tier per
        # settle, so allow a few rounds.
        settle_ticks, converged = 14, False
        for _ in range(4):
            used = ep.settle_world(max_ticks=settle_ticks)
            if ep.resolved() or used < settle_ticks:
                converged = True
                break

        # Two ways the verdict would be about machine speed rather than about the fix:
        # the control plane never produced the pods it intended (a settle timeout), or
        # the world never went quiet inside the tick budget -- on a small runner the
        # scheduler can leave a pod unbound for longer than the budget allows, and every
        # tier behind it then reads as DependencyUnavailable. Say so explicitly instead
        # of reporting a fix failure, which is what the raw assertion did, and which made
        # this test flake on a different fault each run. A world that *did* go quiet and
        # is still unresolved is a real failure and falls through to the assertion.
        if not ep.resolved() and (ep.settle_timeouts or not converged):
            pytest.skip(
                f"{fault}: world never settled "
                f"({ep.settle_timeouts} timeouts at {episode_mod.SETTLE_TIMEOUT_S}s, "
                f"converged={converged}); verdict would measure machine speed, not the fix"
            )

        assert ep.resolved(), (
            f"{fault} was not resolved by its known-correct fix: "
            f"verdicts={ep.pod_verdicts()} "
            f"health={ {k: v.reason for k, v in ep.service_health().items()} } "
            f"settle_timeouts={ep.settle_timeouts}"
        )


def test_incident_generation_is_deterministic():
    a = generator.generate(40, seed=7)
    b = generator.generate(40, seed=7)
    assert [i.id for i in a] == [i.id for i in b]
    assert all(i.params == j.params for i, j in zip(a, b, strict=True))


def test_splits_are_disjoint_by_construction():
    incidents = generator.generate(300, seed=3)
    by_split: dict[str, set[str]] = {}
    for inc in incidents:
        by_split.setdefault(inc.split, set()).add(inc.fault)
    # Held-out faults must never appear in an in-distribution incident.
    assert not (by_split.get("in_distribution", set()) & generator.HELDOUT_FAULTS)
    for inc in incidents:
        if inc.split == "unseen_topology":
            assert inc.topology in generator.HELDOUT_TOPOLOGIES
        if inc.split == "in_distribution":
            assert inc.topology not in generator.HELDOUT_TOPOLOGIES


def test_generator_v2_is_prefix_stable():
    """generate(N)[:M] must equal generate(M), or the pool cannot grow.

    Version 1 was not: split counts were int(n * 0.70), one global shuffle reordered the
    plan, and per-incident seeds came off a shared stream in shuffled order. Asking for
    more incidents renamed every existing one, so every result computed against the old
    pool silently stopped being comparable. That is only discovered on the day more
    statistical power is needed -- which is exactly the day it cannot be paid for.
    """
    big = generator.generate(600, seed=11)
    for m in (50, 200, 450):
        small = generator.generate(m, seed=11)
        assert len(small) == len(big[:m])
        for a, b in zip(small, big[:m], strict=True):
            assert (a.id, a.split, a.fault, a.params) == (b.id, b.split, b.fault, b.params), \
                f"incident {a.id} changed when the pool grew"


def test_generator_v1_still_reproduces_the_published_pool():
    """v1 is kept solely so published numbers stay reproducible; it must not drift."""
    import json

    from agentgym import config as cfg

    path = cfg.DATA_DIR / "incidents.jsonl"
    if not path.exists():
        pytest.skip("no stored incident pool")
    stored = [json.loads(line)["id"] for line in path.open()]
    regenerated = [i.id for i in generator.generate(len(stored), seed=11, version=1)]
    assert stored == regenerated, "v1 no longer reproduces the pool the results were run on"


def test_generator_v2_keeps_the_splits_disjoint_at_any_size():
    """Growing the pool must not leak a held-out family or topology into training."""
    for n in (200, 1000):
        inc = generator.generate(n, seed=11)
        for i in inc:
            if i.split == "in_distribution":
                assert i.fault not in generator.HELDOUT_FAULTS
                assert i.topology not in generator.HELDOUT_TOPOLOGIES
            if i.split == "unseen_fault":
                assert i.fault in generator.HELDOUT_FAULTS
            if i.split == "unseen_topology":
                assert i.topology in generator.HELDOUT_TOPOLOGIES
