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
        for _ in range(4):
            ep.settle_world()
            if ep.resolved():
                break

        assert ep.resolved(), (
            f"{fault} was not resolved by its known-correct fix: "
            f"verdicts={ep.pod_verdicts()} "
            f"health={ {k: v.reason for k, v in ep.service_health().items()} }"
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
