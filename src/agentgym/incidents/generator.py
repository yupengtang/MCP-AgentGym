"""Incident instance generation and the train/eval splits.

An incident is `(topology, fault family, seed)`. Everything else -- which service breaks,
by how much, which config key is blanked -- is derived from the seed, so an incident id
is a complete, reproducible description of a scenario.

Splits are held out along two axes so "generalization" means something specific:

  in_distribution      topologies and faults both seen in training
  unseen_fault         fault family never seen during training
  unseen_topology      familiar fault, unfamiliar application
  compositional        two independent faults injected at once
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentgym.incidents.faults import ALL_FAULTS, FAULTS_BY_NAME, FaultFamily
from agentgym.incidents.topology import TOPOLOGIES_BY_NAME, Topology

# --- the held-out fault axis ------------------------------------------------
#
# The published trade-off (+16.2 pp in-catalogue against -27.6 pp held-out) rests on two
# withheld families. Two is enough to measure an effect and not enough to say what the
# effect is *about*: it cannot distinguish "post-training narrows the diagnosis prior" from
# "post-training happens to be bad at NetworkPolicyDeny and NodeNotReady". Those have
# different implications and the same numbers.
#
# So the withheld set is named rather than hardcoded, and families are grouped by the
# mechanism an agent has to reason about. A held-out set that spans several groups tests
# transfer to unfamiliar *mechanisms*; rotating disjoint folds tests whether the trade-off
# is a property of the training procedure or of particular families.
#
# `published` stays the default and stays exactly what it was. Every number in the README
# and the experiment log was measured against it, and a split that silently changed
# underneath them would invalidate the lot while every table still rendered.

FAULT_GROUPS: dict[str, frozenset[str]] = {
    "workload-resources": frozenset({"OOMKilled", "KVCacheExhaustion"}),
    "workload-config": frozenset({"BadConfigMap", "MissingSecret"}),
    "workload-image": frozenset({"ImagePullBackOff"}),
    "networking": frozenset({"ServiceSelectorMismatch", "NetworkPolicyDeny"}),
    "probes": frozenset({"ReadinessProbeMisconfig"}),
    "scheduling-capacity": frozenset({"NodeSelectorUnschedulable", "ResourceQuotaExhausted",
                                      "NodeNotReady"}),
    "controller-rollout": frozenset({"RolloutStuck", "ReplicasScaledToZero"}),
    "cascade": frozenset({"CascadingDependencyDown"}),
}

HELDOUT_SETS: dict[str, frozenset[str]] = {
    # The published split. Do not edit: it defines what every reported number means.
    "published": frozenset({"NetworkPolicyDeny", "NodeNotReady"}),
    # Six families across six mechanism groups. The breadth answer to "is the regression
    # a property of two specific faults?"
    "broad": frozenset({"NetworkPolicyDeny", "NodeNotReady", "BadConfigMap",
                        "KVCacheExhaustion", "RolloutStuck", "ReadinessProbeMisconfig"}),
    # Three disjoint folds, each withholding four families spanning at least three groups.
    # OOMKilled and ReplicasScaledToZero are in no fold, so every fold keeps a trivial
    # family in training and the corpus never loses its easiest demonstrations.
    "fold_a": frozenset({"NetworkPolicyDeny", "NodeNotReady", "BadConfigMap",
                         "RolloutStuck"}),
    "fold_b": frozenset({"ServiceSelectorMismatch", "ResourceQuotaExhausted",
                         "MissingSecret", "KVCacheExhaustion"}),
    "fold_c": frozenset({"ReadinessProbeMisconfig", "NodeSelectorUnschedulable",
                         "ImagePullBackOff", "CascadingDependencyDown"}),
}

DEFAULT_HELDOUT_SET = "published"


def heldout_faults(name: str | None = None) -> frozenset[str]:
    """Resolve a named held-out set, defaulting to the published one.

    Selected by `AGENTGYM_HELDOUT_SET` so a whole corpus, training run and evaluation can
    be pointed at a different fold without editing code -- but an unknown name raises
    rather than falling back, because a typo that silently produced the published split
    would report a fold's result under the wrong label.
    """
    import os

    key = name or os.environ.get("AGENTGYM_HELDOUT_SET", DEFAULT_HELDOUT_SET)
    try:
        return HELDOUT_SETS[key]
    except KeyError:
        raise ValueError(
            f"unknown held-out set {key!r}; known: {sorted(HELDOUT_SETS)}"
        ) from None


# The active set. Module-level so existing callers keep working unchanged.
HELDOUT_FAULTS: frozenset[str] = heldout_faults()
# Topology withheld from training, to measure transfer to a new application shape.
HELDOUT_TOPOLOGIES: frozenset[str] = frozenset({"event-stream"})


@dataclass
class Incident:
    """A fully specified, reproducible scenario."""

    id: str
    topology: str
    fault: str
    seed: int
    split: str
    difficulty: int
    root_cause: str
    params: dict[str, Any] = field(default_factory=dict)
    # For compositional incidents only.
    secondary_fault: str | None = None
    secondary_params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _incident_id(topology: str, fault: str, seed: int) -> str:
    digest = hashlib.sha1(f"{topology}|{fault}|{seed}".encode()).hexdigest()[:8]
    return f"{fault.lower()}-{topology}-{digest}"


def build(topology_name: str, fault_name: str, seed: int) -> tuple[Topology, FaultFamily, dict]:
    """Materialize the broken topology for one incident. Deterministic in `seed`."""
    topo = TOPOLOGIES_BY_NAME[topology_name]
    fault = FAULTS_BY_NAME[fault_name]
    rng = random.Random(f"{topology_name}|{fault_name}|{seed}")
    broken, params = fault.inject(topo, rng)
    return broken, fault, params


def _compatible(topology_name: str, fault: FaultFamily) -> bool:
    """Can this fault be injected into this topology at all?"""
    try:
        build(topology_name, fault.name, 0)
        return True
    except Exception:
        return False


def compatibility_matrix() -> dict[str, list[str]]:
    return {
        t: sorted(f.name for f in ALL_FAULTS if _compatible(t, f))
        for t in TOPOLOGIES_BY_NAME
    }


# Fraction of each decade of incidents assigned to a split. Fixed positions, not
# proportions of `n`, so incident i's split never depends on how many were requested.
_SPLIT_CYCLE = (("in_distribution",) * 7
                + ("unseen_fault", "unseen_topology", "compositional"))


def _split_at(i: int) -> str:
    return _SPLIT_CYCLE[i % len(_SPLIT_CYCLE)]


def _rank_within_split(i: int) -> int:
    """How many incidents of i's split precede i. Deterministic in i alone."""
    cycle, pos = divmod(i, len(_SPLIT_CYCLE))
    split = _split_at(i)
    per_cycle = _SPLIT_CYCLE.count(split)
    before = sum(1 for j in range(pos) if _SPLIT_CYCLE[j] == split)
    return cycle * per_cycle + before


def generate(n: int, *, seed: int = 0, version: int = 2) -> list[Incident]:
    """Generate `n` incidents spread across the four splits.

    `version=2` is **prefix-stable**: `generate(N)[:M] == generate(M)` for every M <= N, so
    the incident pool can grow without renaming a single existing incident. Version 1 was
    not, because the split counts were `int(n * 0.70)` and friends, one global shuffle
    reordered the whole plan, and per-incident seeds came off a shared stream in shuffled
    order -- asking for more incidents changed every ID, which silently invalidates every
    result computed against the old pool.

    That mattered the day the evaluation needed more statistical power and could not get it
    without discarding its own history. Version 1 is kept so existing published numbers stay
    reproducible; nothing new should use it.
    """
    if version == 1:
        return _generate_v1(n, seed=seed)
    return _generate_v2(n, seed=seed)


def _generate_v2(n: int, *, seed: int) -> list[Incident]:
    """Each incident is a pure function of (seed, index), so prefixes are stable."""
    matrix = compatibility_matrix()
    train_topos = [t for t in matrix if t not in HELDOUT_TOPOLOGIES]
    train_faults = {f.name for f in ALL_FAULTS} - HELDOUT_FAULTS

    def pool(topos: list[str], faults: set[str]) -> list[tuple[str, str]]:
        return sorted((t, f) for t in topos for f in matrix[t] if f in faults)

    pools = {
        "in_distribution": pool(train_topos, train_faults),
        "unseen_fault": pool(train_topos, set(HELDOUT_FAULTS)),
        "unseen_topology": pool(sorted(HELDOUT_TOPOLOGIES), train_faults),
        "compositional": pool(train_topos, train_faults),
    }

    incidents: list[Incident] = []
    for i in range(n):
        split = _split_at(i)
        combos = pools[split]
        if not combos:
            continue
        topo, fault_name = combos[_rank_within_split(i) % len(combos)]
        # Seeded on (seed, i) alone -- not drawn from a stream whose position depends on n.
        rng = random.Random(f"{seed}:{i}")
        s = rng.randrange(1, 2**31)
        try:
            _, fault, params = build(topo, fault_name, s)
        except Exception:
            continue
        inc = Incident(
            id=_incident_id(topo, fault_name, s),
            topology=topo, fault=fault_name, seed=s, split=split,
            difficulty=fault.difficulty, root_cause=fault.root_cause,
            params=params,
        )
        if split == "compositional":
            others = sorted(f for f in matrix[topo]
                            if f != fault_name and f in {x.name for x in ALL_FAULTS})
            for cand in rng.sample(others, k=min(len(others), 4)):
                try:
                    _, cfault, cparams = build(topo, cand, s + 1)
                except Exception:
                    continue
                if cparams.get("service") and cparams.get("service") != params.get("service"):
                    inc.secondary_fault = cand
                    inc.secondary_params = cparams
                    inc.difficulty = max(inc.difficulty, cfault.difficulty) + 1
                    break
        incidents.append(inc)
    return incidents


def _generate_v1(n: int, *, seed: int = 0) -> list[Incident]:
    """The original generator. Kept only so published results stay reproducible."""
    rng = random.Random(seed)
    matrix = compatibility_matrix()
    train_topos = [t for t in matrix if t not in HELDOUT_TOPOLOGIES]

    def pool(topos: list[str], faults: set[str]) -> list[tuple[str, str]]:
        return [(t, f) for t in topos for f in matrix[t] if f in faults]

    train_faults = {f.name for f in ALL_FAULTS} - HELDOUT_FAULTS
    plans: list[tuple[str, str, str]] = []  # (topology, fault, split)

    # 70% in-distribution, then equal thirds of the remainder across the OOD splits.
    n_id = int(n * 0.70)
    n_rest = n - n_id
    n_uf, n_ut = n_rest // 3, n_rest // 3
    n_comp = n_rest - n_uf - n_ut

    for combos, count, split in (
        (pool(train_topos, train_faults), n_id, "in_distribution"),
        (pool(train_topos, set(HELDOUT_FAULTS)), n_uf, "unseen_fault"),
        (pool(sorted(HELDOUT_TOPOLOGIES), train_faults), n_ut, "unseen_topology"),
        (pool(train_topos, train_faults), n_comp, "compositional"),
    ):
        if not combos:
            continue
        for i in range(count):
            topo, fault = combos[i % len(combos)]
            plans.append((topo, fault, split))

    rng.shuffle(plans)
    incidents: list[Incident] = []
    for i, (topo, fault_name, split) in enumerate(plans):
        s = rng.randrange(1, 2**31)
        try:
            _, fault, params = build(topo, fault_name, s)
        except Exception:
            continue
        inc = Incident(
            id=_incident_id(topo, fault_name, s),
            topology=topo, fault=fault_name, seed=s, split=split,
            difficulty=fault.difficulty, root_cause=fault.root_cause,
            params=params,
        )
        if split == "compositional":
            others = [f for f in matrix[topo]
                      if f != fault_name and f in {x.name for x in ALL_FAULTS}]
            # A second fault on a different service, so the two are independently fixable.
            for cand in rng.sample(others, k=min(len(others), 4)):
                try:
                    _, cfault, cparams = build(topo, cand, s + 1)
                except Exception:
                    continue
                if cparams.get("service") and cparams.get("service") != params.get("service"):
                    inc.secondary_fault = cand
                    inc.secondary_params = cparams
                    inc.difficulty = max(inc.difficulty, cfault.difficulty) + 1
                    break
            if inc.secondary_fault is None:
                inc.split = "in_distribution"
        incidents.append(inc)
        del i
    return incidents


def write_jsonl(incidents: list[Incident], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for inc in incidents:
            fh.write(inc.to_json() + "\n")


def read_jsonl(path: Path) -> list[Incident]:
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(Incident(**json.loads(line)))
    return out


__all__ = [
    "Incident", "generate", "build", "compatibility_matrix",
    "write_jsonl", "read_jsonl", "HELDOUT_FAULTS", "HELDOUT_TOPOLOGIES",
]
