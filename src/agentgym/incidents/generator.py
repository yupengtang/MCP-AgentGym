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

# Families withheld from training entirely, to measure transfer to a new failure mode.
HELDOUT_FAULTS: frozenset[str] = frozenset({"NetworkPolicyDeny", "NodeNotReady"})
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


def generate(n: int, *, seed: int = 0) -> list[Incident]:
    """Generate `n` incidents spread across the four splits."""
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
