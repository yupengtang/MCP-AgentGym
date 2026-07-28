"""The gym: one control plane, many episodes.

Starting a control plane costs ~13s; starting an episode inside one costs ~1s. A worker
therefore holds a single `AgentGym` open and streams incidents through it.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agentgym import config
from agentgym.cluster.client import KubeClient
from agentgym.cluster.controlplane import ControlPlane
from agentgym.cluster.kubelet_sim import KubeletSim
from agentgym.incidents import generator
from agentgym.incidents.episode import Episode
from agentgym.incidents.faults import FAULTS_BY_NAME
from agentgym.incidents.generator import Incident


@dataclass
class AgentGym:
    """Owns a control plane and mints episodes against it."""

    offset: int = 0
    root: Path | None = None
    keep_data: bool = False

    control_plane: ControlPlane = field(init=False)
    kube: KubeClient = field(init=False)
    sim: KubeletSim = field(init=False)
    _open: bool = field(default=False, init=False)
    _episode_count: int = field(default=0, init=False)

    def __enter__(self) -> AgentGym:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> AgentGym:
        root = self.root or (config.RUNTIME_DIR / f"gym-{self.offset}-{uuid.uuid4().hex[:6]}")
        shutil.rmtree(root, ignore_errors=True)
        self.control_plane = ControlPlane(root=root, offset=self.offset,
                                          keep_data=self.keep_data).start()
        self.kube = KubeClient(self.control_plane.kubeconfig)
        self.sim = KubeletSim(kube=self.kube, namespaces=[])
        self.sim.register_nodes()
        self._open = True
        return self

    def stop(self) -> None:
        if not self._open:
            return
        self.kube.close()
        self.control_plane.stop()
        if not self.keep_data:
            shutil.rmtree(self.control_plane.root, ignore_errors=True)
        self._open = False

    # --- episodes -----------------------------------------------------------

    def episode(self, incident: Incident) -> Episode:
        """Materialize an incident into a fresh namespace."""
        broken, fault, params = generator.build(incident.topology, incident.fault, incident.seed)
        # A generated incident carries its params; rebuilding gives the same ones, but
        # trust the stored copy so a re-run of an old dataset stays faithful.
        params = incident.params or params

        if incident.secondary_fault:
            secondary = FAULTS_BY_NAME[incident.secondary_fault]
            rng_seed = incident.seed + 1
            import random as _random
            broken, _ = secondary.inject(
                broken, _random.Random(f"{incident.topology}|{incident.secondary_fault}|{rng_seed}")
            )

        self._episode_count += 1
        namespace = f"inc-{self._episode_count:05d}-{incident.id[-8:]}"
        ep = Episode(
            control_plane=self.control_plane, kube=self.kube, sim=self.sim,
            namespace=namespace, topology=broken, fault=fault,
            params=params, seed=incident.seed,
        )
        # A compositional incident is only resolved when both faults are; wrap the
        # primary predicate so the verifier stays a single boolean.
        if incident.secondary_fault:
            secondary = FAULTS_BY_NAME[incident.secondary_fault]
            sec_params = incident.secondary_params
            primary_resolved = fault.resolved

            def both(ep_: Episode, p: dict) -> bool:
                return bool(primary_resolved(ep_, p) and secondary.resolved(ep_, sec_params))

            ep.fault = type(fault)(  # dataclass is frozen; build a resolved-composed copy
                name=f"{fault.name}+{secondary.name}",
                root_cause=fault.root_cause,
                difficulty=incident.difficulty,
                summary=f"{fault.summary} Simultaneously: {secondary.summary}",
                inject=fault.inject, resolved=both,
                unsafe=tuple(set(fault.unsafe) | set(secondary.unsafe)),
                post_apply=fault.post_apply,
                tags=tuple(set(fault.tags) | set(secondary.tags)),
            )
        return ep.setup()

    def run_episode(self, incident: Incident):
        """Context manager that guarantees namespace cleanup."""
        gym = self

        class _Ctx:
            def __enter__(self) -> Episode:
                self.ep = gym.episode(incident)
                return self.ep

            def __exit__(self, *exc: object) -> None:
                self.ep.teardown()

        return _Ctx()


__all__ = ["AgentGym"]
