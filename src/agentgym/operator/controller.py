"""The IncidentRun controller.

A genuine Kubernetes controller: it watches custom resources on a real API server and
reconciles each one to a terminal state, with the properties that make controllers
correct rather than merely functional --

  idempotency        reconciling the same object twice does not run the episode twice
  finalizers         the episode namespace is cleaned up before the object disappears
  owner references   the episode namespace is garbage-collected with its IncidentRun
  status subresource results are written to /status, never to spec
  observedGeneration a spec change is noticed and re-reconciled
  backoff            a failing reconcile retries without hot-looping

Run it against a live control plane:

    python -m agentgym.operator.controller --kubeconfig <path>
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kubernetes.client.rest import ApiException

from agentgym import config as cfg
from agentgym.cluster.client import KubeClient
from agentgym.operator import crd as crd_mod

FINALIZER = "agentgym.io/episode-cleanup"
REQUEUE_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class IncidentRunController:
    kube: KubeClient
    gym: Any                      # AgentGym, injected so episodes share one control plane
    namespace: str = "default"
    trajectory_dir: Path = field(default_factory=lambda: cfg.RUNS_DIR / "operator")
    poll_seconds: float = 2.0

    _backoff: dict[str, float] = field(default_factory=dict, init=False)
    _next_attempt: dict[str, float] = field(default_factory=dict, init=False)
    _running: bool = field(default=True, init=False)

    # --- API helpers --------------------------------------------------------

    def _list(self) -> list[dict[str, Any]]:
        try:
            resp = self.kube.custom.list_namespaced_custom_object(
                crd_mod.GROUP, crd_mod.VERSION, self.namespace, crd_mod.PLURAL
            )
            return resp.get("items", [])
        except ApiException as exc:
            if exc.status == 404:
                return []
            raise

    def _patch_status(self, name: str, status: dict[str, Any]) -> None:
        self.kube.custom.patch_namespaced_custom_object_status(
            crd_mod.GROUP, crd_mod.VERSION, self.namespace, crd_mod.PLURAL, name,
            body={"status": status},
        )

    def _patch_meta(self, name: str, metadata: dict[str, Any]) -> None:
        self.kube.custom.patch_namespaced_custom_object(
            crd_mod.GROUP, crd_mod.VERSION, self.namespace, crd_mod.PLURAL, name,
            body={"metadata": metadata},
        )

    # --- reconcile ----------------------------------------------------------

    def reconcile_once(self) -> int:
        """One pass over all IncidentRuns. Returns how many were acted on."""
        acted = 0
        for obj in self._list():
            name = obj["metadata"]["name"]
            if time.time() < self._next_attempt.get(name, 0.0):
                continue
            try:
                if self._reconcile(obj):
                    acted += 1
                self._backoff.pop(name, None)
                self._next_attempt.pop(name, None)
            except Exception as exc:
                delay = min(MAX_BACKOFF_SECONDS,
                            max(REQUEUE_SECONDS, self._backoff.get(name, 1.0) * 2))
                self._backoff[name] = delay
                self._next_attempt[name] = time.time() + delay
                self._set_failed(name, f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        return acted

    def _reconcile(self, obj: dict[str, Any]) -> bool:
        meta = obj["metadata"]
        name = meta["name"]
        spec = obj.get("spec", {})
        status = obj.get("status") or {}
        generation = meta.get("generation", 1)

        # 1. Deletion: run the finalizer, then release the object.
        if meta.get("deletionTimestamp"):
            self._cleanup(status)
            finalizers = [f for f in meta.get("finalizers", []) if f != FINALIZER]
            self._patch_meta(name, {"finalizers": finalizers})
            return True

        # 2. Adopt: attach the finalizer before creating anything we must clean up.
        if FINALIZER not in (meta.get("finalizers") or []):
            self._patch_meta(name, {
                "finalizers": [*(meta.get("finalizers") or []), FINALIZER]
            })
            return True

        # 3. Terminal and current: nothing to do. This is what makes reconcile
        # idempotent -- a re-list of a finished run must not re-run the episode.
        phase = status.get("phase")
        if phase in ("Succeeded", "Failed") and \
                status.get("observedGeneration") == generation:
            return False

        # 4. Run it.
        self._patch_status(name, {
            "phase": "Injecting", "startedAt": _now(),
            "observedGeneration": generation,
            "message": f"materializing {spec.get('fault')} on {spec.get('topology')}",
        })
        self._run_episode(name, spec, generation)
        return True

    def _run_episode(self, name: str, spec: dict[str, Any], generation: int) -> None:
        from agentgym.data import trajectory as traj_mod
        from agentgym.harness.expert import ExpertPolicy
        from agentgym.harness.harness import AgentHarness, HarnessConfig
        from agentgym.incidents import generator
        from agentgym.incidents.generator import Incident
        from agentgym.mcp.gateway import Budget, ToolGateway
        from agentgym.mcp.perturb import SUITES, PerturbationSpec

        fault = spec["fault"]
        topology = spec["topology"]
        seed = int(spec.get("seed", 0))
        max_calls = int(spec.get("maxToolCalls", 24))

        _, fault_obj, params = generator.build(topology, fault, seed)
        incident = Incident(
            id=f"{name}", topology=topology, fault=fault, seed=seed,
            split="operator", difficulty=fault_obj.difficulty,
            root_cause=fault_obj.root_cause, params=params,
        )

        with self.gym.run_episode(incident) as episode:
            self._patch_status(name, {
                "phase": "Running",
                "episodeNamespace": episode.namespace,
                "trueRootCause": fault_obj.root_cause,
            })
            # The episode namespace is owned by the IncidentRun, so deleting the custom
            # resource garbage-collects the cluster state it created.
            self._set_owner(episode.namespace, name)

            base = SUITES.get(spec.get("perturbation", "clean"), SUITES["clean"])
            gateway = ToolGateway.create(
                episode,
                PerturbationSpec(kinds=base.kinds, severity=base.severity, seed=seed),
                budget=Budget(max_tool_calls=max_calls),
            )
            policy = ExpertPolicy(episode=episode, toolset=gateway.toolset)
            harness = AgentHarness(
                episode=episode, gateway=gateway, policy=policy,
                config=HarnessConfig(max_steps=max_calls),
            )
            self._patch_status(name, {"phase": "Verifying"})
            trajectory = harness.run(
                incident_id=incident.id, split="operator",
                model=spec.get("model") or "expert-scripted",
                perturbation=spec.get("perturbation", "clean"),
                difficulty=incident.difficulty,
            )

        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = self.trajectory_dir / f"{name}.json"
        path.write_text(trajectory.to_json())

        v = trajectory.verification
        self._patch_status(name, {
            "phase": "Succeeded" if v.resolved else "Failed",
            "resolved": v.resolved,
            "rootCauseCorrect": v.root_cause_correct,
            "predictedRootCause": v.predicted_root_cause or "",
            "trueRootCause": v.true_root_cause,
            "reward": f"{trajectory.reward:.2f}",
            "toolCalls": v.tool_calls,
            "unsafeActions": v.unsafe_actions,
            "trajectoryPath": str(path),
            "finishedAt": _now(),
            "observedGeneration": generation,
            "message": "incident resolved" if v.resolved else "incident not resolved",
            "conditions": [{
                "type": "Resolved",
                "status": "True" if v.resolved else "False",
                "reason": "VerifierPassed" if v.resolved else "VerifierFailed",
                "message": trajectory.stop_reason,
                "lastTransitionTime": _now(),
            }],
        })
        del traj_mod

    def _set_owner(self, namespace: str, run_name: str) -> None:
        try:
            run = self.kube.custom.get_namespaced_custom_object(
                crd_mod.GROUP, crd_mod.VERSION, self.namespace, crd_mod.PLURAL, run_name
            )
        except ApiException:
            return
        self.kube.core.patch_namespace(name=namespace, body={"metadata": {
            "ownerReferences": [{
                "apiVersion": f"{crd_mod.GROUP}/{crd_mod.VERSION}",
                "kind": crd_mod.KIND,
                "name": run_name,
                "uid": run["metadata"]["uid"],
                "blockOwnerDeletion": False,
            }],
        }})

    def _cleanup(self, status: dict[str, Any]) -> None:
        namespace = status.get("episodeNamespace")
        if not namespace:
            return
        try:
            self.kube.core.delete_namespace(name=namespace)
        except ApiException:
            pass
        self.gym.sim.reap_namespace(namespace)
        self.gym.sim.forget(namespace)

    def _set_failed(self, name: str, message: str) -> None:
        try:
            self._patch_status(name, {
                "phase": "Failed", "message": message[:500], "finishedAt": _now(),
            })
        except ApiException:
            pass

    # --- loop ---------------------------------------------------------------

    def run_forever(self, max_seconds: float | None = None) -> None:
        crd_mod.ensure_crd(self.kube)
        started = time.time()
        while self._running:
            self.reconcile_once()
            if max_seconds is not None and time.time() - started > max_seconds:
                return
            time.sleep(self.poll_seconds)

    def stop(self) -> None:
        self._running = False


def main() -> None:
    from agentgym.incidents.gym import AgentGym

    p = argparse.ArgumentParser(description="IncidentRun controller")
    p.add_argument("--namespace", default="default")
    p.add_argument("--offset", type=int, default=80)
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()

    with AgentGym(offset=args.offset) as gym:
        controller = IncidentRunController(
            kube=gym.kube, gym=gym, namespace=args.namespace
        )
        crd_mod.ensure_crd(gym.kube)
        print(json.dumps({
            "kubeconfig": str(gym.control_plane.kubeconfig),
            "server": gym.control_plane.server_url,
        }))
        controller.run_forever(max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
