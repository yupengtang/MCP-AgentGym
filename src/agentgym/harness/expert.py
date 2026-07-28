"""A scripted expert policy.

This is the environment's *solver*: it has access to ground truth and executes a
realistic diagnosis-then-fix sequence for each fault family. It exists for three reasons:

  1. it produces high-quality demonstration trajectories for SFT without a teacher model,
  2. it establishes the performance ceiling the evaluation tables are read against, and
  3. it exercises the whole pipeline end to end without a GPU.

It is a scripted policy, not a language model, and is always reported as such. The
investigation steps are genuine tool calls against the live cluster -- the expert does
not skip to the answer, because a trajectory that jumps straight to the fix teaches a
student model to guess rather than to look.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from agentgym.harness.state import Action, AgentState, Finish, ToolCall
from agentgym.incidents.episode import Episode
from agentgym.mcp.perturb import PerturbedToolset


@dataclass
class _Plan:
    """A queued sequence of (canonical tool, args, thought) plus a terminal diagnosis."""

    steps: list[tuple[str, dict[str, Any], str]]
    root_cause: str
    summary: str


@dataclass
class ExpertPolicy:
    """Solves an incident using ground truth, via ordinary tool calls."""

    episode: Episode
    toolset: PerturbedToolset
    name: str = "expert-scripted"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    _queue: list[tuple[str, dict[str, Any], str]] = field(default_factory=list, init=False)
    _plan: _Plan | None = field(default=None, init=False)
    _failed: set[str] = field(default_factory=set, init=False)

    # --- policy protocol ----------------------------------------------------

    def reset(self, state: AgentState) -> None:
        self._plan = self._build_plan()
        steps = self._diversify(self._plan.steps)
        self._queue = self._with_rollout_waits(steps)
        self._failed.clear()

    def _diversify(
        self, steps: list[tuple[str, dict[str, Any], str]],
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Vary the investigation without changing the diagnosis or the fix.

        A demonstration set where every episode of a family follows the identical script
        teaches the *script*. The model then reproduces the phrasing of step 3 rather
        than learning that step 3 exists to answer a question. Varying which optional
        evidence is gathered, and in what order, keeps every trajectory correct while
        making the surface form far less memorisable.

        Only read-only orientation steps are touched. The mutating fix and the final
        verification stay exactly where they are, because those are what the reward
        actually depends on.
        """
        rng = random.Random(f"{self.episode.namespace}|{self.episode.seed}")

        # Split at the first mutating call: everything before it is investigation.
        mutating = {
            "patch_deployment_resources", "set_image", "scale_deployment",
            "patch_deployment_probe", "patch_deployment_node_selector",
            "patch_deployment_secret_ref", "rollout_restart", "patch_configmap",
            "patch_service_selector", "patch_resource_quota", "delete_network_policy",
        }
        pivot = next((i for i, (name, _, _) in enumerate(steps) if name in mutating),
                     len(steps))
        investigation, remedy = steps[:pivot], steps[pivot:]
        if len(investigation) < 2:
            return steps

        # The first call always establishes context; the rest may be reordered.
        head, tail = investigation[:1], investigation[1:]
        rng.shuffle(tail)

        # Sometimes gather extra evidence a good responder plausibly would.
        ns = self.episode.namespace
        optional = [
            ("list_events", {"namespace": ns},
             "Check cluster events for anything that explains the failure."),
            ("get_recent_changes", {"namespace": ns},
             "See what shipped recently -- the cause is often the newest change."),
            ("list_alerts", {"namespace": ns},
             "Check which alerts are firing to confirm the blast radius."),
        ]
        already = {name for name, _, _ in investigation}
        for candidate in optional:
            if candidate[0] not in already and rng.random() < 0.35:
                tail.insert(rng.randrange(len(tail) + 1), candidate)

        return head + tail + remedy

    @staticmethod
    def _with_rollout_waits(
        steps: list[tuple[str, dict[str, Any], str]],
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Insert a wait after every change to a deployment.

        Checking health the instant a patch lands reports the old state, so the expert
        waits for the rollout first -- and demonstrating that habit is exactly what we
        want a student model to copy.
        """
        rolling = {
            "patch_deployment_resources", "set_image", "scale_deployment",
            "patch_deployment_probe", "patch_deployment_node_selector",
            "patch_deployment_secret_ref", "rollout_restart",
        }
        out: list[tuple[str, dict[str, Any], str]] = []
        for canonical, args, thought in steps:
            out.append((canonical, args, thought))
            if canonical in rolling and "name" in args:
                out.append((
                    "wait_for_rollout",
                    {"namespace": args["namespace"], "name": args["name"]},
                    "Wait for the new pods to roll out before judging whether this worked.",
                ))
        return out

    def act(self, state: AgentState, tool_schemas: list[dict[str, Any]]) -> Action:
        assert self._plan is not None
        while self._queue:
            canonical, args, thought = self._queue.pop(0)
            presented = self._present(canonical)
            if presented is None:
                # The tool was dropped from this perturbed surface; skip it rather than
                # burning budget on a call that cannot resolve.
                continue
            args = self._rename_args(canonical, presented, args)
            return ToolCall(name=presented, args=args, thought=thought)
        return Finish(
            root_cause=self._plan.root_cause,
            summary=self._plan.summary,
            thought="Service health is restored across the namespace; recording the root cause.",
        )

    # --- perturbation-aware name resolution ---------------------------------

    def _present(self, canonical: str) -> str | None:
        """Map a canonical tool name to whatever it is called on this surface."""
        for presented, canon in self.toolset._tool_alias.items():
            if canon == canonical and presented not in self.toolset._distractors:
                return presented
        return None

    def _rename_args(self, canonical: str, presented: str,
                     args: dict[str, Any]) -> dict[str, Any]:
        alias = self.toolset._arg_alias.get(canonical, {})   # presented -> canonical
        reverse = {v: k for k, v in alias.items()}
        out = {reverse.get(k, k): v for k, v in args.items()}
        # Supply any argument the perturbed contract added.
        for extra in self.toolset._required_extra.get(canonical, []):
            out[extra] = "expert-1" if extra == "request_id" else "v2"
        # Honour retyped arguments (integer -> stringly-typed integer).
        for arg in self.toolset._retyped.get(canonical, {}):
            key = reverse.get(arg, arg)
            if key in out:
                out[key] = str(out[key])
        return out

    # --- the plans ----------------------------------------------------------

    def _build_plan(self) -> _Plan:
        ep = self.episode
        ns = ep.namespace
        p = ep.params
        fault = ep.fault.name.split("+")[0]
        svc = p.get("service") or p.get("consumer") or ""

        # Every incident starts the same way: understand the system, then the symptom.
        common: list[tuple[str, dict, str]] = [
            ("get_service_topology", {"namespace": ns},
             "First, establish what this application is supposed to look like and which "
             "services depend on which."),
            ("get_service_health", {"namespace": ns},
             "Now compare that against reality to see which services are actually down."),
            ("list_pods", {"namespace": ns},
             "Look at pod-level state to find which workloads are unhealthy and whether "
             "they are restarting."),
        ]

        broken_pod = self._failing_pod(svc)

        def logs(name: str, note: str) -> tuple[str, dict, str]:
            return ("get_pod_logs", {"namespace": ns, "name": name, "tail_lines": 40}, note)

        match fault:
            case "OOMKilled" | "CascadingDependencyDown":
                need = int(p["required_mi"])
                steps = common + [
                    logs(broken_pod, f"{svc} is restarting; read its logs to see why it dies."),
                    ("query_metrics",
                     {"namespace": ns, "query": f'container_memory_limit_mib{{pod="{broken_pod}"}}'},
                     "Confirm the configured memory ceiling against what the process needs."),
                    ("patch_deployment_resources",
                     {"namespace": ns, "name": svc,
                      "memory_limit_mi": need + 256, "memory_request_mi": need},
                     f"The working set exceeds the limit, so raise {svc}'s memory limit "
                     f"above {need}Mi rather than restarting it."),
                    ("get_service_health", {"namespace": ns},
                     "Verify every service recovered after the change."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc} had a memory limit below its working set and was "
                             f"OOM-killed on start; raised the limit above {need}Mi.")

            case "KVCacheExhaustion":
                need = int(p["needed_limit_mi"])
                steps = common + [
                    logs(broken_pod, f"{svc} is crashlooping; read the engine startup logs."),
                    ("search_runbooks", {"query": "KV cache memory inference"},
                     "Check the runbook for KV cache allocation failures."),
                    ("patch_deployment_resources",
                     {"namespace": ns, "name": svc, "memory_limit_mi": need + 512},
                     "The KV cache no longer fits after weights load, so raise the memory "
                     "limit to cover weights plus cache."),
                    ("get_service_health", {"namespace": ns}, "Confirm the model server is serving."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc}'s KV cache exceeded the memory left after weights "
                             f"loaded; raised the limit to {need + 512}Mi.")

            case "ImagePullBackOff" | "RolloutStuck":
                steps = common + [
                    ("list_events", {"namespace": ns},
                     "Pods are not starting at all; cluster events will say why."),
                    ("get_recent_changes", {"namespace": ns},
                     "Check what shipped recently to find the previous good image."),
                    ("set_image", {"namespace": ns, "name": svc, "image": p["good_image"]},
                     f"The rollout points at a tag that does not exist; put {svc} back on "
                     "the last known-good image."),
                    ("get_service_health", {"namespace": ns}, "Verify the rollout completed."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc} was deployed with a nonexistent image tag; rolled back "
                             f"to {p['good_image']}.")

            case "BadConfigMap":
                steps = common + [
                    logs(broken_pod, f"{svc} exits immediately; the logs name the bad setting."),
                    ("get_configmap", {"namespace": ns, "name": p["configmap"]},
                     "Read the ConfigMap the app validates at startup."),
                    ("patch_configmap",
                     {"namespace": ns, "name": p["configmap"], "key": p["key"],
                      "value": p["good_value"] or "restored"},
                     f"Key {p['key']} is empty; restore its value."),
                    ("get_service_health", {"namespace": ns}, "Confirm the service starts cleanly."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"ConfigMap {p['configmap']} key {p['key']} was blank, failing "
                             f"{svc}'s startup validation; restored it.")

            case "MissingSecret":
                steps = common + [
                    ("describe_pod", {"namespace": ns, "name": broken_pod},
                     "The container never starts, so inspect what it is trying to mount."),
                    ("get_secret_meta", {"namespace": ns},
                     "List the Secrets that actually exist to compare against the reference."),
                    ("patch_deployment_secret_ref",
                     {"namespace": ns, "name": svc, "secret_name": p["good_secret"]},
                     f"The deployment references a Secret that does not exist; repoint it "
                     f"at {p['good_secret']}."),
                    ("get_service_health", {"namespace": ns}, "Confirm the container starts."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc} referenced nonexistent secret {p['bad_secret']}; "
                             f"repointed it to {p['good_secret']}.")

            case "ServiceSelectorMismatch":
                steps = common + [
                    ("get_service", {"namespace": ns, "name": p["service"]},
                     "Pods look healthy, so check whether the Service is actually selecting them."),
                    ("patch_service_selector",
                     {"namespace": ns, "name": p["service"], "selector": p["good_selector"]},
                     "The selector matches no pod labels; point it back at the real workload."),
                    ("get_service_health", {"namespace": ns}, "Confirm endpoints are populated."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"Service {p['service']} selected labels no pod carries, leaving "
                             "it with zero endpoints; restored the selector.")

            case "ReadinessProbeMisconfig":
                steps = common + [
                    logs(broken_pod, "Pods run but never go Ready; the logs show the probe result."),
                    ("get_deployment", {"namespace": ns, "name": p["service"]},
                     "Compare the configured readiness probe against the port the app serves."),
                    ("patch_deployment_probe",
                     {"namespace": ns, "name": p["service"],
                      "path": p["good_path"], "port": int(p["good_port"])},
                     "The probe targets an endpoint the app does not serve; realign it."),
                    ("get_service_health", {"namespace": ns}, "Confirm pods become Ready."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{p['service']}'s readiness probe pointed at "
                             f"{p['bad_path']}:{p['bad_port']} instead of "
                             f"{p['good_path']}:{p['good_port']}.")

            case "ResourceQuotaExhausted":
                total = sum(s.replicas for s in ep.topology.services)
                steps = common + [
                    ("list_events", {"namespace": ns},
                     "Replicas are missing; events will show whether creation was rejected."),
                    ("get_resource_quota", {"namespace": ns},
                     "Check the namespace quota against the total desired replicas."),
                    ("patch_resource_quota",
                     {"namespace": ns, "name": "namespace-quota", "hard": {"pods": str(total + 2)}},
                     "The quota caps pods below what the deployments need; raise it."),
                    ("get_service_health", {"namespace": ns}, "Confirm full capacity returns."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"Namespace quota capped pods below the {total} required; raised it.")

            case "NodeSelectorUnschedulable":
                steps = common + [
                    ("describe_pod", {"namespace": ns, "name": broken_pod},
                     "Pods are Pending; the scheduling condition will name the constraint."),
                    ("patch_deployment_node_selector",
                     {"namespace": ns, "name": svc, "node_selector": p["good_selector"]},
                     "The nodeSelector matches no node in this cluster; clear it."),
                    ("get_service_health", {"namespace": ns}, "Confirm the pods schedule and start."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc} was pinned to a node tier that does not exist; "
                             "removed the constraint.")

            case "NetworkPolicyDeny":
                steps = common + [
                    ("trace_request", {"namespace": ns},
                     "Every pod is Ready, so trace a request to find where it actually fails."),
                    ("list_network_policies", {"namespace": ns},
                     "Check whether a NetworkPolicy is blocking the dependency."),
                    ("delete_network_policy", {"namespace": ns, "name": p["policy"]},
                     "The policy admits only a tier that is not deployed; remove it."),
                    ("get_service_health", {"namespace": ns}, "Confirm connectivity is restored."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"NetworkPolicy {p['policy']} blocked all ingress to "
                             f"{p['target']}; removed it.")

            case "ReplicasScaledToZero":
                steps = common + [
                    ("get_deployment", {"namespace": ns, "name": svc},
                     "The service has no endpoints; check its desired replica count."),
                    ("get_recent_changes", {"namespace": ns},
                     "Confirm from deploy history that this was a recent change."),
                    ("scale_deployment",
                     {"namespace": ns, "name": svc, "replicas": int(p["good_replicas"])},
                     f"{svc} was scaled to zero; restore it to {p['good_replicas']} replicas."),
                    ("get_service_health", {"namespace": ns}, "Confirm the service is serving."),
                ]
                return _Plan(steps, ep.fault.root_cause,
                             f"{svc} had been scaled to zero replicas; scaled it back to "
                             f"{p['good_replicas']}.")

            case "NodeNotReady":
                # Repairing the node is a cluster-admin action, but the workload-side
                # remediation is the real on-call move: force a fresh rollout so the
                # scheduler places the pods on nodes that are still Ready. The dead node
                # is tainted NoSchedule, so replacements land elsewhere.
                stranded = self._stranded_services(p.get("node", ""))
                steps = common + [
                    ("list_events", {"namespace": ns},
                     "Capacity dropped without a config change; check for node-level events."),
                    ("search_runbooks", {"query": "node NotReady pods unknown capacity"},
                     "Confirm the standard response to a lost node."),
                ]
                for name in stranded:
                    steps.append((
                        "rollout_restart", {"namespace": ns, "name": name},
                        f"{name} has pods stranded on the failed node; restart the rollout "
                        "so they are rescheduled onto healthy nodes.",
                    ))
                steps.append(("get_service_health", {"namespace": ns},
                              "Confirm full capacity is restored on the remaining nodes."))
                return _Plan(steps, ep.fault.root_cause,
                             f"Node {p.get('node')} stopped reporting Ready, stranding its "
                             "pods; rescheduled the affected workloads onto healthy nodes.")

            case _:
                return _Plan(common, ep.fault.root_cause, "Investigated service degradation.")

    def _stranded_services(self, node: str) -> list[str]:
        """Services with at least one pod on the failed node."""
        try:
            pods = self.episode.kube.core.list_namespaced_pod(self.episode.namespace).items
        except Exception:
            return []
        names = {
            (pod.metadata.labels or {}).get("app")
            for pod in pods if pod.spec.node_name == node
        }
        return sorted(n for n in names if n)

    def _failing_pod(self, service: str) -> str:
        """Name a pod belonging to `service`, preferring an unhealthy one."""
        try:
            pods = self.episode.kube.core.list_namespaced_pod(
                self.episode.namespace, label_selector=f"app={service}"
            ).items
        except Exception:
            pods = []
        if not pods:
            return f"{service}-0"
        for pod in pods:
            cs = (pod.status.container_statuses or [None])[0]
            if cs is not None and not cs.ready:
                return pod.metadata.name
        return pods[0].metadata.name


__all__ = ["ExpertPolicy", "json"]
