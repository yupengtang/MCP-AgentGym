"""The IncidentRun controller against a real API server.

Asserts the properties that separate a controller from a script: it reaches a terminal
state, it is idempotent, it records results on the status subresource, and its finalizer
cleans up the episode namespace before the object goes away.
"""

from __future__ import annotations

import pytest

from agentgym.incidents.gym import AgentGym
from agentgym.operator import crd as crd_mod
from agentgym.operator.controller import FINALIZER, IncidentRunController


@pytest.fixture(scope="module")
def gym():
    with AgentGym(offset=75) as g:
        yield g


@pytest.fixture(scope="module")
def controller(gym):
    crd_mod.ensure_crd(gym.kube)
    return IncidentRunController(kube=gym.kube, gym=gym, namespace="default")


def _create(kube, name: str, **spec) -> None:
    body = {
        "apiVersion": f"{crd_mod.GROUP}/{crd_mod.VERSION}",
        "kind": crd_mod.KIND,
        "metadata": {"name": name},
        "spec": {"topology": "web-api", "seed": 7, "maxToolCalls": 20, **spec},
    }
    kube.custom.create_namespaced_custom_object(
        crd_mod.GROUP, crd_mod.VERSION, "default", crd_mod.PLURAL, body
    )


def _get(kube, name: str) -> dict:
    return kube.custom.get_namespaced_custom_object(
        crd_mod.GROUP, crd_mod.VERSION, "default", crd_mod.PLURAL, name
    )


def _drive(controller, name: str, max_passes: int = 12) -> dict:
    """Reconcile until the object reaches a terminal phase."""
    for _ in range(max_passes):
        controller.reconcile_once()
        obj = _get(controller.kube, name)
        if (obj.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
            return obj
    return _get(controller.kube, name)


def test_crd_is_installed(gym):
    crd_mod.ensure_crd(gym.kube)
    crd = gym.kube.apiextensions.read_custom_resource_definition(name=crd_mod.CRD_NAME)
    assert crd.spec.names.kind == "IncidentRun"
    assert crd.spec.versions[0].subresources.status is not None


def test_reconcile_runs_and_verifies_an_incident(gym, controller):
    _create(gym.kube, "run-oom", fault="OOMKilled")
    obj = _drive(controller, "run-oom")
    status = obj.get("status") or {}

    assert status.get("phase") == "Succeeded", status
    assert status.get("resolved") is True
    assert status.get("rootCauseCorrect") is True
    assert status["trueRootCause"] == "memory_limit_below_working_set"
    assert int(status.get("toolCalls", 0)) > 0
    assert int(status.get("unsafeActions", 1)) == 0
    assert status.get("episodeNamespace")
    assert FINALIZER in obj["metadata"].get("finalizers", [])


def test_reconcile_is_idempotent(gym, controller):
    """A finished run must not be executed again on the next pass."""
    before = _get(gym.kube, "run-oom")["status"]
    acted = controller.reconcile_once()
    after = _get(gym.kube, "run-oom")["status"]

    assert acted == 0, "a terminal IncidentRun was reconciled again"
    assert before["startedAt"] == after["startedAt"]
    assert before["finishedAt"] == after["finishedAt"]


def test_failed_incident_is_reported_not_hidden(gym, controller):
    """An unresolvable spec still reaches a terminal state with a reason."""
    _create(gym.kube, "run-bogus", fault="ImagePullBackOff", maxToolCalls=1)
    obj = _drive(controller, "run-bogus")
    status = obj.get("status") or {}
    assert status.get("phase") in ("Succeeded", "Failed")
    assert "message" in status


def test_finalizer_cleans_up_the_episode_namespace(gym, controller):
    obj = _get(gym.kube, "run-oom")
    namespace = obj["status"]["episodeNamespace"]

    gym.kube.custom.delete_namespaced_custom_object(
        crd_mod.GROUP, crd_mod.VERSION, "default", crd_mod.PLURAL, "run-oom"
    )
    # The finalizer keeps the object alive until cleanup has run.
    for _ in range(10):
        controller.reconcile_once()
        try:
            _get(gym.kube, "run-oom")
        except Exception:
            break
    else:
        pytest.fail("IncidentRun was never released by its finalizer")

    namespaces = {n.metadata.name for n in gym.kube.core.list_namespace().items
                  if n.status.phase != "Terminating"}
    assert namespace not in namespaces
