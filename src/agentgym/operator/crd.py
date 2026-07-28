"""The IncidentRun CustomResourceDefinition.

Declares a scenario; the controller reconciles it to a verified result. Structural schema
with validation, a status subresource, and printer columns -- so `kubectl get
incidentruns` reads like any other Kubernetes resource.
"""

from __future__ import annotations

from typing import Any

from kubernetes.client.rest import ApiException

GROUP = "agentgym.io"
VERSION = "v1"
PLURAL = "incidentruns"
KIND = "IncidentRun"
CRD_NAME = f"{PLURAL}.{GROUP}"

INCIDENT_RUN_CRD: dict[str, Any] = {
    "apiVersion": "apiextensions.k8s.io/v1",
    "kind": "CustomResourceDefinition",
    "metadata": {"name": CRD_NAME},
    "spec": {
        "group": GROUP,
        "scope": "Namespaced",
        "names": {
            "kind": KIND,
            "listKind": f"{KIND}List",
            "plural": PLURAL,
            "singular": "incidentrun",
            "shortNames": ["ir"],
        },
        "versions": [{
            "name": VERSION,
            "served": True,
            "storage": True,
            "subresources": {"status": {}},
            "additionalPrinterColumns": [
                {"name": "Fault", "type": "string", "jsonPath": ".spec.fault"},
                {"name": "Topology", "type": "string", "jsonPath": ".spec.topology"},
                {"name": "Phase", "type": "string", "jsonPath": ".status.phase"},
                {"name": "Resolved", "type": "boolean", "jsonPath": ".status.resolved"},
                {"name": "Reward", "type": "string", "jsonPath": ".status.reward"},
                {"name": "Age", "type": "date", "jsonPath": ".metadata.creationTimestamp"},
            ],
            "schema": {"openAPIV3Schema": {
                "type": "object",
                "required": ["spec"],
                "properties": {
                    "spec": {
                        "type": "object",
                        "required": ["fault", "topology"],
                        "properties": {
                            "fault": {"type": "string",
                                      "description": "Fault family to inject."},
                            "topology": {"type": "string",
                                         "description": "Application topology."},
                            "seed": {"type": "integer", "default": 0},
                            "policy": {"type": "string", "default": "expert",
                                       "enum": ["expert", "llm"]},
                            "model": {"type": "string", "default": ""},
                            "perturbation": {"type": "string", "default": "clean"},
                            "maxToolCalls": {"type": "integer", "default": 24,
                                             "minimum": 1, "maximum": 200},
                            "ttlSecondsAfterFinished": {"type": "integer", "default": 300},
                        },
                    },
                    "status": {
                        "type": "object",
                        "properties": {
                            "phase": {"type": "string",
                                      "enum": ["Pending", "Injecting", "Running",
                                               "Verifying", "Succeeded", "Failed"]},
                            "resolved": {"type": "boolean"},
                            "rootCauseCorrect": {"type": "boolean"},
                            "predictedRootCause": {"type": "string"},
                            "trueRootCause": {"type": "string"},
                            "reward": {"type": "string"},
                            "toolCalls": {"type": "integer"},
                            "unsafeActions": {"type": "integer"},
                            "episodeNamespace": {"type": "string"},
                            "trajectoryPath": {"type": "string"},
                            "message": {"type": "string"},
                            "startedAt": {"type": "string"},
                            "finishedAt": {"type": "string"},
                            "observedGeneration": {"type": "integer"},
                            "conditions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "status": {"type": "string"},
                                        "reason": {"type": "string"},
                                        "message": {"type": "string"},
                                        "lastTransitionTime": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            }},
        }],
    },
}


def ensure_crd(kube) -> bool:
    """Install the CRD if absent. Returns True when it was created."""
    try:
        kube.apiextensions.create_custom_resource_definition(body=INCIDENT_RUN_CRD)
        return True
    except ApiException as exc:
        if exc.status == 409:
            return False
        raise


__all__ = ["INCIDENT_RUN_CRD", "ensure_crd", "GROUP", "VERSION", "PLURAL",
           "KIND", "CRD_NAME"]
