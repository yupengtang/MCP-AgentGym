"""Typed access to a ControlPlane via the official Kubernetes Python client."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from kubernetes import client as k8s
from kubernetes import config as k8s_config


@dataclass
class KubeClient:
    """Thin façade so the rest of the codebase never imports `kubernetes` directly."""

    kubeconfig: Path

    @cached_property
    def api_client(self) -> k8s.ApiClient:
        cfg = k8s.Configuration()
        k8s_config.load_kube_config(config_file=str(self.kubeconfig), client_configuration=cfg)
        return k8s.ApiClient(configuration=cfg)

    @cached_property
    def core(self) -> k8s.CoreV1Api:
        return k8s.CoreV1Api(self.api_client)

    @cached_property
    def apps(self) -> k8s.AppsV1Api:
        return k8s.AppsV1Api(self.api_client)

    @cached_property
    def networking(self) -> k8s.NetworkingV1Api:
        return k8s.NetworkingV1Api(self.api_client)

    @cached_property
    def autoscaling(self) -> k8s.AutoscalingV2Api:
        return k8s.AutoscalingV2Api(self.api_client)

    @cached_property
    def custom(self) -> k8s.CustomObjectsApi:
        return k8s.CustomObjectsApi(self.api_client)

    @cached_property
    def apiextensions(self) -> k8s.ApiextensionsV1Api:
        return k8s.ApiextensionsV1Api(self.api_client)

    @cached_property
    def rbac(self) -> k8s.RbacAuthorizationV1Api:
        return k8s.RbacAuthorizationV1Api(self.api_client)

    def close(self) -> None:
        if "api_client" in self.__dict__:
            self.api_client.close()


# --- quantity parsing --------------------------------------------------------

_BIN = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}
_DEC = {"k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_quantity(value: str | int | float | None) -> float:
    """Parse a Kubernetes resource quantity into a plain float (bytes, or cores).

    Handles the binary suffixes (Ki/Mi/Gi/Ti), decimal suffixes (k/M/G/T), and the
    milli suffix used for CPU ("500m" -> 0.5).
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    v = value.strip()
    if not v:
        return 0.0
    for suffix, mult in _BIN.items():
        if v.endswith(suffix):
            return float(v[: -len(suffix)]) * mult
    if v.endswith("m"):
        return float(v[:-1]) / 1000.0
    for suffix, mult in _DEC.items():
        if v.endswith(suffix):
            return float(v[: -len(suffix)]) * mult
    return float(v)


def mib(value: str | int | float | None) -> float:
    """Parse a memory quantity and return mebibytes."""
    return parse_quantity(value) / (1024**2)
