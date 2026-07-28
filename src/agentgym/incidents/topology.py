"""Microservice topologies that incidents are injected into.

A topology is a declarative description of a small production-shaped application. It
renders to ordinary Kubernetes manifests; fault families then mutate those manifests
before they are applied. Keeping topology and fault separate is what lets the same fault
appear in unfamiliar surroundings, which is how the held-out generalization splits are
built.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

ANN = "agentgym.io"


@dataclass(frozen=True)
class ServiceDef:
    name: str
    image: str
    replicas: int = 2
    # Physical behaviour consumed by kubelet_sim
    working_set_mi: int = 512
    boot_ticks: int = 1
    health_path: str = "/healthz"
    health_port: int = 8080
    depends_on: tuple[str, ...] = ()
    config_ref: tuple[str, tuple[str, ...]] | None = None
    secret_ref: str | None = None
    kv_cache_mi: int = 0
    # Scheduling
    memory_limit_mi: int = 2048
    memory_request_mi: int = 512
    cpu_request_m: int = 250
    gpu: int = 0
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: tuple[tuple[str, str, str], ...] = ()
    with_readiness_probe: bool = True
    expose: bool = True
    # Stateful services render as a StatefulSet with a volumeClaimTemplate rather than a
    # Deployment: it is what a datastore actually is, and it is what makes storage a real
    # failure surface (an unbound PVC keeps the pod Pending forever).
    stateful: bool = False
    storage_gi: int = 0
    storage_class: str = "standard"


@dataclass(frozen=True)
class ConfigMapDef:
    name: str
    data: dict[str, str]


@dataclass(frozen=True)
class SecretDef:
    name: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Topology:
    name: str
    description: str
    services: tuple[ServiceDef, ...]
    configmaps: tuple[ConfigMapDef, ...] = ()
    secrets: tuple[SecretDef, ...] = ()
    entrypoint: str = ""

    def service(self, name: str) -> ServiceDef:
        for s in self.services:
            if s.name == name:
                return s
        raise KeyError(f"{name} not in topology {self.name}")

    def with_service(self, svc: ServiceDef) -> Topology:
        return replace(self, services=tuple(
            svc if s.name == svc.name else s for s in self.services
        ))

    def render(self) -> list[dict[str, Any]]:
        """Render to Kubernetes manifests, in apply order."""
        out: list[dict[str, Any]] = [
            # A dedicated identity per incident namespace: the agent's RBAC is bound to
            # this, not to cluster-admin, so a permission bug shows up as a Forbidden
            # rather than as silent success.
            {
                "apiVersion": "v1", "kind": "ServiceAccount",
                "metadata": {"name": "workload"},
            },
            # Defaults for anything that forgets to declare resources. Without a
            # LimitRange a container with no limits is unbounded, and the OOM fault
            # family would be unreachable for that pod.
            #
            # The default *limit* must sit above the largest request any service makes.
            # A LimitRange default lower than an existing request is not clamped -- the
            # API server rejects the pod at admission ("requests: Invalid value: 2: must
            # be less than or equal to cpu limit of 500m"), and the ReplicaSet then fails
            # to create anything at all. The default exists to bound containers that
            # declare nothing, not to constrain ones that do.
            {
                "apiVersion": "v1", "kind": "LimitRange",
                "metadata": {"name": "defaults"},
                "spec": {"limits": [{
                    "type": "Container",
                    "default": {"memory": "32Gi", "cpu": "4"},
                    "defaultRequest": {"memory": "128Mi", "cpu": "100m"},
                }]},
            },
        ]
        for cm in self.configmaps:
            out.append({
                "apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": cm.name},
                "data": dict(cm.data),
            })
        for sec in self.secrets:
            out.append({
                "apiVersion": "v1", "kind": "Secret",
                "metadata": {"name": sec.name},
                "type": "Opaque",
                # Values are placeholders; get_secret_meta never returns them.
                "stringData": {k: f"redacted-{k.lower()}" for k in sec.keys},
            })
        for svc in self.services:
            out.append(_statefulset(svc) if svc.stateful else _deployment(svc))
            if svc.expose:
                out.append(_service(svc))
        return out


def _annotations(svc: ServiceDef) -> dict[str, str]:
    ann = {
        f"{ANN}/working-set-mi": str(svc.working_set_mi),
        f"{ANN}/boot-ticks": str(svc.boot_ticks),
        f"{ANN}/health-path": svc.health_path,
        f"{ANN}/health-port": str(svc.health_port),
    }
    if svc.depends_on:
        ann[f"{ANN}/depends-on"] = ",".join(svc.depends_on)
    if svc.config_ref:
        name, keys = svc.config_ref
        ann[f"{ANN}/config-ref"] = f"{name}:{','.join(keys)}"
    if svc.secret_ref:
        ann[f"{ANN}/secret-ref"] = svc.secret_ref
    if svc.kv_cache_mi:
        ann[f"{ANN}/kv-cache-mi"] = str(svc.kv_cache_mi)
    return ann


def _pod_template(svc: ServiceDef) -> dict[str, Any]:
    """The pod template both Deployments and StatefulSets share."""
    dep = _deployment(svc)
    return dep["spec"]["template"]


def _statefulset(svc: ServiceDef) -> dict[str, Any]:
    template = _pod_template(svc)
    if svc.storage_gi:
        template["spec"]["containers"][0]["volumeMounts"] = [
            {"name": "data", "mountPath": "/var/lib/data"}
        ]
    body: dict[str, Any] = {
        "apiVersion": "apps/v1", "kind": "StatefulSet",
        "metadata": {"name": svc.name, "labels": {"app": svc.name}},
        "spec": {
            "serviceName": svc.name,
            "replicas": svc.replicas,
            "selector": {"matchLabels": {"app": svc.name}},
            "template": template,
        },
    }
    if svc.storage_gi:
        body["spec"]["volumeClaimTemplates"] = [{
            "metadata": {"name": "data"},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": svc.storage_class,
                "resources": {"requests": {"storage": f"{svc.storage_gi}Gi"}},
            },
        }]
    return body


def _deployment(svc: ServiceDef) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": svc.name,
        "image": svc.image,
        "ports": [{"name": "http", "containerPort": svc.health_port}],
        "resources": {
            "limits": {"memory": f"{svc.memory_limit_mi}Mi"},
            "requests": {
                "memory": f"{svc.memory_request_mi}Mi",
                "cpu": f"{svc.cpu_request_m}m",
            },
        },
    }
    if svc.gpu:
        container["resources"]["limits"]["nvidia.com/gpu"] = str(svc.gpu)
    if svc.with_readiness_probe:
        container["readinessProbe"] = {
            "httpGet": {"path": svc.health_path, "port": svc.health_port},
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
            "failureThreshold": 3,
        }
    if svc.secret_ref:
        container["envFrom"] = [{"secretRef": {"name": svc.secret_ref}}]
    if svc.config_ref:
        container.setdefault("envFrom", []).append(
            {"configMapRef": {"name": svc.config_ref[0]}}
        )

    pod_spec: dict[str, Any] = {
        "containers": [container],
        "serviceAccountName": "workload",
    }
    if svc.node_selector:
        pod_spec["nodeSelector"] = dict(svc.node_selector)
    if svc.tolerations:
        pod_spec["tolerations"] = [
            {"key": k, "value": v, "effect": e, "operator": "Equal"}
            for k, v, e in svc.tolerations
        ]

    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": svc.name, "labels": {"app": svc.name}},
        "spec": {
            "replicas": svc.replicas,
            "selector": {"matchLabels": {"app": svc.name}},
            "template": {
                "metadata": {"labels": {"app": svc.name}, "annotations": _annotations(svc)},
                "spec": pod_spec,
            },
        },
    }


def _service(svc: ServiceDef) -> dict[str, Any]:
    return {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": svc.name, "labels": {"app": svc.name}},
        "spec": {
            "selector": {"app": svc.name},
            "ports": [{"name": "http", "port": 80, "targetPort": svc.health_port}],
        },
    }


# --- the catalog of topologies ----------------------------------------------

REDIS = ServiceDef(
    name="redis", image="redis:7.2-alpine", replicas=1,
    working_set_mi=256, memory_limit_mi=1024, memory_request_mi=256,
    health_path="/ping", health_port=6379, cpu_request_m=100,
    stateful=True, storage_gi=2,
)
POSTGRES = ServiceDef(
    name="postgres", image="postgres:16-alpine", replicas=1,
    working_set_mi=512, memory_limit_mi=2048, memory_request_mi=512,
    health_path="/ready", health_port=5432, cpu_request_m=250,
    secret_ref="postgres-credentials",
    stateful=True, storage_gi=8,
)

ML_INFERENCE = Topology(
    name="ml-inference",
    description="A GPU model server behind a gateway, with a Redis response cache.",
    entrypoint="inference-gateway",
    services=(
        ServiceDef(
            name="inference-gateway", image="ghcr.io/agentgym/gateway:0.9.1", replicas=2,
            working_set_mi=384, memory_limit_mi=1024, memory_request_mi=384,
            depends_on=("model-service",), health_port=8080,
            config_ref=("gateway-config", ("UPSTREAM_URL", "REQUEST_TIMEOUT_MS")),
        ),
        ServiceDef(
            name="model-service", image="vllm/vllm-openai:v0.6.0", replicas=1,
            working_set_mi=6144, memory_limit_mi=16384, memory_request_mi=6144,
            kv_cache_mi=4096, gpu=1, boot_ticks=2, health_port=8000,
            cpu_request_m=2000, depends_on=("redis",),
            node_selector={"tier": "gpu"},
            tolerations=(("nvidia.com/gpu", "present", "NoSchedule"),),
            config_ref=("model-config", ("MODEL_PATH", "MAX_MODEL_LEN")),
        ),
        REDIS,
    ),
    configmaps=(
        ConfigMapDef("gateway-config", {
            "UPSTREAM_URL": "http://model-service:80", "REQUEST_TIMEOUT_MS": "30000",
        }),
        ConfigMapDef("model-config", {
            "MODEL_PATH": "/models/qwen3-8b", "MAX_MODEL_LEN": "16384",
        }),
    ),
)

WEB_API = Topology(
    name="web-api",
    description="A public API with a Postgres store and a Redis session cache.",
    entrypoint="edge-proxy",
    services=(
        ServiceDef(
            name="edge-proxy", image="nginx:1.25-alpine", replicas=2,
            working_set_mi=128, memory_limit_mi=512, memory_request_mi=128,
            depends_on=("api-service",), health_path="/nginx-health", health_port=80,
            cpu_request_m=100,
        ),
        ServiceDef(
            name="api-service", image="ghcr.io/agentgym/api-service:1.4.2", replicas=3,
            working_set_mi=768, memory_limit_mi=2048, memory_request_mi=768,
            depends_on=("postgres", "redis"), health_port=8080,
            config_ref=("api-config", ("DATABASE_URL", "CACHE_URL", "LOG_LEVEL")),
            secret_ref="api-credentials",
        ),
        POSTGRES, REDIS,
    ),
    configmaps=(
        ConfigMapDef("api-config", {
            "DATABASE_URL": "postgres://postgres:5432/app",
            "CACHE_URL": "redis://redis:6379/0",
            "LOG_LEVEL": "info",
        }),
    ),
    secrets=(
        SecretDef("api-credentials", ("API_SIGNING_KEY",)),
        SecretDef("postgres-credentials", ("POSTGRES_PASSWORD",)),
    ),
)

BATCH_PIPELINE = Topology(
    name="batch-pipeline",
    description="A queue-driven worker fleet writing results to Postgres.",
    entrypoint="ingest-api",
    services=(
        ServiceDef(
            name="ingest-api", image="ghcr.io/agentgym/api-service:1.4.3", replicas=2,
            working_set_mi=512, memory_limit_mi=1536, memory_request_mi=512,
            depends_on=("redis",), health_port=8080,
            config_ref=("ingest-config", ("QUEUE_URL", "BATCH_SIZE")),
        ),
        ServiceDef(
            name="batch-worker", image="ghcr.io/agentgym/worker:2.1.0", replicas=4,
            working_set_mi=1024, memory_limit_mi=2048, memory_request_mi=1024,
            depends_on=("redis", "postgres"), boot_ticks=2, health_port=9090,
            cpu_request_m=500,
            config_ref=("worker-config", ("CONCURRENCY", "RESULT_TABLE")),
            secret_ref="postgres-credentials",
        ),
        POSTGRES, REDIS,
    ),
    configmaps=(
        ConfigMapDef("ingest-config", {"QUEUE_URL": "redis://redis:6379/1", "BATCH_SIZE": "64"}),
        ConfigMapDef("worker-config", {"CONCURRENCY": "8", "RESULT_TABLE": "results"}),
    ),
    secrets=(SecretDef("postgres-credentials", ("POSTGRES_PASSWORD",)),),
)

MULTI_TENANT_SERVING = Topology(
    name="multi-tenant-serving",
    description="Two model replicas served behind one router, sharing a cache tier.",
    entrypoint="model-router",
    services=(
        ServiceDef(
            name="model-router", image="ghcr.io/agentgym/gateway:0.9.1", replicas=2,
            working_set_mi=384, memory_limit_mi=1024, memory_request_mi=384,
            depends_on=("model-primary", "model-canary"), health_port=8080,
            config_ref=("router-config", ("PRIMARY_URL", "CANARY_URL", "CANARY_WEIGHT")),
        ),
        ServiceDef(
            name="model-primary", image="vllm/vllm-openai:v0.6.1", replicas=1,
            working_set_mi=6144, memory_limit_mi=16384, memory_request_mi=6144,
            kv_cache_mi=4096, gpu=1, boot_ticks=2, health_port=8000, cpu_request_m=2000,
            node_selector={"tier": "gpu"},
            tolerations=(("nvidia.com/gpu", "present", "NoSchedule"),),
            config_ref=("model-config", ("MODEL_PATH", "MAX_MODEL_LEN")),
        ),
        ServiceDef(
            name="model-canary", image="vllm/vllm-openai:v0.6.1", replicas=1,
            working_set_mi=6144, memory_limit_mi=12288, memory_request_mi=6144,
            kv_cache_mi=2048, gpu=1, boot_ticks=2, health_port=8000, cpu_request_m=2000,
            node_selector={"tier": "gpu"},
            tolerations=(("nvidia.com/gpu", "present", "NoSchedule"),),
            config_ref=("model-config", ("MODEL_PATH", "MAX_MODEL_LEN")),
        ),
        REDIS,
    ),
    configmaps=(
        ConfigMapDef("router-config", {
            "PRIMARY_URL": "http://model-primary:80",
            "CANARY_URL": "http://model-canary:80",
            "CANARY_WEIGHT": "10",
        }),
        ConfigMapDef("model-config", {
            "MODEL_PATH": "/models/qwen3-8b", "MAX_MODEL_LEN": "16384",
        }),
    ),
)

EVENT_STREAM = Topology(
    name="event-stream",
    description="A collector feeding a processor tier, with Redis buffering.",
    entrypoint="collector",
    services=(
        ServiceDef(
            name="collector", image="ghcr.io/agentgym/api-service:1.4.2", replicas=3,
            working_set_mi=512, memory_limit_mi=1536, memory_request_mi=512,
            depends_on=("redis",), health_port=8080,
            config_ref=("collector-config", ("BUFFER_URL", "FLUSH_INTERVAL_MS")),
        ),
        ServiceDef(
            name="processor", image="ghcr.io/agentgym/worker:2.1.0", replicas=2,
            working_set_mi=1536, memory_limit_mi=3072, memory_request_mi=1536,
            depends_on=("redis", "postgres"), boot_ticks=2, health_port=9090,
            cpu_request_m=750,
            config_ref=("processor-config", ("WINDOW_SECONDS", "SINK_TABLE")),
            secret_ref="postgres-credentials",
        ),
        POSTGRES, REDIS,
    ),
    configmaps=(
        ConfigMapDef("collector-config", {
            "BUFFER_URL": "redis://redis:6379/2", "FLUSH_INTERVAL_MS": "500",
        }),
        ConfigMapDef("processor-config", {"WINDOW_SECONDS": "60", "SINK_TABLE": "events"}),
    ),
    secrets=(SecretDef("postgres-credentials", ("POSTGRES_PASSWORD",)),),
)

ALL_TOPOLOGIES: tuple[Topology, ...] = (
    ML_INFERENCE, WEB_API, BATCH_PIPELINE, MULTI_TENANT_SERVING, EVENT_STREAM,
)
TOPOLOGIES_BY_NAME = {t.name: t for t in ALL_TOPOLOGIES}

__all__ = [
    "Topology", "ServiceDef", "ConfigMapDef", "SecretDef",
    "ALL_TOPOLOGIES", "TOPOLOGIES_BY_NAME",
]
