"""What the Helm chart actually renders, versus what the documents claim it renders.

`helm lint` passes on a chart whose values are consumed by nothing, which is how
`values-eks.yaml` carried a full `aws.storageClass` block -- and DESIGN.md §9b claimed a
gp3 StorageClass backed the datastore StatefulSets -- while no template produced one.
Linting a chart proves it is well-formed; only rendering it proves it does something.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

CHART = "deploy/helm/agentgym"
EKS_VALUES = f"{CHART}/values-eks.yaml"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH (it is added by env.sh)")


def _render(*extra: str) -> str:
    r = subprocess.run(["helm", "template", "agentgym", CHART, *extra],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"helm template failed:\n{r.stderr[:800]}"
    return r.stdout


def test_chart_lints():
    r = subprocess.run(["helm", "lint", CHART], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_default_values_render_the_core_objects():
    out = _render()
    for kind in ("ServiceAccount", "ClusterRole", "ClusterRoleBinding",
                 "Deployment", "Service", "NetworkPolicy", "HorizontalPodAutoscaler"):
        assert f"kind: {kind}" in out, f"default render is missing {kind}"


def test_default_values_do_not_render_cloud_objects():
    """A plain install must not create AWS-specific objects."""
    out = _render()
    assert "kind: StorageClass" not in out
    assert "eks.amazonaws.com/role-arn" not in out


def test_eks_overlay_renders_the_gp3_storageclass_it_documents():
    """Regression: the block existed in values and in the docs, and rendered nothing."""
    out = _render("-f", EKS_VALUES)
    assert "kind: StorageClass" in out, (
        "values-eks.yaml defines aws.storageClass and DESIGN.md claims a gp3 StorageClass, "
        "but nothing renders one")
    assert "provisioner: ebs.csi.aws.com" in out
    assert "type: \"gp3\"" in out or "type: gp3" in out


def test_eks_storageclass_binds_late():
    """Immediate binding puts the volume in an arbitrary AZ.

    An incident namespace's PVC has to bind where its pod is scheduled; on a multi-AZ node
    group, Immediate binding yields pods that can never mount their volume.
    """
    out = _render("-f", EKS_VALUES)
    assert "volumeBindingMode: WaitForFirstConsumer" in out


def test_eks_overlay_carries_irsa():
    out = _render("-f", EKS_VALUES)
    assert "eks.amazonaws.com/role-arn" in out, (
        "the overlay claims IRSA; without the annotation the pod falls back to node "
        "credentials, which is the thing IRSA exists to avoid")
