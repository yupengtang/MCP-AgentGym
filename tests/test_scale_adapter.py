"""The adapter scaling tool, which every frontier number depends on.

The attenuation result is "scaling the LoRA update by s moves the model along the
capability/generalization trade". If the tool does not scale by exactly s, or scales
something it should not, the dose-response curve is measuring the tool rather than the
model -- and it would still come out monotone, because a monotone bug in the scale factor
produces a monotone curve. So the arithmetic is asserted directly rather than inferred
from the shape of the result.

Synthetic adapters, not real ones: an 8B adapter is ~160 MB on shared storage and these
assertions are about arithmetic, not about any particular checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from scale_adapter import scale_adapter  # noqa: E402

R, DIM, LAYERS = 4, 8, 6


def _make_adapter(tmp: Path) -> Path:
    """A minimal peft-shaped adapter: one q_proj A/B pair per block."""
    src = tmp / "src"
    src.mkdir()
    tensors = {}
    torch.manual_seed(0)
    for i in range(LAYERS):
        stem = f"base_model.model.model.layers.{i}.self_attn.q_proj"
        tensors[f"{stem}.lora_A.weight"] = torch.randn(R, DIM)
        tensors[f"{stem}.lora_B.weight"] = torch.randn(DIM, R)
    safetensors.save_file(tensors, str(src / "adapter_model.safetensors"),
                          metadata={"format": "pt"})
    (src / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": 2 * R}))
    return src


def _load(d: Path) -> dict:
    return safetensors.load_file(str(d / "adapter_model.safetensors"))


@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
def test_lora_b_is_scaled_exactly_and_lora_a_is_untouched(tmp_path, scale):
    """dW = (alpha/r) B@A, so scaling B by s scales dW by exactly s.

    Scaling both factors would apply s twice and quietly square the sweep's axis: the
    curve would still be monotone and every reported scale would be wrong.
    """
    src = _make_adapter(tmp_path)
    before = _load(src)
    n = scale_adapter(src, tmp_path / "dst", scale)
    after = _load(tmp_path / "dst")

    assert n == LAYERS, f"expected {LAYERS} lora_B tensors scaled, got {n}"
    for name, t in before.items():
        if "lora_B" in name:
            assert torch.allclose(after[name], t * scale, atol=1e-6), name
        else:
            assert torch.equal(after[name], t), f"{name} must not be touched"


def test_scaling_is_recorded_so_a_copy_cannot_pass_as_trained(tmp_path):
    src = _make_adapter(tmp_path)
    scale_adapter(src, tmp_path / "dst", 0.5)
    meta = json.loads((tmp_path / "dst" / "scaling.json").read_text())
    assert meta["scale"] == 0.5
    assert meta["scaled_tensors"] == LAYERS
    assert str(src) in meta["source_adapter"]


def test_banded_scaling_touches_only_its_blocks(tmp_path):
    """Scaling a band localises the mechanism; scaling the wrong blocks misattributes it."""
    src = _make_adapter(tmp_path)
    before = _load(src)
    band = range(2, 5)                                  # blocks 2, 3, 4
    n = scale_adapter(src, tmp_path / "dst", 0.5, layers=band)
    after = _load(tmp_path / "dst")

    assert n == len(band)
    for name, t in before.items():
        if "lora_B" not in name:
            continue
        idx = int(name.split(".layers.")[1].split(".")[0])
        expected = t * 0.5 if idx in band else t
        assert torch.allclose(after[name], expected, atol=1e-6), (
            f"block {idx} {'should' if idx in band else 'should not'} have been scaled")

    meta = json.loads((tmp_path / "dst" / "scaling.json").read_text())
    assert meta["layers"] == [2, 5]
    assert meta["unscaled_out_of_band"] == LAYERS - len(band)


def test_refuses_an_adapter_with_no_lora_b(tmp_path):
    """Silently writing an unscaled copy would put a mislabelled arm in the sweep."""
    src = tmp_path / "empty"
    src.mkdir()
    safetensors.save_file({"something.else.weight": torch.zeros(2, 2)},
                          str(src / "adapter_model.safetensors"), metadata={"format": "pt"})
    (src / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": 2 * R}))
    with pytest.raises(SystemExit, match="no lora_B"):
        scale_adapter(src, tmp_path / "dst", 0.5)


def test_frobenius_norm_scales_by_exactly_s(tmp_path):
    """The end-to-end property the frontier assumes: ||dW|| is linear in s.

    Computed the way adapter_norms.py computes it, so the two tools are checked against
    each other rather than each against itself.
    """
    src = _make_adapter(tmp_path)

    def total_norm(d: Path) -> float:
        t = _load(d)
        cfg = json.loads((d / "adapter_config.json").read_text())
        scaling = cfg["lora_alpha"] / cfg["r"]
        tot = 0.0
        for name in t:
            if "lora_A" not in name:
                continue
            a = t[name].double()
            b = t[name.replace("lora_A", "lora_B")].double()
            gram = (b.T @ b) @ (a @ a.T)
            tot += (scaling ** 2) * float(torch.diagonal(gram).sum())
        return tot ** 0.5

    ref = total_norm(src)
    for s in (0.25, 0.5, 0.75):
        scale_adapter(src, tmp_path / f"d{s}", s)
        got = total_norm(tmp_path / f"d{s}")
        # rel=1e-6, not tighter: the product is computed in fp64 but the scaled tensors
        # are *stored* as fp32, so the round-trip cannot beat fp32 epsilon (~1e-7) no
        # matter how exact the arithmetic is. A tolerance below that tests the storage
        # format rather than the tool, and fails on a correct implementation.
        assert got == pytest.approx(s * ref, rel=1e-6), f"s={s}: {got} vs {s * ref}"
