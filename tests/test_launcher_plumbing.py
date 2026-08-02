"""Every knob a trainer accepts must actually be reachable from its launcher.

This exists because of a defect that invalidated a published finding. All three trainers
take `--seed` and use it. Only one launcher forwarded it. Three jobs were submitted with
`--export=ALL,SEED=1` and `SEED=2`, every one of them trained at seed 0, and the 15.6 pp
gap between those adapters was written up as *seed variance* — the largest effect in the
project — when it was a learning-rate difference (5e-5 against 1e-4).

An environment variable nobody reads is indistinguishable from one that works. The
experiment ran, produced numbers, and answered a question nobody had asked.

This is the same shape as three other defects found in this repository: a
`--max-prompt-length` that reached no code path, a `values-eks.yaml` block consumed by no
template, and a `--settle-timeout` counter that summed the wrong thing. None errored. All
of them described a system that was not the one running.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (launcher, trainer module) — the pairs where a mismatch has consequences for a result.
PAIRS = [
    ("slurm/train_sft.sbatch", "src/agentgym/train/sft_train.py"),
    ("slurm/train_dpo.sbatch", "src/agentgym/train/dpo_train.py"),
    ("slurm/train_grpo_action.sbatch", "src/agentgym/train/grpo_train.py"),
]

# Arguments whose silent absence changes what a run means rather than merely how it runs.
# `--seed` is the one that already caused a retraction; the others are here because a run
# that quietly used the wrong value for any of them is not the run its name claims.
RESULT_BEARING = {"--seed", "--lr", "--epochs"}


def _without_comments(sh: str) -> str:
    """Shell comments stripped, so a flag *mentioned* in prose cannot satisfy the check.

    The first version of this test searched the raw file. The comment above the fix in
    `train_sft.sbatch` contains the words `--seed`, so deleting the actual argument left
    the test passing. A guard that a docstring can satisfy protects nothing.
    """
    out = []
    for line in sh.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _existing_pairs():
    return [(a, b) for a, b in PAIRS
            if (ROOT / a).exists() and (ROOT / b).exists()]


@pytest.mark.parametrize("launcher,trainer", _existing_pairs())
def test_result_bearing_args_are_forwarded(launcher, trainer):
    """A trainer argument that changes the meaning of a result must reach it."""
    trainer_src = (ROOT / trainer).read_text()
    launcher_src = _without_comments((ROOT / launcher).read_text())

    accepted = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', trainer_src))
    relevant = accepted & RESULT_BEARING
    assert relevant, f"{trainer} accepts none of {sorted(RESULT_BEARING)}; check the pattern"

    missing = sorted(a for a in relevant if a not in launcher_src)
    assert not missing, (
        f"{trainer} accepts {missing} and {launcher} never passes them. Exporting the "
        f"corresponding environment variable would change nothing while appearing to work.")


@pytest.mark.parametrize("launcher,trainer", _existing_pairs())
def test_forwarded_args_carry_a_default(launcher, trainer):
    """`--seed "$SEED"` with SEED unset passes an empty string and the run dies at parse.

    The `${SEED:-0}` form is what makes the launcher usable both with and without the
    variable, which is how it is invoked in practice.
    """
    launcher_src = _without_comments((ROOT / launcher).read_text())
    for arg in sorted(RESULT_BEARING):
        for m in re.finditer(rf'{re.escape(arg)}\s+"([^"]*)"', launcher_src):
            val = m.group(1)
            if val.startswith("${") and ":-" not in val:
                pytest.fail(f"{launcher} passes {arg} as {val} with no default; "
                            "an unset variable becomes an empty argument")


def test_sft_launcher_echoes_its_effective_configuration():
    """The log must record what a run actually used, not what was requested.

    Reconstructing the learning rate of an already-trained adapter required loading its
    `training_args.bin`. That worked, but only because the file happened to be kept.
    """
    src = (ROOT / "slurm/train_sft.sbatch").read_text()
    assert "effective SFT configuration" in src
    for knob in ("EPOCHS=", "LR=", "SEED="):
        assert knob in src, f"the effective-config echo omits {knob}"
