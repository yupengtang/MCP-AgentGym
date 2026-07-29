"""The CLI must import and every command must resolve.

Typer evaluates the annotations of *all* registered commands when the module is imported,
so one bad annotation anywhere breaks every command. That failure mode is unusually
expensive here: `rollout` runs inside Slurm shards whose stdout goes to a per-shard log
nobody reads, so the batch job reports "evaluation complete" with zero episodes and the
report silently falls back to whatever stale files were on disk.

A `--help` on each command costs milliseconds and closes that hole.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agentgym.cli import app

COMMANDS = sorted(
    c.name or c.callback.__name__.replace("_", "-")
    for c in app.registered_commands
)


def test_cli_module_imports():
    assert COMMANDS, "no commands registered"


@pytest.mark.parametrize("command", COMMANDS)
def test_command_annotations_resolve(command):
    """`--help` forces typer to resolve the command's annotations."""
    proc = subprocess.run(
        [sys.executable, "-m", "agentgym.cli", command, "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"`agentgym {command} --help` failed -- a broken annotation here disables every "
        f"command, including rollout:\n{proc.stderr[-800:]}"
    )


def test_build_datasets_refuses_to_overwrite(tmp_path):
    """The quickstart runs this command; its default target holds the published corpora.

    Overwriting them is unrecoverable and silent -- every later training run would use
    different data while every reported number still claimed to describe the old data.
    """
    import json
    import subprocess
    import sys

    out = tmp_path / "sft"
    out.mkdir()
    (out / "sft_train.jsonl").write_text(json.dumps({"messages": []}) + "\n")
    before = (out / "sft_train.jsonl").read_text()

    r = subprocess.run(
        [sys.executable, "-m", "agentgym.cli", "build-datasets", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=300)

    assert r.returncode != 0, "a populated output directory must not be silently replaced"
    assert "refusing to overwrite" in (r.stderr + r.stdout).lower()
    assert (out / "sft_train.jsonl").read_text() == before, "the file was modified anyway"
