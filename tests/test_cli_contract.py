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
