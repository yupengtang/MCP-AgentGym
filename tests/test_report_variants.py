"""Variant normalisation in the evaluation report.

This function decides which row of every results table an episode lands in. When it is
wrong the failure is silent and flattering: two policies average into one row that
describes neither, and the number still looks plausible. That has happened once already in
this project -- a `dpo5` ablation arm was merged into `dpo` by substring matching -- so the
distinctions the tables depend on are asserted here rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgym.eval.report import (
    VARIANT_ORDER,
    _variant_of,
    capability_matrix,
)


@dataclass
class FakeTraj:
    model: str
    perturbation: str = "clean"
    split: str = "in_distribution"
    reward: float = 1.0
    wall_seconds: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated_turns: int = 0
    settle_timeouts: int = 0
    clean: bool = True
    fault: str = "OOMKilled"

    @property
    def verification(self):
        @dataclass
        class V:
            resolved: bool = True
            root_cause_correct: bool = True
            unsafe_actions: int = 0
            invalid_calls: int = 0
            tool_calls: int = 5
        return V()


def test_families_are_recognised():
    assert _variant_of(FakeTraj("Qwen/Qwen3-8B")) == "base"
    assert _variant_of(FakeTraj("expert-scripted")) == "expert-scripted"
    assert _variant_of(FakeTraj("sft")) == "sft"
    assert _variant_of(FakeTraj("dpo")) == "dpo"
    assert _variant_of(FakeTraj("grpo")) == "grpo"


def test_ablation_suffixes_are_preserved():
    """Regression: `dpo5` must not be absorbed into `dpo`.

    Both arms exist to be compared; merging them averages two policies into one row and
    makes the ablation invisible while the table still renders.
    """
    assert _variant_of(FakeTraj("dpo5")) == "dpo5"
    # Separators are stripped from the suffix, so `dpo-v5` normalises to `dpov5`. The
    # property that matters is not the exact spelling but that it stays distinct.
    assert _variant_of(FakeTraj("dpo-v5")) == "dpov5"
    assert _variant_of(FakeTraj("dpo5")) != _variant_of(FakeTraj("dpo"))
    assert _variant_of(FakeTraj("dpo-v5")) != _variant_of(FakeTraj("dpo"))


def test_episode_grpo_does_not_collapse_into_action_grpo():
    """Episode-level and action-level GRPO are different policies optimising different
    objectives. `grpo-episode` keeps them apart; `egrpo` and `episode-grpo` do not, which
    is why the serving name is fixed by convention and asserted here."""
    assert _variant_of(FakeTraj("grpo-episode")) == "grpoepisode"
    assert _variant_of(FakeTraj("grpo-episode")) != _variant_of(FakeTraj("grpo"))
    # The unsafe spellings, documented so the trap is visible rather than rediscovered.
    assert _variant_of(FakeTraj("egrpo")) == "grpo"
    assert _variant_of(FakeTraj("episode-grpo")) == "grpo"


def test_every_ordered_variant_is_reachable():
    """Each name in VARIANT_ORDER must be producible, or the row can never be filled."""
    producible = {
        "expert-scripted": "expert-scripted",
        "base": "Qwen/Qwen3-8B",
        "prompted": "Qwen/Qwen3-8B",   # served as base, labelled by the run
        "sft": "sft",
        "dpo": "dpo",
        "dpo5": "dpo5",
        "grpo": "grpo",
        "grpoepisode": "grpo-episode",
    }
    for variant in VARIANT_ORDER:
        assert variant in producible, f"{variant} has no known served name"
        if variant != "prompted":
            assert _variant_of(FakeTraj(producible[variant])) == variant


def test_unlisted_variant_still_appears_in_the_table():
    """A variant missing from VARIANT_ORDER must not vanish from the report.

    Silently dropping a row is worse than an odd sort order: the run looks complete and
    the arm simply is not there.
    """
    rows = capability_matrix([
        FakeTraj("sft"),
        FakeTraj("dpo9-experimental"),
    ])
    assert "sft" in rows
    assert "dpo9-experimental" in rows


def test_distinct_variants_are_not_pooled():
    rows = capability_matrix([FakeTraj("dpo"), FakeTraj("dpo5"), FakeTraj("grpo-episode")])
    assert set(rows) == {"dpo", "dpo5", "grpoepisode"}
    assert all(m["n"] == 1 for m in rows.values()), "one episode each, not pooled"


def test_readme_serving_numbers_match_the_stored_benchmark():
    """Every serving figure in README must be recomputable from artifacts/serving/.

    The README carried 82,943 ms / 6,685 ms / 12.4x / 17x while the stored benchmark said
    84,418 / 6,576 / 12.8x / 18x -- a stale transcription from an earlier run. Close enough
    to survive any reading, wrong enough that the shop-window numbers could not be
    regenerated from what was on disk. Both directions matter, but note this one understated
    the result, which is the direction nobody re-checks.
    """
    import json
    import os
    import re
    from pathlib import Path

    art = Path(os.environ.get("AGENTGYM_ARTIFACTS", "")) / "serving"
    if not (art / "vllm_prefix_cache_on.json").exists():
        import pytest
        pytest.skip("serving benchmark artifacts not present")

    def rows(name):
        data = json.loads((art / name).read_text())["prefix_scaling"]
        return {(r["turns"], r["concurrency"]): r for r in data}

    off, on = rows("vllm_prefix_cache_off.json"), rows("vllm_prefix_cache_on.json")
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    a, b = off[(20, 32)], on[(20, 32)]
    speedup = a["latency_p50_ms"] / b["latency_p50_ms"]
    tput = b["requests_per_second"] / a["requests_per_second"]

    assert f"{a['latency_p50_ms']:,.0f} ms" in readme, "p50-without does not match the artifact"
    assert f"{b['latency_p50_ms']:,.0f} ms" in readme, "p50-with does not match the artifact"
    assert f"{speedup:.1f}×" in readme, f"speedup {speedup:.1f}x is not the one quoted"
    assert re.search(rf"{tput:.0f}×\)", readme), f"throughput ratio {tput:.0f}x is not quoted"
