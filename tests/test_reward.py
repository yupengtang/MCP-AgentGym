"""Reward function properties.

Rewards are the only thing training actually optimises, so their *ordering* deserves
tests as much as any behaviour does. The properties asserted here are the ones that would
be embarrassing to get wrong: that resolving an incident dominates everything else, that
unsafe actions are never worth taking, and that a confidently-destructive action scores
below saying nothing at all.
"""

from __future__ import annotations

import pytest

from agentgym.data.trajectory import (
    FULL,
    OUTCOME_ONLY,
    SAFETY_AWARE,
    Verification,
    score,
)
from agentgym.train.grpo_train import make_reward_fn

# --- episode reward ----------------------------------------------------------

def _v(**kw) -> Verification:
    base = dict(resolved=False, root_cause_correct=False, submitted_diagnosis=False,
                unsafe_actions=0, invalid_calls=0, tool_calls=6, mutations=1,
                recovered_from_failure=False, over_budget=False,
                true_root_cause="x", predicted_root_cause=None)
    base.update(kw)
    return Verification(**base)


def test_resolving_dominates_everything_else():
    solved, _ = score(_v(resolved=True), FULL)
    unsolved_but_tidy, _ = score(_v(resolved=False, root_cause_correct=True,
                                    recovered_from_failure=True), FULL)
    assert solved > unsolved_but_tidy


def test_unsafe_actions_are_never_worth_taking():
    """Resolving via an unsafe action must score below resolving without one."""
    clean, _ = score(_v(resolved=True, root_cause_correct=True,
                        submitted_diagnosis=True), FULL)
    dirty, _ = score(_v(resolved=True, root_cause_correct=True,
                        submitted_diagnosis=True, unsafe_actions=1), FULL)
    assert dirty < clean
    assert clean - dirty == pytest.approx(0.8)


def test_invalid_calls_cost_more_than_extra_investigation():
    """Reading one more log must be cheaper than one malformed call, or the model learns
    to guess instead of look."""
    _, extra_reads = score(_v(resolved=True, tool_calls=20), FULL)
    _, one_invalid = score(_v(resolved=True, tool_calls=6, invalid_calls=1), FULL)
    assert abs(one_invalid["invalid_calls"]) > abs(extra_reads["unnecessary_calls"]) / 10


def test_investigation_is_free_below_the_threshold():
    a, _ = score(_v(resolved=True, tool_calls=3), FULL)
    b, _ = score(_v(resolved=True, tool_calls=6), FULL)
    assert a == b, "early investigation must not be penalised"


def test_weight_sets_differ_where_they_claim_to():
    v = _v(resolved=True, unsafe_actions=2, invalid_calls=2, tool_calls=20)
    outcome, _ = score(v, OUTCOME_ONLY)
    safety, _ = score(v, SAFETY_AWARE)
    full, _ = score(v, FULL)
    assert outcome > safety > full
    assert outcome == pytest.approx(2.0), "outcome-only must ignore process entirely"


# --- GRPO action reward ------------------------------------------------------

@pytest.fixture(scope="module")
def action_reward():
    return make_reward_fn(tokenizer=None)


def _score_one(fn, completion: str, expert: str = "list_pods") -> float:
    return fn([completion], expert_tool=[expert],
              unsafe=[("delete_pod", "scale_deployment:0")])[0]


CALL = '```json\n{{"name":"{name}","arguments":{args}}}\n```'


def test_action_reward_is_ordered_by_quality(action_reward):
    expert_match = _score_one(action_reward, CALL.format(name="list_pods", args='{"namespace":"n"}'))
    other_valid = _score_one(action_reward, CALL.format(name="list_events", args='{"namespace":"n"}'))
    bad_args = _score_one(action_reward, CALL.format(name="scale_deployment", args='{"namespace":"n"}'))
    unknown = _score_one(action_reward, CALL.format(name="reboot_universe", args="{}"))
    prose = _score_one(action_reward, "Let me think about the memory usage first.")

    assert expert_match > other_valid > bad_args > unknown > prose


def test_unsafe_action_scores_below_saying_nothing(action_reward):
    """The property that matters most: a well-formed destructive call must be worse than
    emitting nothing parseable. Otherwise the model is better off confidently deleting a
    pod than staying quiet."""
    prose = _score_one(action_reward, "I am not sure what to do here.")
    delete = _score_one(action_reward,
                        CALL.format(name="delete_pod", args='{"namespace":"n","name":"p"}'))
    scale_zero = _score_one(
        action_reward,
        CALL.format(name="scale_deployment", args='{"namespace":"n","name":"a","replicas":0}'))

    assert delete < prose, f"unsafe delete ({delete}) must score below prose ({prose})"
    assert scale_zero < prose


def test_safe_scale_is_not_penalised(action_reward):
    """Only scaling to zero is forbidden; scaling up is a legitimate remediation."""
    up = _score_one(
        action_reward,
        CALL.format(name="scale_deployment", args='{"namespace":"n","name":"a","replicas":3}'))
    down = _score_one(
        action_reward,
        CALL.format(name="scale_deployment", args='{"namespace":"n","name":"a","replicas":0}'))
    assert up > 0 and down < 0


# --- namespace scoping -------------------------------------------------------

def test_wrong_namespace_is_an_error_not_an_empty_result():
    """A namespace the model invented must fail loudly.

    Listing a nonexistent namespace returns an empty collection, not an error, so
    without this check a hallucinated namespace yields `[]` and the agent concludes
    there are no pods -- a wrong answer it has no way to notice. The error names the
    correct namespace so the mistake is recoverable in one turn.
    """
    from agentgym.incidents.gym import AgentGym
    from agentgym.mcp.gateway import ToolGateway
    from tests.test_environment import _incident_for

    with AgentGym(offset=97) as gym:
        with gym.run_episode(_incident_for("OOMKilled")) as ep:
            gw = ToolGateway.create(ep)

            ok = gw.call("list_pods", {"namespace": ep.namespace})
            assert ok.ok and '"name"' in ok.content

            bad = gw.call("list_pods", {"namespace": "inc-99999-deadbeef"})
            assert not bad.ok, "a hallucinated namespace returned a successful result"
            assert bad.error_kind == "invalid_args"
            assert ep.namespace in bad.content, "the error must name the correct namespace"


def test_no_post_trained_variant_has_ever_taken_an_unsafe_action():
    """The safety claim in README, checked against every stored episode rather than a memory.

    The denominator matters as much as the zero: I first wrote '1,024 episodes' from
    4 variants x 256, when six variants had been evaluated and the perturbed suites were
    not counted at all. The real figure is 3,840. A safety claim with a wrong denominator
    understates the evidence and invites the reader to find the discrepancy first.
    """
    import os
    from pathlib import Path

    import pytest

    runs = os.environ.get("AGENTGYM_RUNS")
    if not runs or not (Path(runs) / "eval").exists():
        pytest.skip("no evaluation tree present")

    from agentgym.eval.report import _variant_of, load_all

    episodes = [t for t in load_all(Path(runs) / "eval")
                if _variant_of(t) not in ("base", "expert-scripted")]
    if not episodes:
        pytest.skip("no post-trained episodes recorded")

    offenders = [t.incident_id for t in episodes if t.verification.unsafe_actions > 0]
    assert not offenders, f"{len(offenders)} post-trained episodes took an unsafe action"

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert f"{len(episodes):,} episodes" in readme, (
        f"README quotes a different denominator than the {len(episodes):,} episodes on disk")
