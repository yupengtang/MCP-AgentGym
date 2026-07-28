"""LLM policy parsing and the harness's error-recovery path.

Models emit tool calls in several shapes, and a real one will eventually emit something
malformed. What matters is that the harness turns each case into the *right* trajectory
record: a valid call dispatches, an invalid one is penalised rather than repaired, and a
run of failures triggers the recovery hint instead of burning the whole budget.

A scripted backend makes all of that testable without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from agentgym.backends.base import GenerationResult, InferenceBackend
from agentgym.harness.harness import RECOVERY_AFTER_ERRORS, AgentHarness, HarnessConfig
from agentgym.harness.llm_policy import LLMPolicy
from agentgym.harness.state import AgentState, Finish, ToolCall
from agentgym.incidents.gym import AgentGym
from agentgym.mcp.gateway import Budget, ToolGateway


@dataclass
class ScriptedBackend(InferenceBackend):
    """Replays a fixed list of completions, one per turn."""

    completions: list[str] = field(default_factory=list)
    native_calls: list[list[dict]] = field(default_factory=list)
    finish_reason: str = "stop"
    name: str = "scripted"
    _i: int = 0

    def generate(self, messages, tools=None, *, temperature=0.0, max_tokens=1024,
                 adapter=None) -> GenerationResult:
        idx = min(self._i, len(self.completions) - 1)
        text = self.completions[idx]
        calls = self.native_calls[idx] if idx < len(self.native_calls) else []
        self._i += 1
        return GenerationResult(text=text, tool_calls=calls, prompt_tokens=len(str(messages)) // 4,
                                completion_tokens=len(text) // 4, backend="scripted",
                                finish_reason=self.finish_reason)


def _policy(completions, native=None, finish_reason="stop") -> LLMPolicy:
    return LLMPolicy(backend=ScriptedBackend(completions=completions,
                                             native_calls=native or [],
                                             finish_reason=finish_reason))


def _state() -> AgentState:
    """A minimal state for the parsing tests, which do not need a cluster."""
    state = AgentState(incident_id="test", namespace="test")
    state.messages = [{"role": "system", "content": "sys"},
                      {"role": "user", "content": "incident"}]
    return state


# --- parsing -----------------------------------------------------------------

def test_parses_fenced_json_block():
    p = _policy(['I will look at the pods.\n\n```json\n'
                 '{"name": "list_pods", "arguments": {"namespace": "inc-1"}}\n```'])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "list_pods"
    assert action.args == {"namespace": "inc-1"}
    assert "look at the pods" in action.thought


def test_parses_native_tool_call_field():
    p = _policy(["", ], native=[[{"name": "get_deployment",
                                  "arguments": {"namespace": "n", "name": "api"}}]])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "get_deployment"


def test_parses_bare_json_object():
    p = _policy(['{"name": "list_events", "arguments": {"namespace": "inc-1"}}'])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "list_events"


def test_strips_think_blocks_from_thought():
    p = _policy(['<think>internal reasoning</think>\nChecking health.\n\n'
                 '```json\n{"name": "list_pods", "arguments": {"namespace": "x"}}\n```'])
    action = p.act(_state(), [])
    assert "internal reasoning" not in action.thought
    assert "Checking health" in action.thought


def test_submit_diagnosis_becomes_finish():
    p = _policy(['```json\n{"name": "submit_diagnosis", "arguments": '
                 '{"root_cause": "invalid_container_image_tag", "summary": "bad tag"}}\n```'])
    action = p.act(_state(), [])
    assert isinstance(action, Finish)
    assert action.root_cause == "invalid_container_image_tag"


def test_unparseable_output_is_surfaced_not_repaired():
    """A model that emits prose must produce a rejected call, not a silent no-op --
    otherwise the trajectory understates how often the model failed."""
    p = _policy(["I think the problem is probably memory related, let me consider."])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "__unparseable__"
    assert p.parse_failures == 1


def test_malformed_arguments_are_passed_through():
    """Broken JSON in the arguments field must reach the gateway so it is recorded as an
    invalid call, rather than being quietly fixed up."""
    p = _policy([""], native=[[{"name": "list_pods", "arguments": {"__malformed__": "{oops"}}]])
    action = p.act(_state(), [])
    assert action.args == {"__malformed__": "{oops"}


def test_parses_bare_object_with_nested_arguments_after_think():
    """The exact shape a reasoning model emits, and the one a lazy regex truncates.

    A `.*?}` stops at the brace closing `arguments`, yielding invalid JSON -- so every
    well-formed call with a nested argument object was being discarded and scored as a
    model failure. Brace-balanced extraction is the fix.
    """
    p = _policy(['<think>\nThe image tag looks wrong, I should check deploy history.\n</think>\n\n'
                 'tool\n{\n  "name": "get_recent_changes",\n  "arguments": {\n'
                 '    "namespace": "inc-00002"\n  }\n}'])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "get_recent_changes"
    assert action.args == {"namespace": "inc-00002"}


def test_parses_qwen_style_tool_call_tags():
    p = _policy(['<tool_call>\n{"name": "list_pods", "arguments": {"namespace": "n"}}\n'
                 '</tool_call>'])
    action = p.act(_state(), [])
    assert isinstance(action, ToolCall)
    assert action.name == "list_pods"


def test_object_inside_think_is_not_mistaken_for_the_call():
    """A model musing about a call must not be treated as making it."""
    p = _policy(['<think>\nMaybe I should call {"name": "delete_pod", "arguments": {"name": "x"}}?\n'
                 'No, that is unsafe.\n</think>\n\n'
                 '{"name": "list_pods", "arguments": {"namespace": "n"}}'])
    action = p.act(_state(), [])
    assert action.name == "list_pods", "picked up an object from inside the think block"


def test_narration_without_a_call_is_still_a_failure():
    """Recovering genuine calls must not make the parser credulous: prose describing an
    intention is not an action."""
    p = _policy(["I'll wait for the rollout to finish and then check service health."])
    action = p.act(_state(), [])
    assert action.name == "__unparseable__"


def test_hallucinated_tool_result_is_not_treated_as_a_call():
    """Models sometimes emit what a tool *would have returned*. That object has no
    `name` key and must be rejected, not dispatched."""
    p = _policy(['<think>done</think>\n\n{"service": "gateway", "availability": "100%", '
                 '"endpoint_count": 2}'])
    action = p.act(_state(), [])
    assert action.name == "__unparseable__"


def test_truncated_turns_are_counted_separately_from_parse_failures():
    """A reasoning model that spends its whole budget inside <think> emits no JSON.

    That must be distinguishable from emitting malformed output: one is a budget
    problem, the other a capability problem, and conflating them turns a
    misconfigured cap into a false conclusion about the model.
    """
    p = _policy(["<think>Let me consider the memory limits and the probe config and",
                 ], finish_reason="length")
    action = p.act(_state(), [])
    assert action.name == "__unparseable__"
    assert p.truncations == 1
    assert "truncated at the token limit" in action.thought, (
        "the model must be told why the turn failed, or it repeats the same over-long turn"
    )


def test_untruncated_parse_failure_is_not_counted_as_truncation():
    p = _policy(["just some prose"], finish_reason="stop")
    p.act(_state(), [])
    assert p.parse_failures == 1
    assert p.truncations == 0


def test_default_token_budget_leaves_room_for_reasoning():
    """Guards the regression directly: the cap must exceed a plausible think block."""
    assert LLMPolicy(backend=None).max_tokens >= 1024


# --- harness integration -----------------------------------------------------

@pytest.fixture(scope="module")
def gym():
    with AgentGym(offset=78) as g:
        yield g


def _episode(gym):
    from tests.test_environment import _incident_for
    return gym.episode(_incident_for("ImagePullBackOff"))


def test_invalid_calls_are_penalised_and_recorded(gym):
    ep = _episode(gym)
    try:
        gw = ToolGateway.create(ep, budget=Budget(max_tool_calls=8))
        policy = _policy(["not a tool call at all"] * 8)
        h = AgentHarness(episode=ep, gateway=gw, policy=policy,
                         config=HarnessConfig(max_steps=6))
        t = h.run(incident_id="test-invalid", model="scripted")

        assert not t.success
        assert t.verification.invalid_calls > 0
        assert t.reward < 0, "invalid calls must cost reward"
        assert all(s.error_kind for s in t.steps if s.tool is None)
    finally:
        ep.teardown()


def test_recovery_hint_is_injected_after_repeated_failures(gym):
    ep = _episode(gym)
    try:
        gw = ToolGateway.create(ep, budget=Budget(max_tool_calls=12))
        policy = _policy(["garbage"] * 12)
        h = AgentHarness(episode=ep, gateway=gw, policy=policy,
                         config=HarnessConfig(max_steps=RECOVERY_AFTER_ERRORS + 1))
        h.run(incident_id="test-recovery", model="scripted")

        hints = [m for m in h.state.messages
                 if m["role"] == "user" and "failed tool calls in a row" in m["content"]]
        assert hints, "harness never offered a recovery hint"
    finally:
        ep.teardown()


def test_a_scripted_correct_fix_is_verified(gym):
    """End to end through the LLM path: parse -> dispatch -> verify -> reward."""
    ep = _episode(gym)
    try:
        good_image = ep.params["good_image"]
        service = ep.params["service"]
        ns = ep.namespace
        script = [
            f'Checking pods.\n\n```json\n{json.dumps({"name": "list_pods", "arguments": {"namespace": ns}})}\n```',
            f'Rolling back the image.\n\n```json\n{json.dumps({"name": "set_image", "arguments": {"namespace": ns, "name": service, "image": good_image}})}\n```',
            f'Waiting for rollout.\n\n```json\n{json.dumps({"name": "wait_for_rollout", "arguments": {"namespace": ns, "name": service}})}\n```',
            '```json\n{"name": "submit_diagnosis", "arguments": '
            '{"root_cause": "invalid_container_image_tag", "summary": "rolled back"}}\n```',
        ]
        gw = ToolGateway.create(ep, budget=Budget(max_tool_calls=10))
        h = AgentHarness(episode=ep, gateway=gw, policy=_policy(script),
                         config=HarnessConfig(max_steps=8))
        t = h.run(incident_id="test-fix", model="scripted")

        assert t.success, f"verifier did not accept the fix: {t.verification}"
        assert t.verification.root_cause_correct
        assert t.verification.invalid_calls == 0
        assert t.reward > 2.0
    finally:
        ep.teardown()
