"""The prompt contract.

These exist because a prompt-construction bug fails *open*: nothing raises, nothing logs,
and the resulting numbers look plausible. Tool schemas were once passed as an OpenAI API
field that the project's own chat template ignored, so the model was asked to call tools
it had never been shown -- which silently voided both the base-model comparison and the
entire schema-shift suite.

The invariant asserted here is simple and worth stating plainly: **whatever tool surface
the gateway presents must be literally present in the text the model receives.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentgym.harness.prompts import render_tool_catalog, system_prompt
from agentgym.mcp import perturb as perturb_mod
from agentgym.mcp.perturb import SUITES
from agentgym.mcp.tools import REGISTRY

# The tokenizer is the whole point of this suite: it renders the chat template the tool
# schema has to survive, which is the invariant the OpenAI-field bug violated. Without
# transformers there is nothing to assert, so skip rather than fail at collection -- the
# same treatment torch gets in test_episode_grpo.py. A `pip install -e .[dev]` checkout
# (which is what CI builds) has no transformers; `.[train]` does.
AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

TEMPLATE = Path("src/agentgym/train/agentgym_chat_template.jinja")


def _clean_schemas() -> list[dict]:
    return [t.openai_schema() for t in REGISTRY.values()]


def test_every_tool_appears_in_the_system_prompt():
    prompt = system_prompt(_clean_schemas())
    missing = [t.name for t in REGISTRY.values() if t.name not in prompt]
    assert not missing, f"tools absent from the prompt the model receives: {missing}"


def test_required_arguments_appear_in_the_catalog():
    catalog = render_tool_catalog(_clean_schemas())
    # A model cannot supply an argument it was never told about.
    assert "namespace" in catalog
    assert "memory_limit_mi" in catalog
    assert "root_cause" in catalog


@pytest.mark.parametrize("suite", ["rename", "arg_rename", "extra_required_arg", "combined"])
def test_perturbed_surface_is_visible_in_the_prompt(suite):
    """A renamed tool must be renamed *in the prompt*, or the suite measures nothing."""
    spec = SUITES[suite]
    toolset = perturb_mod.build(
        perturb_mod.PerturbationSpec(spec.kinds, spec.severity, seed=7)
    )
    prompt = system_prompt(toolset.openai_schemas())

    presented = {t.name for t in toolset.tools}
    missing = [name for name in presented if name not in prompt]
    assert not missing, f"{suite}: presented tools absent from prompt: {missing[:5]}"

    if suite in ("rename", "combined"):
        renamed = [p for p, canon in toolset._tool_alias.items() if p != canon]
        assert renamed, "rename suite produced no renamed tools"
        # The new name is shown; the retired name must not linger anywhere in the prompt.
        assert any(name in prompt for name in renamed)
        retired = [
            canon for presented_name, canon in toolset._tool_alias.items()
            if presented_name != canon and canon not in presented
        ]
        leaked = [c for c in retired if f"- {c}(" in prompt]
        assert not leaked, f"retired tool names still advertised: {leaked[:5]}"

    if suite == "extra_required_arg":
        added = {a for args in toolset._required_extra.values() for a in args}
        assert added, "extra_required_arg suite added no arguments"
        assert all(a in prompt for a in added)


def test_chat_template_preserves_the_system_prompt():
    """The template must not drop or truncate the catalog on its way to the model."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    tok.chat_template = TEMPLATE.read_text()
    prompt = system_prompt(_clean_schemas())
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "incident"}],
        tokenize=False, add_generation_prompt=True,
    )
    missing = [t.name for t in REGISTRY.values() if t.name not in rendered]
    assert not missing, f"template dropped tools: {missing[:5]}"
    assert rendered.rstrip().endswith("<|im_start|>assistant")


def test_inference_history_matches_the_training_target_format():
    """The conversation the model is *given* must look like the one it was *trained* on.

    Training targets are `reasoning + fenced JSON call` inside `content`. If the harness
    instead records the call in a separate `tool_calls` field, the chat template (which
    renders `content` only) silently drops it -- and from turn two the model sees a
    history of prose-without-calls and continues that pattern. Turn one still looks
    perfect, which is what makes this failure so easy to ship.
    """
    from agentgym.data.curate import _assistant_text
    from agentgym.harness.harness import _assistant_turn
    from agentgym.harness.state import ToolCall

    action = ToolCall(name="list_pods", args={"namespace": "inc-1"},
                      thought="Check pod state.")
    inference = _assistant_turn(action, result=None)

    class _Step:
        tool = presented_tool = "list_pods"
        args = {"namespace": "inc-1"}
        thought = "Check pod state."

    training = _assistant_text(_Step())

    assert inference == training, (
        "training target and inference history diverged:\n"
        f"  training : {training!r}\n  inference: {inference!r}"
    )
    assert "```json" in inference


def test_harness_history_carries_the_tool_call(monkeypatch):
    """End-to-end: after a step, the recorded assistant message contains the call."""
    from agentgym.harness.harness import _assistant_turn
    from agentgym.harness.state import ToolCall

    msg = _assistant_turn(
        ToolCall(name="set_image", args={"namespace": "n", "image": "x:1"},
                 thought="Roll back."),
        result=None,
    )
    assert "set_image" in msg and "```json" in msg, msg


def test_stop_token_is_inside_the_loss_mask():
    """The model must be trained to *end its turn*, not just to produce content.

    With `<|im_end|>` outside the {% generation %} block the loss mask covers the
    content but not the token that terminates the turn, so the model never learns to
    stop. Nothing in training reports this -- teacher forcing always supplies the next
    token, so loss and token accuracy look excellent. It surfaces only at generation
    time, as turns that ramble until the sampling cap and emit no tool call.
    """
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    tok.chat_template = TEMPLATE.read_text()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "incident"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "TOOL RESULT: ok"},
        {"role": "assistant", "content": "second"},
    ]
    out = tok.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    masked = [i for i, m in zip(out["input_ids"], out["assistant_masks"], strict=True) if m]
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    assert eos in masked, "the stop token is not trained; the model will never end a turn"
    assert masked.count(eos) == 2, "every assistant turn must train its own stop token"
    assert tok.decode(masked) == "first<|im_end|>second<|im_end|>"


def test_loss_mask_excludes_tool_output():
    """The counterpart guarantee: observations must NOT be trained on, or the model
    learns to invent cluster state instead of querying for it."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    tok.chat_template = TEMPLATE.read_text()
    messages = [
        {"role": "user", "content": "TOOL RESULT: SENTINEL_OBSERVATION_TEXT"},
        {"role": "assistant", "content": "acting"},
    ]
    out = tok.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    masked = [i for i, m in zip(out["input_ids"], out["assistant_masks"], strict=True) if m]
    assert "SENTINEL" not in tok.decode(masked)


def test_training_examples_carry_the_catalog():
    """SFT data must be trained with the same catalog it is served with."""
    from agentgym import config

    path = config.DATA_DIR / "sft" / "sft_train.jsonl"
    if not path.exists():
        pytest.skip("no SFT dataset built yet")
    first = json.loads(path.read_text().splitlines()[0])
    system = first["messages"][0]["content"]
    assert "# Available tools" in system
    missing = [t.name for t in REGISTRY.values() if t.name not in system]
    assert not missing, f"training prompt is missing tools: {missing[:5]}"
