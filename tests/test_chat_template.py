"""The chat templates, and the properties that fail silently when they are wrong.

Every assertion here corresponds to a failure that produces a *healthy-looking* training
run. The loss curve, the token accuracy and the eval loss are all computed under teacher
forcing, which supplies the next token no matter what the model would have done, so a
prompt-format defect is invisible until generation time. That is how the OpenAI-tool-field
bug survived long enough to void a base-model comparison (reports/EXPERIMENTS.md).

Adding a second base model multiplies the surface: the same trajectory corpus now renders
through two templates, and the arm only tests "is the trade-off a property of agent
post-training or of Qwen3-8B" if the two differ in delimiters and nothing else.
"""

from __future__ import annotations

import pytest

from agentgym.train.chat_template import CHATML, LLAMA3, template_for, template_path_for

MESSAGES = [
    {"role": "system", "content": "SYSTEM_MARKER"},
    {"role": "user", "content": "USER_MARKER"},
    {"role": "assistant", "content": "ASSISTANT_MARKER"},
]


@pytest.mark.parametrize(("model", "expected"), [
    ("Qwen/Qwen3-8B", CHATML),
    ("Qwen/Qwen3-32B", CHATML),
    ("meta-llama/Llama-3.1-8B-Instruct", LLAMA3),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", LLAMA3),
    ("mistralai/Mistral-7B-Instruct-v0.3", CHATML),
])
def test_model_maps_to_its_template(model, expected):
    assert template_path_for(model) == expected


def test_unregistered_model_raises_rather_than_defaulting():
    """A default here trains a model on delimiters its tokenizer does not know.

    The run completes, every metric looks normal, and the policy cannot stop generating.
    Refusing is the only outcome that surfaces the mistake while it is still cheap.
    """
    with pytest.raises(ValueError, match="no chat template registered"):
        template_path_for("some-org/BrandNewModel-9B")


@pytest.mark.parametrize("path", [CHATML, LLAMA3])
def test_stop_token_is_inside_the_generation_block(path):
    """The load-bearing placement, recorded in both templates.

    With the stop token outside {% generation %}, the loss mask covers the assistant's
    content but not the token that ends its turn, so the model is never trained to stop.
    Teacher forcing hides it completely; at generation time the policy rambles to the
    sampling cap and emits no tool call.
    """
    body = _emitting_body(path)
    start = body.index("{%- generation -%}")
    end = body.index("{%- endgeneration -%}")
    block = body[start:end]
    stop = "<|im_end|>" if path == CHATML else "<|eot_id|>"
    assert stop in block, f"{path.name}: {stop} must be inside the generation block"


def _emitting_body(path) -> str:
    """The template with its leading {# ... #} comment stripped.

    Checking the raw file conflates what a template *emits* with what its comment
    *mentions* -- the first version of the BOS check below failed on the comment that
    explains why the BOS is omitted. That is the same defect shape as the first
    test_launcher_plumbing, which searched raw text and passed because the explanatory
    comment contained the very flag it was asserting on.
    """
    body = path.read_text()
    if body.lstrip().startswith("{#"):
        body = body[body.index("-#}") + 3:]
    return body


def test_llama_template_does_not_emit_bos():
    """Llama 3 tokenizers add BOS themselves; emitting it here would double it.

    A doubled BOS shifts every position by one against what the model saw in
    pre-training, and like the rest of this file it costs nothing visible in training.
    """
    assert "<|begin_of_text|>" not in _emitting_body(LLAMA3)


@pytest.mark.parametrize(("path", "opener"), [
    (CHATML, "<|im_start|>"),
    (LLAMA3, "<|start_header_id|>"),
])
def test_templates_agree_on_structure_and_differ_only_in_delimiters(path, opener):
    """Both render the same three roles and a generation prompt, in the same order."""
    body = _emitting_body(path)
    for token in ("{%- generation -%}", "{%- endgeneration -%}", "add_generation_prompt",
                  "message['role'] == 'assistant'"):
        assert token in body, f"{path.name} is missing {token}"
    assert opener in body


def test_rendered_prompt_round_trips_through_a_real_tokenizer():
    """Render with the actual tokenizer, not a mock: the template is Jinja the tokenizer
    executes, so a syntax error or a bad variable only shows up here."""
    transformers = pytest.importorskip("transformers")
    import os
    model = os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model)
    except Exception as exc:                                  # no weights cached locally
        pytest.skip(f"tokenizer for {model} unavailable: {exc}")

    tok.chat_template = template_for(model)
    text = tok.apply_chat_template(MESSAGES, tokenize=False)
    for marker in ("SYSTEM_MARKER", "USER_MARKER", "ASSISTANT_MARKER"):
        assert marker in text

    # The generation prompt must open an assistant turn and leave it open.
    prompt = tok.apply_chat_template(MESSAGES[:2], tokenize=False, add_generation_prompt=True)
    assert prompt.rstrip().endswith("assistant") or prompt.endswith("\n"), prompt[-40:]
    assert "ASSISTANT_MARKER" not in prompt
