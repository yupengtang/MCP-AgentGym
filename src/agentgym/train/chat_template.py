"""Which chat template a base model gets, resolved in one place.

The project ships its own chat templates rather than using a model's stock one, for two
reasons recorded in the templates themselves: TRL needs `{% generation %}` markers to mask
the loss to assistant tokens, and the same file is handed to vLLM at serving time so the
trained prompt and the served prompt are byte-identical.

Adding a second base model turns that into a routing problem. Llama 3 does not speak
ChatML, so feeding it the Qwen template would tokenise `<|im_start|>` as ordinary text and
train the model to stop on a token its tokenizer does not treat as EOS. Nothing about that
raises: the loss curve looks fine, because teacher forcing always supplies the next token,
and it shows up only as turns that never terminate at generation time. That is the same
failure mode as the OpenAI-tool-field bug in reports/EXPERIMENTS.md, so the mapping lives
here and is asserted by tests rather than being duplicated across trainers and launchers.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent

CHATML = _DIR / "agentgym_chat_template.jinja"
LLAMA3 = _DIR / "agentgym_chat_template_llama3.jinja"

# Matched against a lowercased model id. First hit wins, so order matters only where one
# family's name is a substring of another's.
_FAMILIES: tuple[tuple[tuple[str, ...], Path], ...] = (
    (("llama-3", "llama3", "llama_3"), LLAMA3),
    (("qwen", "mistral", "gemma"), CHATML),
)


def template_path_for(model: str) -> Path:
    """The template file a base model must be trained and served with.

    Raises rather than falling back to a default. A silent default is how a model ends up
    trained on delimiters its tokenizer does not know, and the resulting run looks healthy
    in every metric that is cheap to check.
    """
    needle = model.lower()
    for keys, path in _FAMILIES:
        if any(k in needle for k in keys):
            return path
    raise ValueError(
        f"no chat template registered for base model {model!r}. Add it to _FAMILIES in "
        f"{__file__}, and add the pairing to tests/test_chat_template.py -- guessing a "
        f"template produces a run that trains cleanly and cannot stop generating."
    )


def template_for(model: str) -> str:
    """The template's text, for assigning to `tokenizer.chat_template`."""
    return template_path_for(model).read_text()


__all__ = ["CHATML", "LLAMA3", "template_for", "template_path_for"]


if __name__ == "__main__":  # `python -m agentgym.train.chat_template --model X`
    # The Slurm launchers pass a --chat-template path to vLLM. They resolve it through
    # this entry point rather than hardcoding one, so serving cannot disagree with
    # training about which template a model gets -- the disagreement would be silent.
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Print the chat template path for a model.")
    ap.add_argument("--model", required=True)
    try:
        print(template_path_for(ap.parse_args().model))
    except ValueError as exc:
        sys.exit(str(exc))
