"""The language-model policy.

Turns a model turn into a harness action. The parsing is deliberately forgiving about
*form* and strict about *content*: models emit tool calls through the native tool-call
field, as a fenced JSON block, or as bare JSON, and all three are accepted -- but a call
naming a tool that does not exist is passed straight through to the gateway so the
mistake is recorded and penalised rather than silently repaired.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agentgym.backends.base import InferenceBackend
from agentgym.harness.state import Action, AgentState, Finish, ToolCall

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# Qwen and several other families wrap calls in explicit tags rather than a code fence.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _balanced_objects(text: str) -> list[str]:
    """Every top-level {...} in `text`, brace-balanced and string-aware.

    Regex cannot do this correctly: a lazy match stops at the first `}` and truncates
    nested objects, while a greedy one swallows everything between the first `{` and the
    last `}`. Both silently produce unparseable output, which reads as "the model failed"
    when the model was fine.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


@dataclass
class LLMPolicy:
    """Drives an InferenceBackend as the agent's brain."""

    backend: InferenceBackend
    name: str = "llm"
    temperature: float = 0.0
    # Reasoning models emit a <think> block before answering. A budget that fits the
    # answer but not the thinking produces a truncated turn with no JSON in it, which
    # would be scored as "the model cannot call tools" when the real cause is the cap.
    max_tokens: int = 1536
    adapter: str | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_failures: int = 0
    # Turns that hit the token cap. Tracked separately from parse failures so a report
    # can distinguish "rambled past the budget" from "emitted malformed output".
    truncations: int = 0
    _last_raw: str = field(default="", init=False)

    def reset(self, state: AgentState) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.parse_failures = 0
        self.truncations = 0

    def act(self, state: AgentState, tool_schemas: list[dict[str, Any]]) -> Action:
        # Tools are not passed as an API field: the catalog is already in the system
        # prompt (see prompts.system_prompt), and the chat template used at both training
        # and serving time renders messages only. Sending them twice would either be
        # dropped silently or duplicated, depending on the template.
        result = self.backend.generate(
            state.messages, None,
            temperature=self.temperature, max_tokens=self.max_tokens,
            adapter=self.adapter,
        )
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self._last_raw = result.text
        if result.finish_reason == "length":
            self.truncations += 1

        action = self._to_action(result.text, result.tool_calls)
        if (isinstance(action, ToolCall) and action.name == "__unparseable__"
                and result.finish_reason == "length"):
            # Say why it failed, so the model can shorten its reasoning rather than
            # repeating the same over-long turn until the budget runs out.
            action.thought = (
                "[turn truncated at the token limit before a tool call was emitted]\n"
                + action.thought
            )
        # Keep the exact generated text so SFT trains on what the model really produced,
        # not on a reconstruction of it.
        action.raw = result.text
        return action

    # --- parsing ------------------------------------------------------------

    def _to_action(self, text: str, native_calls: list[dict[str, Any]]) -> Action:
        if native_calls:
            call = native_calls[0]
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            return self._as_action(call.get("name", ""), args, text)

        for candidate in self._json_candidates(text):
            name = candidate.get("name") or candidate.get("tool") or candidate.get("tool_name")
            if not name:
                continue
            args = (candidate.get("arguments") or candidate.get("args")
                    or candidate.get("parameters") or {})
            if not isinstance(args, dict):
                args = {}
            return self._as_action(str(name), args, text)

        # Nothing parseable. Return a call the gateway will reject, so the failure lands
        # in the trajectory as an invalid call and the recovery path gets exercised.
        self.parse_failures += 1
        return ToolCall(
            name="__unparseable__",
            args={},
            thought=text[:600],
        )

    def _as_action(self, name: str, args: dict[str, Any], text: str) -> Action:
        thought = self._thought(text)
        if name in ("submit_diagnosis", "report_root_cause"):
            return Finish(
                root_cause=str(args.get("root_cause", "")),
                summary=str(args.get("summary", "")),
                thought=thought,
            )
        return ToolCall(name=name, args=args, thought=thought)

    @staticmethod
    def _json_candidates(text: str) -> list[dict[str, Any]]:
        """Extract every JSON object a model might have meant as a tool call.

        Models are inconsistent about framing: a fenced ```json block, Qwen-style
        <tool_call> tags, or a bare object after a </think> block are all common, and a
        reasoning model often emits the last of those. Extraction is done by scanning for
        balanced braces rather than by regex -- a lazy `.*?}` stops at the first closing
        brace and silently truncates any call whose `arguments` is itself an object,
        which is nearly all of them.
        """
        out: list[dict[str, Any]] = []

        def try_load(blob: str) -> None:
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                return
            if isinstance(obj, dict):
                out.append(obj)

        for match in _JSON_BLOCK.finditer(text):
            try_load(match.group(1))
        for match in _TOOL_CALL_TAG.finditer(text):
            try_load(match.group(1).strip())
        if out:
            return out

        # Reasoning output lives before the answer; searching after </think> avoids
        # picking up an object the model was only musing about.
        body = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
        for blob in _balanced_objects(body):
            try_load(blob)
        if not out and body is not text:
            for blob in _balanced_objects(text):
                try_load(blob)
        return out

    @staticmethod
    def _thought(text: str) -> str:
        """Prose before the first JSON block is the model's reasoning."""
        cut = text.split("```")[0].strip()
        # Strip any <think> block: it is useful for training but noisy in a transcript.
        cut = re.sub(r"<think>.*?</think>", "", cut, flags=re.DOTALL).strip()
        return cut[:800]


__all__ = ["LLMPolicy"]
