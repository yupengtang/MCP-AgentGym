"""System prompt and incident briefing.

The prompt is versioned by hash and recorded on every trajectory, so a change to the
wording never silently invalidates a comparison between model variants.
"""

from __future__ import annotations

import hashlib

from agentgym.incidents.faults import ROOT_CAUSES

SYSTEM_PROMPT = """\
You are an on-call site reliability engineer for a Kubernetes platform. An incident has \
been assigned to you. Diagnose it and fix it using the tools provided.

How to work:
1. Orient. Read the intended architecture and the current health of every service before \
touching anything.
2. Investigate. Follow the evidence: alerts, then pod status, then container logs and \
events for the failing pods, then the specific resource that looks wrong. Recent \
deployment history is often decisive.
3. Remediate. Change the one thing that is actually wrong. Prefer the narrowest fix that \
addresses the root cause.
4. Validate. A change to a deployment does not take effect instantly. After changing \
one, call wait_for_rollout, then re-check service health. A pod that restarts is not the \
same as an incident that is resolved.
5. Finish by calling submit_diagnosis with the root cause and a one-line summary.

Rules that matter:
- Restarting or deleting a pod does not fix a configuration or resource problem. The \
replacement inherits the same spec. Fix the spec.
- Never scale a deployment to zero to make an alert stop.
- Only call tools that appear in the tool schema you were given. Tool names and argument \
names may differ from what you expect; read the schema rather than guessing.
- If a tool call fails, read the error and correct the call. Do not repeat it unchanged.

Valid root_cause values for submit_diagnosis:
""" + "\n".join(f"- {rc}" for rc in ROOT_CAUSES)


def incident_brief(episode) -> str:
    """The alert that pages the engineer. Deliberately thin: symptom, not cause."""
    return _brief(episode.namespace, episode.topology.name,
                  episode.topology.description, episode.topology.entrypoint)


def render_tool_catalog(tool_schemas: list[dict]) -> str:
    """Render the tool surface into text the model can actually read.

    The catalog lives in the system prompt rather than the OpenAI `tools` field because
    the training and serving chat template is a plain message renderer -- anything passed
    as a separate API field would be silently dropped before the model ever saw it. That
    also makes the schema-shift experiment meaningful: when a tool is renamed or gains an
    argument, the change appears in the prompt, so the model can read it and adapt.
    """
    lines: list[str] = []
    for schema in tool_schemas:
        fn = schema.get("function", schema)
        params = fn.get("parameters", {}) or {}
        props: dict = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        args = []
        for arg, spec in props.items():
            kind = spec.get("type", "any")
            if "enum" in spec:
                kind = "one of " + "|".join(str(v) for v in spec["enum"][:6])
            args.append(f"{arg}: {kind}{'' if arg in required else ' (optional)'}")
        signature = ", ".join(args) if args else ""
        lines.append(f"- {fn['name']}({signature})\n    {fn.get('description', '').strip()}")
    return "\n".join(lines)


def system_prompt(tool_schemas: list[dict] | None = None) -> str:
    """The full system prompt, including the tool catalog when one is supplied."""
    if not tool_schemas:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "# Available tools\n\n"
        "Call exactly one tool per turn by emitting a JSON object in a ```json fenced "
        "block, with keys \"name\" and \"arguments\". Keep any reasoning to a sentence "
        "or two before the block -- a turn that runs out of tokens before the JSON "
        "appears is a wasted turn. Only these tools exist:\n\n"
        f"{render_tool_catalog(tool_schemas)}"
    )


def _brief(namespace: str, application: str, description: str, entrypoint: str) -> str:
    return f"""\
INCIDENT: service degradation detected in namespace `{namespace}`.

Namespace: {namespace}
Application: {application} -- {description}
Entrypoint service: {entrypoint}

Monitoring is reporting failed health checks. Diagnose the root cause, fix it, and \
confirm the application is serving again. Use namespace `{namespace}` for every tool \
call that takes one."""


def incident_brief_from_trajectory(t) -> str:
    """Rebuild the exact user turn offline, so dataset construction needs no cluster."""
    return _brief(t.namespace, t.topology, t.topology_description, t.entrypoint)


def system_prompt_hash() -> str:
    return hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12]


__all__ = ["SYSTEM_PROMPT", "incident_brief", "system_prompt_hash"]
