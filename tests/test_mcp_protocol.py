"""Protocol compliance: a real MCP client, over a real transport, against real tools.

This is what backs the claim that the tools are MCP tools rather than functions with an
MCP-shaped docstring. The client here is the reference `mcp` client library speaking the
actual JSON-RPC protocol over an in-memory stream pair.
"""

from __future__ import annotations

import json

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from agentgym.incidents.gym import AgentGym
from agentgym.mcp.servers import build_server
from agentgym.mcp.tools import REGISTRY, SERVERS


@pytest.fixture(scope="module")
def gym():
    with AgentGym(offset=70) as g:
        yield g


@pytest.fixture(scope="module")
def episode(gym):
    from tests.test_environment import _incident_for

    ep = gym.episode(_incident_for("OOMKilled"))
    yield ep
    ep.teardown()


async def _session(server_name: str, episode):
    """Connect a real MCP client to a real MCP server over paired streams."""
    server = build_server(server_name, lambda: episode)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        import anyio

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    server_read, server_write, server.create_initialization_options()
                )
            )
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


@pytest.mark.parametrize("server_name", list(SERVERS))
async def test_server_advertises_registry_schemas(server_name, episode):
    """Every tool the server advertises matches the registry byte for byte."""
    expected = {t.name: t for t in REGISTRY.values() if t.server == server_name}
    assert expected, f"{server_name} has no tools"

    async for session in _session(server_name, episode):
        listed = await session.list_tools()
        advertised = {t.name: t for t in listed.tools}

        assert set(advertised) == set(expected), (
            f"{server_name} advertises {sorted(advertised)} "
            f"but registry has {sorted(expected)}"
        )
        for name, tool in expected.items():
            assert advertised[name].description == tool.description
            assert advertised[name].inputSchema == tool.parameters
        break


async def test_call_tool_over_protocol(episode):
    """A read tool invoked through the protocol returns the same data as in-process."""
    async for session in _session("mcp-kubernetes", episode):
        result = await session.call_tool(
            "list_pods", {"namespace": episode.namespace}
        )
        assert not result.isError
        text = result.content[0].text
        pods = json.loads(text)
        assert isinstance(pods, list) and pods, "expected pods in the incident namespace"
        assert {"name", "phase", "ready", "restarts"} <= set(pods[0])
        break


async def test_tool_error_is_reported_not_raised(episode):
    """A bad argument surfaces as an error result rather than killing the session."""
    async for session in _session("mcp-kubernetes", episode):
        result = await session.call_tool(
            "describe_pod", {"namespace": episode.namespace, "name": "does-not-exist"}
        )
        text = result.content[0].text
        assert "ERROR" in text or result.isError, text
        # The session must still be usable afterwards.
        ok = await session.call_tool("list_pods", {"namespace": episode.namespace})
        assert not ok.isError
        break


async def test_observability_tools_over_protocol(episode):
    async for session in _session("mcp-observability", episode):
        health = await session.call_tool(
            "get_service_health", {"namespace": episode.namespace}
        )
        services = json.loads(health.content[0].text)
        assert isinstance(services, list) and services
        assert {"service", "available", "ready_endpoints"} <= set(services[0])
        break


def test_documented_tool_count_matches_the_registry():
    """The docs quote a tool count in five places; the registry is the only source of truth.

    PROJECT.md carried 26 for weeks after the registry reached 36 -- nothing failed, the
    diagram simply described a smaller system than the one that shipped. A number repeated
    across documents drifts silently unless something checks it.
    """
    import re
    from pathlib import Path

    from agentgym.mcp.tools import REGISTRY

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"(\d+) typed tools")
    found: dict[str, set[str]] = {}
    for name in ("README.md", "PROJECT.md", "DESIGN.md"):
        f = root / name
        if f.exists():
            hits = set(pattern.findall(f.read_text()))
            if hits:
                found[name] = hits

    assert found, "no document quotes a tool count; this guard has nothing to protect"
    for name, hits in found.items():
        assert hits == {str(len(REGISTRY))}, (
            f"{name} claims {sorted(hits)} typed tools but the registry has {len(REGISTRY)}")
