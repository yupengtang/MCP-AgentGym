"""Real MCP servers over the wire.

The rollout path calls tools in-process, because spawning four subprocesses per episode
would dominate the cost of running tens of thousands of episodes. But the tools are not
"MCP-shaped" by analogy -- they are served over the actual protocol here, from the same
registry, so a standard MCP client can drive this environment unchanged.

Run one:

    python -m agentgym.mcp.servers --server mcp-kubernetes --namespace inc-00001

`tests/test_mcp_protocol.py` connects a real MCP client to these and asserts the
advertised schemas match the registry exactly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from agentgym.mcp.tools import REGISTRY, SERVERS, ToolError


def build_server(server_name: str, episode_provider) -> Server:
    """An MCP server exposing exactly the tools registered to `server_name`."""
    server = Server(server_name)
    tools = [t for t in REGISTRY.values() if t.server == server_name]
    by_name = {t.name: t for t in tools}

    @server.list_tools()
    async def list_tools() -> list[MCPTool]:
        return [
            MCPTool(
                name=t.name,
                description=t.description,
                inputSchema=t.parameters,
            )
            for t in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        tool = by_name.get(name)
        if tool is None:
            raise ValueError(f"unknown tool {name!r} on {server_name}")
        episode = episode_provider()
        try:
            # Tool implementations are synchronous and talk to the API server over
            # HTTP; keep the event loop free while they do.
            result = await asyncio.to_thread(tool.fn, episode, arguments)
        except ToolError as exc:
            return [TextContent(type="text", text=f"ERROR: {exc}")]
        payload = result if isinstance(result, str) else json.dumps(result, indent=2, default=str)
        return [TextContent(type="text", text=payload)]

    return server


async def serve_stdio(server_name: str, episode_provider) -> None:
    server = build_server(server_name, episode_provider)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def _episode_from_env():
    """Attach to a live control plane described by the environment.

    Used when a server is run standalone against an episode another process created.
    """
    from pathlib import Path

    from agentgym.cluster.client import KubeClient
    from agentgym.cluster.kubelet_sim import KubeletSim
    from agentgym.cluster.mesh import Mesh
    from agentgym.cluster.telemetry import Telemetry
    from agentgym.incidents.episode import Episode
    from agentgym.incidents.faults import FAULTS_BY_NAME
    from agentgym.incidents.topology import TOPOLOGIES_BY_NAME

    kubeconfig = Path(os.environ["AGENTGYM_KUBECONFIG"])
    namespace = os.environ["AGENTGYM_NAMESPACE"]
    topology = TOPOLOGIES_BY_NAME[os.environ.get("AGENTGYM_TOPOLOGY", "web-api")]
    fault = FAULTS_BY_NAME[os.environ.get("AGENTGYM_FAULT", "OOMKilled")]

    kube = KubeClient(kubeconfig)
    sim = KubeletSim(kube=kube, namespaces=[namespace])
    episode = Episode(
        control_plane=None,  # type: ignore[arg-type]  # unused on the read path
        kube=kube, sim=sim, namespace=namespace,
        topology=topology, fault=fault, params={}, seed=0,
    )
    episode.mesh = Mesh(kube=kube, sim=sim)
    episode.telemetry = Telemetry(kube=kube, sim=sim, mesh=episode.mesh)
    sim.reconcile(ticks=1)
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AgentGym MCP server over stdio")
    parser.add_argument("--server", required=True, choices=list(SERVERS))
    args = parser.parse_args()

    episode = _episode_from_env()
    asyncio.run(serve_stdio(args.server, lambda: episode))


if __name__ == "__main__":
    main()
