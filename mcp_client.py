"""Synchronous wrapper around the async MCP Python SDK.

Streamlit reruns the script top-to-bottom on every interaction and isn't
async-friendly. This module owns a long-lived asyncio loop running on a
dedicated daemon thread, and exposes a small synchronous API on top of it.
The loop owns the MCP ClientSession and the stdio child process, so the
subprocess and connection survive across Streamlit reruns.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    text: str
    is_error: bool = False
    parsed: Any = None


@dataclass
class _ClientState:
    session: ClientSession | None = None
    exit_stack: AsyncExitStack | None = None
    tools: list[ToolSpec] = field(default_factory=list)


class IntersightMCPClient:
    """Synchronous facade over the async MCP SDK.

    Lifecycle:
      client = IntersightMCPClient(command, args, env)
      client.start()                  # spawns child + initializes session
      tools = client.list_tools()
      result = client.call_tool("get_server_profiles", {"top": 5})
      client.stop()                   # tears down on app shutdown
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._state = _ClientState()
        self._started = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ start/stop

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, name="mcp-client-loop", daemon=True
            )
            self._thread.start()
            self._submit(self._async_start()).result()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started or self._loop is None:
                return
            try:
                self._submit(self._async_stop()).result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._started = False

    @property
    def started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------ public API

    def list_tools(self) -> list[ToolSpec]:
        self._require_started()
        return list(self._state.tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self._require_started()
        return self._submit(self._async_call_tool(name, arguments)).result()

    def configure_credentials(
        self, key_id: str, pem: str, base_url: str = "https://intersight.com"
    ) -> ToolResult:
        return self.call_tool(
            "configure_credentials",
            {"key_id": key_id, "pem": pem, "base_url": base_url},
        )

    def test_connection(self) -> ToolResult:
        return self.call_tool("test_connection", {})

    # ------------------------------------------------------------------ internals

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("MCP client not started. Call start() first.")

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def _submit(self, coro):
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _async_start(self) -> None:
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(self._params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_resp = await session.list_tools()
            self._state.tools = [
                ToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                for t in tools_resp.tools
            ]
            self._state.session = session
            self._state.exit_stack = stack
        except Exception:
            await stack.aclose()
            raise

    async def _async_stop(self) -> None:
        if self._state.exit_stack is not None:
            await self._state.exit_stack.aclose()
        self._state.session = None
        self._state.exit_stack = None
        self._state.tools = []

    async def _async_call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        session = self._state.session
        if session is None:
            return ToolResult(ok=False, text="MCP session not initialized.", is_error=True)
        try:
            resp = await session.call_tool(name, arguments)
        except Exception as exc:
            return ToolResult(ok=False, text=f"MCP call failed: {exc}", is_error=True)

        text_chunks: list[str] = []
        for item in resp.content:
            inner = getattr(item, "text", None)
            if inner is not None:
                text_chunks.append(inner)
            else:
                # Fall back to JSON representation for non-text content blocks.
                try:
                    text_chunks.append(json.dumps(item.model_dump(), default=str))
                except Exception:
                    text_chunks.append(str(item))
        text = "\n".join(text_chunks).strip()

        parsed: Any = None
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        is_error = bool(getattr(resp, "isError", False))
        return ToolResult(ok=not is_error, text=text, is_error=is_error, parsed=parsed)


# ---------------------------------------------------------------- factory helpers

def default_client_from_env() -> IntersightMCPClient:
    """Build a client from MCP_SERVER_CMD / MCP_SERVER_ARGS env vars.

    Defaults match the project layout:
      MCP_SERVER_CMD=node
      MCP_SERVER_ARGS=mcp-server/dist/index.js
    """
    command = os.environ.get("MCP_SERVER_CMD", "node")
    args_raw = os.environ.get("MCP_SERVER_ARGS", "mcp-server/dist/index.js")
    args = [a for a in args_raw.split() if a]
    cwd = os.path.dirname(os.path.abspath(__file__))
    return IntersightMCPClient(command=command, args=args, cwd=cwd)
