"""Conversation loop: drives an Ollama chat model that issues tool calls
against the Intersight MCP server.

The model talks via Ollama's OpenAI-compatible Chat Completions endpoint.
Tool definitions are derived from the live MCP tool list, so adding tools
to the MCP server makes them automatically available to the model with no
Python changes needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from openai import OpenAI

from mcp_client import IntersightMCPClient, ToolSpec


SYSTEM_PROMPT = """\
You are a Cisco Intersight assistant. The user manages compute, network,
and storage infrastructure through Intersight, and you help them inspect
and reason about it.

You have tools that call the Intersight REST API. Follow these rules
exactly — most failures come from breaking them.

TOOL USE
- Always issue tool calls through the function-calling interface. NEVER
  print a tool-call JSON object as text in your reply. If you decide a
  tool is needed, call it; do not narrate it.
- Prefer the most specific tool (e.g. get_server_profiles over
  generic_api_call). Use generic_api_call only for endpoints not covered
  by a dedicated tool.
- Chain tool calls when needed: list, then drill in by Moid. Don't ask
  the user for data you can fetch yourself.

ODATA QUERY PARAMETERS
- $filter, $top, $skip, $orderby are safe to use freely.
- $select is dangerous: ONLY use field names you have already seen in a
  prior tool result for that resource type. Never invent dotted paths
  like 'DeviceInfo.Sku.Inventory.AvailableSlots'. If you don't know the
  exact field, omit $select entirely — every list tool already returns
  a curated default field set.

INTERSIGHT ERROR SEMANTICS
- Intersight sometimes returns HTTP 403 with `code: "InvalidUrl"` and
  message "Operation not supported. Check if the API path and method
  are valid." This is a REQUEST-VALIDATION error, not a permissions
  problem. It usually means a bad $select field, a wrong path, or a
  wrong method. Retry without $select, or with corrected fields.
- Genuine permission errors come back as 401 or 403 with a different
  code. Only then should you tell the user it's a credentials issue.

DERIVED ANSWERS
- "Available / free / empty slots" in a chassis: chassis.NumSlots minus
  the count of blades whose Chassis.Moid matches that chassis's Moid.
  Get chassis with get_chassis, blades with get_compute_blades, and do
  the arithmetic in your reply.

PRESENTATION
- Use markdown tables for lists. Don't dump raw JSON unless the user
  asks for it.
- If a query is ambiguous, ask one short clarifying question instead of
  guessing.
- If a tool errors, read the error string carefully — it now includes
  the Intersight error `code` and `message`. Explain plainly and try a
  corrected call when the error suggests one. Never invent data.

Credentials are already configured by the application; you do not need
to call configure_credentials.
"""

# Tools we hide from the model — these are managed by the host app, not the LLM.
HIDDEN_TOOLS = {"configure_credentials"}

# Hard cap on tool-call rounds per user turn. Higher = more headroom for
# legitimate multi-step questions; lower = faster failure on stuck models.
MAX_TOOL_ROUNDS = 12

# If the model issues the SAME tool call with the SAME arguments this many
# times in a row, we cut it off — it's looping, not making progress.
MAX_REPEAT_CALLS = 3


@dataclass
class TurnEvent:
    """Streamed during a turn so the UI can show progress."""

    kind: str  # "tool_call" | "tool_result" | "assistant_text" | "error"
    name: str = ""
    arguments: dict[str, Any] | None = None
    result_preview: str = ""
    is_error: bool = False
    text: str = ""


@dataclass
class TurnRecord:
    """Persisted on the message in chat history so the sidebar can replay it."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""


def mcp_tools_to_openai_schema(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Convert MCP ToolSpecs to OpenAI 'function' tool definitions."""
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.name in HIDDEN_TOOLS:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _safe_parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class Orchestrator:
    def __init__(
        self,
        mcp_client: IntersightMCPClient,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_api_key: str = "ollama",
    ) -> None:
        self.mcp = mcp_client
        self.client = OpenAI(base_url=ollama_base_url, api_key=ollama_api_key)

    def run_turn(
        self,
        model: str,
        history: list[dict[str, Any]],
        user_message: str,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> tuple[str, TurnRecord]:
        """Run one user turn end-to-end.

        Returns (final_assistant_text, TurnRecord). Mutates `history` in place
        with the user message, all assistant tool-call rounds, tool results,
        and the final assistant message — so the next turn picks up the full
        context.
        """
        record = TurnRecord()

        def emit(event: TurnEvent) -> None:
            if on_event is not None:
                on_event(event)

        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        messages = list(history)
        messages.append({"role": "user", "content": user_message})
        history.append({"role": "user", "content": user_message})

        tool_defs = mcp_tools_to_openai_schema(self.mcp.list_tools())
        recent_call_signatures: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    tool_choice="auto" if tool_defs else None,
                )
            except Exception as exc:
                msg = f"Ollama call failed: {exc}"
                emit(TurnEvent(kind="error", text=msg, is_error=True))
                history.append({"role": "assistant", "content": msg})
                record.final_text = msg
                return msg, record

            choice = completion.choices[0]
            assistant_msg = choice.message
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []

            if not tool_calls:
                final_text = assistant_msg.content or ""
                emit(TurnEvent(kind="assistant_text", text=final_text))
                history.append({"role": "assistant", "content": final_text})
                record.final_text = final_text
                return final_text, record

            assistant_history_entry: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_history_entry)
            history.append(assistant_history_entry)

            for tc in tool_calls:
                tool_name = tc.function.name
                args = _safe_parse_arguments(tc.function.arguments)

                # Loop detection: same tool + same args repeated MAX_REPEAT_CALLS
                # times in a row is the model stuck. Cut it off with a clear note.
                signature = f"{tool_name}::{json.dumps(args, sort_keys=True)}"
                recent_call_signatures.append(signature)
                if len(recent_call_signatures) > MAX_REPEAT_CALLS:
                    recent_call_signatures.pop(0)
                if (
                    len(recent_call_signatures) == MAX_REPEAT_CALLS
                    and len(set(recent_call_signatures)) == 1
                ):
                    msg = (
                        f"The model called `{tool_name}` with the same arguments "
                        f"{MAX_REPEAT_CALLS} times in a row — stopping to avoid a loop. "
                        "Try a different model (e.g. qwen2.5:14b) or a more "
                        "specific question."
                    )
                    emit(TurnEvent(kind="error", text=msg, is_error=True))
                    history.append({"role": "assistant", "content": msg})
                    record.final_text = msg
                    return msg, record

                emit(TurnEvent(kind="tool_call", name=tool_name, arguments=args))

                if tool_name in HIDDEN_TOOLS:
                    result_text = json.dumps(
                        {"ok": False, "error": f"Tool {tool_name} is not available to the model."}
                    )
                    is_error = True
                else:
                    try:
                        result = self.mcp.call_tool(tool_name, args)
                        result_text = result.text or json.dumps(
                            {"ok": result.ok, "error": "Empty response"}
                        )
                        is_error = result.is_error
                    except Exception as exc:
                        result_text = json.dumps({"ok": False, "error": str(exc)})
                        is_error = True

                emit(
                    TurnEvent(
                        kind="tool_result",
                        name=tool_name,
                        result_preview=_truncate(result_text, 500),
                        is_error=is_error,
                    )
                )

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                }
                messages.append(tool_msg)
                history.append(tool_msg)

                record.tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": args,
                        "result_preview": _truncate(result_text, 1500),
                        "is_error": is_error,
                    }
                )

        msg = (
            "Reached the maximum number of tool-calling rounds for one turn. "
            "Try rephrasing your question or breaking it into smaller steps."
        )
        emit(TurnEvent(kind="error", text=msg, is_error=True))
        history.append({"role": "assistant", "content": msg})
        record.final_text = msg
        return msg, record
