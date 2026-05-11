"""Conversation loop: drives an Ollama chat model that issues tool calls
against the Intersight MCP server.

The model talks via Ollama's OpenAI-compatible Chat Completions endpoint.
Tool definitions are derived from the live MCP tool list, so adding tools
to the MCP server makes them automatically available to the model with no
Python changes needed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from openai import OpenAI

from mcp_client import IntersightMCPClient, ToolSpec


# Keep models pinned in VRAM between turns so consecutive prompts don't pay
# the model-load cost. Ollama's default is 5 minutes.
KEEP_ALIVE = "24h"


def _log(msg: str) -> None:
    """Emit timing diagnostics to stderr; surfaces in `make logs`."""
    print(f"[orchestrator] {msg}", file=sys.stderr, flush=True)


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
- "Available / free / empty slots" in a chassis: PCIe nodes (UCSX-440P
  etc.) occupy chassis slots too, so a blades-only count is WRONG. The
  correct math is:
      used = (blades in this chassis) + (PCIe nodes whose paired blade is
              in this chassis)
      free = total_slots − used
  PCIe nodes do NOT reference a chassis directly — they reference their
  paired blade via `ComputeBlade` (with `Parent` as a fallback). Two-hop
  join: pci.Node -> compute.Blade -> equipment.Chassis. Call get_chassis,
  get_compute_blades, AND get_pci_nodes, then do the arithmetic yourself.
- Intersight does NOT always populate `NumSlots` on the chassis MO (often
  empty for X-Series). If NumSlots is 0 or missing, use known capacities
  by model: UCSX-9508 = 8 slots, UCSB-5108-AC2 = 8 slots. If the model is
  not in that list and NumSlots is missing, say so rather than guessing.

PRESENTATION
- ALWAYS reply in English. Use English for every section title, table
  header, bullet, and field label, even when summarizing structured data.
  This applies regardless of the language the user's question was asked
  in (default to English unless the user explicitly asks for another
  language).
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

    # "round_start"     — beginning of a new model-call round; UI should clear
    #                     any in-progress assistant text from the previous round
    # "assistant_delta" — incremental text token from a streaming completion
    # "tool_call"       — model issued a tool call
    # "tool_result"     — tool returned (or errored)
    # "assistant_text"  — final assistant text for the turn (authoritative)
    # "error"           — turn aborted with a user-visible error
    kind: str
    name: str = ""
    arguments: dict[str, Any] | None = None
    result_preview: str = ""
    is_error: bool = False
    text: str = ""


@dataclass
class TurnMetrics:
    """Per-turn performance numbers shown in the UI and logs.

    `model_seconds` is the wall time spent inside model_call rounds
    (sum across rounds), distinct from `total_seconds` which also covers
    tool execution and Streamlit overhead. `prompt_tokens` is the LAST
    round's prompt size — that's the peak context usage for the turn,
    after every tool result has been folded back into the messages.
    """

    total_seconds: float = 0.0
    model_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    ctx_max: int | None = None

    @property
    def tok_per_s(self) -> float | None:
        if self.model_seconds > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.model_seconds
        return None


@dataclass
class TurnRecord:
    """Persisted on the message in chat history so the sidebar can replay it."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    metrics: TurnMetrics = field(default_factory=TurnMetrics)


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
        # Stored separately so we can hit Ollama's native /api/show endpoint
        # for context-length metadata. The OpenAI-compat endpoint doesn't
        # expose that.
        self._ollama_base_url = ollama_base_url
        self._ctx_cache: dict[str, int | None] = {}

    def _get_model_context_max(self, model: str) -> int | None:
        """Return the model's max context length, or None on failure.

        Hits Ollama's `/api/show` once per model and caches the result.
        Used for the demo metrics caption in the UI.
        """
        if model in self._ctx_cache:
            return self._ctx_cache[model]
        api_base = self._ollama_base_url.rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
        try:
            import httpx  # transitive dep of openai

            with httpx.Client(timeout=5.0) as c:
                r = c.post(f"{api_base}/api/show", json={"name": model})
                r.raise_for_status()
                data = r.json()
            info = data.get("model_info") or {}
            for k, v in info.items():
                if k.endswith(".context_length"):
                    self._ctx_cache[model] = int(v)
                    return self._ctx_cache[model]
        except Exception as exc:
            _log(f"context_length lookup failed for {model}: {exc}")
        self._ctx_cache[model] = None
        return None

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
        turn_started = time.perf_counter()
        metrics = record.metrics
        metrics.ctx_max = self._get_model_context_max(model)

        def _finalize() -> None:
            metrics.total_seconds = time.perf_counter() - turn_started

        for round_idx in range(MAX_TOOL_ROUNDS):
            emit(TurnEvent(kind="round_start"))

            round_started = time.perf_counter()
            content_buf = ""
            tool_calls_acc: dict[int, dict[str, str]] = {}
            first_token_at: float | None = None

            last_usage = None
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    tool_choice="auto" if tool_defs else None,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"keep_alive": KEEP_ALIVE},
                )
                for chunk in stream:
                    # The usage chunk arrives at the end with an empty `choices`
                    # list, so capture it before the early-continue below.
                    if getattr(chunk, "usage", None):
                        last_usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if getattr(delta, "content", None):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        content_buf += delta.content
                        emit(TurnEvent(kind="assistant_delta", text=delta.content))

                    for tc_delta in (getattr(delta, "tool_calls", None) or []):
                        idx = tc_delta.index
                        slot = tool_calls_acc.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc_delta.id:
                            slot["id"] = tc_delta.id
                        fn = getattr(tc_delta, "function", None)
                        if fn is not None:
                            if fn.name:
                                slot["name"] = fn.name
                            if fn.arguments:
                                slot["arguments"] += fn.arguments
            except Exception as exc:
                msg = f"Ollama call failed: {exc}"
                emit(TurnEvent(kind="error", text=msg, is_error=True))
                history.append({"role": "assistant", "content": msg})
                record.final_text = msg
                _finalize()
                return msg, record

            completion_elapsed = time.perf_counter() - round_started
            ttft = (
                f"{first_token_at - round_started:.2f}s"
                if first_token_at is not None
                else "n/a"
            )
            metrics.model_seconds += completion_elapsed
            metrics.rounds += 1
            if last_usage is not None:
                # prompt_tokens grows each round as tool results are folded
                # back into messages, so the last round is the peak.
                metrics.prompt_tokens = last_usage.prompt_tokens
                metrics.completion_tokens += last_usage.completion_tokens
            _log(
                f"round={round_idx} model_call elapsed={completion_elapsed:.2f}s "
                f"ttft={ttft} content_chars={len(content_buf)} "
                f"tool_calls={len(tool_calls_acc)} "
                f"prompt_tokens={last_usage.prompt_tokens if last_usage else '?'} "
                f"completion_tokens={last_usage.completion_tokens if last_usage else '?'}"
            )

            # No tool calls means the model is done — content_buf is the final answer.
            if not tool_calls_acc:
                emit(TurnEvent(kind="assistant_text", text=content_buf))
                history.append({"role": "assistant", "content": content_buf})
                record.final_text = content_buf
                _finalize()
                rate_str = (
                    f"{metrics.tok_per_s:.1f}"
                    if metrics.tok_per_s is not None
                    else "n/a"
                )
                _log(
                    f"turn complete elapsed={metrics.total_seconds:.2f}s "
                    f"rounds={metrics.rounds} "
                    f"completion_tokens={metrics.completion_tokens} "
                    f"tok_per_s={rate_str}"
                )
                return content_buf, record

            sorted_tcs = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
            assistant_history_entry: dict[str, Any] = {
                "role": "assistant",
                "content": content_buf,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] or "{}",
                        },
                    }
                    for tc in sorted_tcs
                ],
            }
            messages.append(assistant_history_entry)
            history.append(assistant_history_entry)

            for tc in sorted_tcs:
                tool_name = tc["name"]
                tool_id = tc["id"]
                args = _safe_parse_arguments(tc["arguments"])

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
                    _finalize()
                    return msg, record

                emit(TurnEvent(kind="tool_call", name=tool_name, arguments=args))

                tool_started = time.perf_counter()
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
                tool_elapsed = time.perf_counter() - tool_started
                _log(
                    f"round={round_idx} tool={tool_name} elapsed={tool_elapsed:.2f}s "
                    f"result_chars={len(result_text)} is_error={is_error}"
                )

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
                    "tool_call_id": tool_id,
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
        _finalize()
        return msg, record

    def run_format_turn(
        self,
        *,
        model: str,
        history: list[dict[str, Any]],
        user_history_message: str,
        format_system_prompt: str,
        format_user_message: str,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> tuple[str, TurnRecord]:
        """Single streaming completion with no tools — used by deterministic
        report presets where Python has already gathered the data.

        Two design choices worth flagging:

        * The model call uses a standalone (system, user) message pair built
          from `format_system_prompt` + `format_user_message`. We do NOT mix
          in the existing chat history. That keeps the (potentially huge)
          pre-computed JSON blob out of subsequent turns and lets us swap
          the system prompt to a focused "you are a formatter" without
          mutating chat state.
        * Chat history gets a clean (user, assistant) pair: the short
          `user_history_message` (e.g. "Generate an Intersight inventory
          report.") and the model's formatted reply. So a follow-up turn
          can reference the report by what's visible in the chat without
          drowning in JSON.
        """
        record = TurnRecord()

        def emit(event: TurnEvent) -> None:
            if on_event is not None:
                on_event(event)

        # Keep the chat-history system prompt as-is (or insert the default
        # if this is the very first turn). The format-only call uses its
        # own (system, user) pair below.
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        history.append({"role": "user", "content": user_history_message})

        messages = [
            {"role": "system", "content": format_system_prompt},
            {"role": "user", "content": format_user_message},
        ]

        metrics = record.metrics
        metrics.ctx_max = self._get_model_context_max(model)
        turn_started = time.perf_counter()

        emit(TurnEvent(kind="round_start"))
        round_started = time.perf_counter()
        content_buf = ""
        first_token_at: float | None = None
        last_usage = None

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"keep_alive": KEEP_ALIVE},
            )
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    last_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_buf += delta.content
                    emit(TurnEvent(kind="assistant_delta", text=delta.content))
        except Exception as exc:
            msg = f"Ollama call failed: {exc}"
            emit(TurnEvent(kind="error", text=msg, is_error=True))
            history.append({"role": "assistant", "content": msg})
            record.final_text = msg
            metrics.total_seconds = time.perf_counter() - turn_started
            return msg, record

        completion_elapsed = time.perf_counter() - round_started
        ttft = (
            f"{first_token_at - round_started:.2f}s"
            if first_token_at is not None
            else "n/a"
        )
        metrics.model_seconds = completion_elapsed
        metrics.rounds = 1
        if last_usage is not None:
            metrics.prompt_tokens = last_usage.prompt_tokens
            metrics.completion_tokens = last_usage.completion_tokens
        rate_str = (
            f"{metrics.tok_per_s:.1f}"
            if metrics.tok_per_s is not None
            else "n/a"
        )
        _log(
            f"format model_call elapsed={completion_elapsed:.2f}s ttft={ttft} "
            f"content_chars={len(content_buf)} "
            f"prompt_tokens={last_usage.prompt_tokens if last_usage else '?'} "
            f"completion_tokens={last_usage.completion_tokens if last_usage else '?'} "
            f"tok_per_s={rate_str}"
        )

        emit(TurnEvent(kind="assistant_text", text=content_buf))
        history.append({"role": "assistant", "content": content_buf})
        record.final_text = content_buf
        metrics.total_seconds = time.perf_counter() - turn_started
        _log(f"format turn complete elapsed={metrics.total_seconds:.2f}s")
        return content_buf, record
