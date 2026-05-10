"""Streamlit chat app for Cisco Intersight, backed by a local Ollama model
and an Intersight MCP server.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from mcp_client import IntersightMCPClient, default_client_from_env
from orchestrator import Orchestrator, TurnEvent, TurnMetrics

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
OLLAMA_TAGS_URL = os.environ.get("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
INTERSIGHT_BASE_URL = os.environ.get("INTERSIGHT_BASE_URL", "https://intersight.com")

st.set_page_config(
    page_title="Intersight Chat",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- MCP singleton

@st.cache_resource(show_spinner="Starting Intersight MCP server…")
def get_mcp_client() -> IntersightMCPClient:
    """Spawned once per Streamlit server process; survives reruns."""
    client = default_client_from_env()
    client.start()
    atexit.register(client.stop)
    return client


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(
        mcp_client=get_mcp_client(),
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_api_key=OLLAMA_API_KEY,
    )


# ---------------------------------------------------------------- Ollama models

def fetch_ollama_models() -> tuple[list[str], str | None]:
    """Returns (models, error_message). Empty list on failure."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        names = sorted({m.get("name", "") for m in data.get("models", []) if m.get("name")})
        return names, None
    except requests.exceptions.ConnectionError:
        return [], f"Could not reach Ollama at {OLLAMA_TAGS_URL}. Is `ollama serve` running?"
    except Exception as exc:
        return [], f"Failed to list Ollama models: {exc}"


# ---------------------------------------------------------------- session state

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # full history sent to model (incl system, tool msgs)
    ss.setdefault("display", [])    # UI-only: list of {role, content, tool_calls?}
    ss.setdefault("selected_model", None)
    ss.setdefault("model_switch_pending", None)
    ss.setdefault("key_id", "")
    ss.setdefault("pem_bytes", None)
    ss.setdefault("pem_filename", None)
    ss.setdefault("credentials_configured", False)
    ss.setdefault("connection_status", None)
    ss.setdefault("mask_key_id", True)


# ---------------------------------------------------------------- credentials

def configure_credentials_on_server() -> tuple[bool, str]:
    ss = st.session_state
    if not ss.key_id.strip() or not ss.pem_bytes:
        return False, "Provide both an API Key ID and a PEM file."
    try:
        pem_text = ss.pem_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False, "PEM file isn't valid UTF-8 text."
    client = get_mcp_client()
    res = client.configure_credentials(ss.key_id.strip(), pem_text, INTERSIGHT_BASE_URL)
    if res.is_error or (res.parsed and res.parsed.get("ok") is False):
        err = (res.parsed or {}).get("error") or res.text or "Unknown error"
        return False, f"configure_credentials failed: {err}"
    ss.credentials_configured = True
    return True, "Credentials configured."


def test_connection() -> tuple[str, str]:
    ok, msg = configure_credentials_on_server()
    if not ok:
        return "err", msg
    client = get_mcp_client()
    res = client.test_connection()
    parsed = res.parsed or {}
    if res.is_error or parsed.get("ok") is False:
        err = parsed.get("error") or res.text or "Unknown error"
        return "err", f"Test failed: {err}"
    accounts = (parsed.get("data") or {}).get("Results") or []
    if accounts:
        name = accounts[0].get("Name") or accounts[0].get("Moid") or "(unnamed)"
        return "ok", f"Connected. Account: {name}"
    return "ok", "Connected (no account name returned)."


# ---------------------------------------------------------------- sidebar UI

def render_sidebar() -> None:
    ss = st.session_state

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ---------- Model selector ----------
        st.markdown("### 1. Model")
        models, err = fetch_ollama_models()
        if err:
            st.error(err)
        if models:
            default_idx = (
                models.index(ss.selected_model)
                if ss.selected_model in models
                else 0
            )
            chosen = st.selectbox(
                "Local model (Ollama)",
                models,
                index=default_idx,
                key="model_select",
            )
            if ss.selected_model is None:
                ss.selected_model = chosen
            elif chosen != ss.selected_model and ss.model_switch_pending is None:
                ss.model_switch_pending = chosen

            if ss.model_switch_pending and ss.model_switch_pending != ss.selected_model:
                st.warning(
                    f"Switching from **{ss.selected_model}** to "
                    f"**{ss.model_switch_pending}**. Clear the conversation?"
                )
                col_a, col_b = st.columns(2)
                if col_a.button("Clear & switch", use_container_width=True):
                    ss.messages = []
                    ss.display = []
                    ss.selected_model = ss.model_switch_pending
                    ss.model_switch_pending = None
                    st.rerun()
                if col_b.button("Keep history", use_container_width=True):
                    ss.selected_model = ss.model_switch_pending
                    ss.model_switch_pending = None
                    st.rerun()
        else:
            st.info("No models found. Pull one with `ollama pull qwen2.5:7b`.")

        st.divider()

        # ---------- Key ID ----------
        st.markdown("### 2. Intersight API Key ID")
        ss.mask_key_id = st.checkbox("Mask Key ID", value=ss.mask_key_id)
        ss.key_id = st.text_input(
            "Key ID",
            value=ss.key_id,
            type="password" if ss.mask_key_id else "default",
            placeholder="e.g. 5f3.../5f3.../5f3...",
            label_visibility="collapsed",
        )

        st.divider()

        # ---------- PEM ----------
        st.markdown("### 3. Intersight PEM File")
        uploaded = st.file_uploader(
            "PEM file (kept in memory only)",
            type=["pem"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded is not None:
            ss.pem_bytes = uploaded.getvalue()
            ss.pem_filename = uploaded.name
        if ss.pem_bytes:
            st.success(f"✓ Loaded `{ss.pem_filename}` ({len(ss.pem_bytes)} bytes)")

        st.divider()

        # ---------- Connection test ----------
        st.markdown("### 4. Connection")
        if st.button("Test Connection", use_container_width=True, type="primary"):
            with st.spinner("Testing Intersight credentials…"):
                status, msg = test_connection()
                ss.connection_status = (status, msg)
        if ss.connection_status:
            status, msg = ss.connection_status
            (st.success if status == "ok" else st.error)(msg)

        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            ss.messages = []
            ss.display = []
            st.rerun()


# ---------------------------------------------------------------- chat rendering

def credentials_ready() -> bool:
    """Always re-push credentials so a container restart can't leave the
    server unconfigured while the browser still thinks it's set."""
    ss = st.session_state
    if ss.key_id.strip() and ss.pem_bytes:
        ok, _ = configure_credentials_on_server()
        return ok
    return False


def render_tool_call_block(tc: dict[str, Any]) -> None:
    name = tc.get("name", "?")
    is_err = tc.get("is_error", False)
    icon = "⚠️" if is_err else "🔧"
    with st.expander(f"{icon} {name}", expanded=False):
        args = tc.get("arguments") or {}
        if args:
            st.markdown("**Arguments**")
            st.code(json.dumps(args, indent=2), language="json")
        preview = tc.get("result_preview") or ""
        st.markdown("**Result**")
        try:
            parsed = json.loads(preview)
            st.code(json.dumps(parsed, indent=2), language="json")
        except Exception:
            st.code(preview)


def render_metrics_caption(metrics: dict[str, Any] | TurnMetrics | None) -> None:
    """One-line performance summary under an assistant turn.

    Accepts either a live `TurnMetrics` (from the orchestrator) or a dict
    (from `st.session_state.display` after a Streamlit rerun, where the
    metrics have been serialized via `dataclasses.asdict`).
    """
    if metrics is None:
        return
    if isinstance(metrics, TurnMetrics):
        completion_tokens = metrics.completion_tokens
        prompt_tokens = metrics.prompt_tokens
        total_seconds = metrics.total_seconds
        rounds = metrics.rounds
        ctx_max = metrics.ctx_max
        rate = metrics.tok_per_s
    else:
        completion_tokens = metrics.get("completion_tokens", 0)
        prompt_tokens = metrics.get("prompt_tokens", 0)
        total_seconds = metrics.get("total_seconds", 0.0)
        rounds = metrics.get("rounds", 0)
        ctx_max = metrics.get("ctx_max")
        ms = metrics.get("model_seconds", 0.0)
        rate = (completion_tokens / ms) if ms > 0 and completion_tokens > 0 else None

    # Skip the caption entirely if Ollama didn't return usage (older versions).
    if completion_tokens <= 0:
        return

    parts = [f"⚡ {total_seconds:.1f}s"]
    if rate is not None:
        parts.append(f"{rate:.0f} tok/s")
    if ctx_max:
        parts.append(f"{prompt_tokens:,} / {ctx_max:,} ctx")
    elif prompt_tokens:
        parts.append(f"{prompt_tokens:,} ctx")
    if rounds > 1:
        parts.append(f"{rounds} rounds")
    st.caption(" · ".join(parts))


def render_chat_history() -> None:
    for msg in st.session_state.display:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant":
                for tc in msg.get("tool_calls", []):
                    render_tool_call_block(tc)
            content = msg.get("content", "")
            if content:
                st.markdown(content)
            if role == "assistant":
                render_metrics_caption(msg.get("metrics"))


def handle_user_message(prompt: str) -> None:
    ss = st.session_state

    if not ss.selected_model:
        st.error("Pick a model in the sidebar first.")
        return
    if not credentials_ready():
        st.error("Configure your Intersight Key ID and PEM in the sidebar first.")
        return

    ss.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    orchestrator = get_orchestrator()
    tool_calls_for_display: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        status_box = st.status("Thinking…", expanded=False)
        text_placeholder = st.empty()
        # Buffer for the streamed assistant text. Cleared at the start of each
        # tool-calling round so only the final round's text survives in the slot.
        text_buf: list[str] = []

        def on_event(ev: TurnEvent) -> None:
            if ev.kind == "round_start":
                text_buf.clear()
                text_placeholder.empty()
            elif ev.kind == "assistant_delta":
                text_buf.append(ev.text)
                text_placeholder.markdown("".join(text_buf) + "▌")
            elif ev.kind == "assistant_text":
                text_placeholder.markdown(ev.text)
            elif ev.kind == "tool_call":
                status_box.update(
                    label=f"Querying Intersight: {ev.name}…", state="running"
                )
            elif ev.kind == "tool_result":
                icon = "⚠️" if ev.is_error else "✓"
                status_box.update(label=f"{icon} {ev.name}", state="running")
            elif ev.kind == "error":
                status_box.update(label="Error", state="error")

        try:
            final_text, record = orchestrator.run_turn(
                model=ss.selected_model,
                history=ss.messages,
                user_message=prompt,
                on_event=on_event,
            )
        except Exception as exc:
            status_box.update(label="Error", state="error")
            st.error(f"Turn failed: {exc}")
            return

        tool_calls_for_display = record.tool_calls
        for tc in tool_calls_for_display:
            render_tool_call_block(tc)

        if record.tool_calls:
            status_box.update(
                label=f"Done — {len(record.tool_calls)} tool call(s)",
                state="complete",
            )
        else:
            status_box.update(label="Done", state="complete")

        text_placeholder.markdown(final_text)
        render_metrics_caption(record.metrics)

    ss.display.append(
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": tool_calls_for_display,
            "metrics": dataclasses.asdict(record.metrics),
        }
    )


# ---------------------------------------------------------------- main

def main() -> None:
    init_state()

    try:
        get_mcp_client()
    except Exception as exc:
        st.error(
            f"Failed to start the Intersight MCP server: {exc}\n\n"
            "Did you run `npm install` and `npm run build` in `mcp-server/`?"
        )
        st.stop()

    render_sidebar()

    st.title("🛰️ Intersight Chat")
    st.caption(
        "Local chat with Cisco Intersight via Ollama + an MCP server. "
        "Credentials stay in memory only."
    )

    if not st.session_state.selected_model:
        st.info("Choose an Ollama model in the sidebar to begin.")
    elif not (st.session_state.key_id.strip() and st.session_state.pem_bytes):
        st.info(
            "Add your Intersight API Key ID and PEM file in the sidebar, then "
            "click **Test Connection**."
        )

    render_chat_history()

    prompt = st.chat_input("Ask about your Intersight environment…")
    if prompt:
        handle_user_message(prompt)


if __name__ == "__main__":
    main()
