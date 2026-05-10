"""Streamlit chat app for Cisco Intersight, backed by a local Ollama model
and an Intersight MCP server.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import atexit
import dataclasses
import io
import json
import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components
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


@st.cache_data(show_spinner=False, max_entries=200)
def _markdown_to_pdf_bytes(text: str) -> bytes:
    """Render assistant markdown to a styled PDF.

    Cached on the markdown text so re-rendering chat history (which happens
    on every Streamlit rerun) doesn't re-generate the same PDF. Imports are
    inside the function to keep app startup snappy when no one ever clicks.
    """
    import markdown as md
    from xhtml2pdf import pisa

    body = md.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    html = f"""<html><head><meta charset="utf-8"><style>
      @page {{ size: letter; margin: 0.75in; }}
      body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #1a1a1a; }}
      h1, h2, h3, h4 {{ color: #14365d; margin: 12pt 0 6pt 0; }}
      h1 {{ font-size: 18pt; }}
      h2 {{ font-size: 14pt; }}
      h3 {{ font-size: 12pt; }}
      h4 {{ font-size: 11pt; }}
      p  {{ margin: 6pt 0; }}
      ul, ol {{ margin: 6pt 0 6pt 16pt; }}
      code {{ background: #f4f4f4; padding: 1pt 4pt; font-family: Courier, monospace; font-size: 10pt; }}
      pre  {{ background: #f4f4f4; padding: 8pt; font-family: Courier, monospace; font-size: 9pt; white-space: pre-wrap; }}
      table {{ border-collapse: collapse; margin: 8pt 0; }}
      th, td {{ border: 1px solid #ccc; padding: 4pt 8pt; font-size: 10pt; }}
      th {{ background: #eaeaea; }}
      blockquote {{ border-left: 3pt solid #ccc; margin: 8pt 0; padding: 0 0 0 10pt; color: #555; }}
    </style></head><body>{body}</body></html>"""
    buf = io.BytesIO()
    result = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} errors")
    return buf.getvalue()


def render_pdf_button(text: str, key: str) -> None:
    """Download-as-PDF button for an assistant message.

    Uses Streamlit's native download_button (lives in the main page context,
    so downloads always work — no iframe sandbox quirks). PDF generation is
    cached, so the first render of a message pays ~100-300ms and every
    rerun after that is instant.
    """
    if not text:
        return
    try:
        pdf_bytes = _markdown_to_pdf_bytes(text)
    except Exception as exc:
        st.caption(f"PDF unavailable ({exc})")
        return
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "📄 PDF",
        data=pdf_bytes,
        file_name=f"intersight-{ts}.pdf",
        mime="application/pdf",
        key=f"pdf-{key}",
        use_container_width=True,
    )


def render_action_buttons(text: str, key: str) -> None:
    """Row of per-message actions (Copy, Download PDF, …)."""
    if not text:
        return
    cols = st.columns([1, 1, 8])
    with cols[0]:
        render_copy_button(text)
    with cols[1]:
        render_pdf_button(text, key=key)


def render_copy_button(text: str) -> None:
    """Small "Copy" button under an assistant message.

    Streamlit strips inline event handlers from `st.markdown(...,
    unsafe_allow_html=True)`, so the button has to live inside an iframe
    via `components.html`. We try the modern Clipboard API first and fall
    back to `document.execCommand('copy')` so this also works when the app
    is served over plain HTTP (Clipboard API requires a secure context —
    HTTPS or localhost).
    """
    if not text:
        return
    # `</` inside a JS string literal would close the surrounding <script>
    # tag, so escape it before embedding the JSON.
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <div style="margin:2px 0 0 0">
      <button id="cp" type="button" style="
        background: transparent;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        cursor: pointer;
        color: rgba(140,140,140,1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      ">📋 Copy</button>
    </div>
    <script>
      const TEXT = {payload};
      const btn = document.getElementById("cp");
      btn.addEventListener("click", () => {{
        const flash = (ok) => {{
          const orig = btn.innerText;
          btn.innerText = ok ? "✓ Copied" : "× Failed";
          setTimeout(() => {{ btn.innerText = orig; }}, 1500);
        }};
        const fallback = () => {{
          const ta = document.createElement("textarea");
          ta.value = TEXT;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          let ok = false;
          try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
          document.body.removeChild(ta);
          flash(ok);
        }};
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(TEXT)
            .then(() => flash(true))
            .catch(() => fallback());
        }} else {{
          fallback();
        }}
      }});
    </script>
    """
    components.html(html, height=32)


def render_chat_history() -> None:
    for i, msg in enumerate(st.session_state.display):
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
                render_action_buttons(content, key=f"hist-{i}")


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
        render_action_buttons(final_text, key=f"live-{len(ss.display)}")

    ss.display.append(
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": tool_calls_for_display,
            "metrics": dataclasses.asdict(record.metrics),
        }
    )


# ---------------------------------------------------------------- presets

INVENTORY_REPORT_PROMPT = """\
Generate a comprehensive infrastructure inventory report for an Intersight
administrator. The audience is an engineer who manages and supports
Intersight, so the report should help answer "what do I have, what state is
it in, and what needs attention" at a glance.

Gather data using these tools (use defaults; pass top=200 if you suspect
truncation):
- get_chassis
- get_compute_blades
- get_compute_rack_units
- get_fabric_interconnects
- get_alarm_summary
- get_hcl_status
- get_server_profiles

Then produce a markdown report with the following structure. Use # for the
title, ## for sections, ### for subsections. Use markdown tables (pipe
syntax) for tabular data. Be concise — favor bullets and tables, not prose.

# Intersight Inventory Report

## Executive Summary
A bulleted overview with the headline numbers:
- Total servers: blades + rack units, with each subtotal
- Power state: how many On vs Off (use OperPowerState on blades and the
  equivalent on rack units; fall back to AdminPowerState if needed)
- Operational health: count by state (Operable/Healthy, Warning/Degraded,
  Inoperable/Critical, Other)
- Total chassis and total slots used vs available
- Fabric interconnects total
- Server profiles: assigned vs unassigned
- Active alarms by severity
- HCL compliance counts

## Servers

### By power state
A markdown table with columns: State | Count | %.

### By operational state
A markdown table with columns: State | Count | %.

### Top server models
A markdown table of the top 5 models by count: Model | Count.

### Servers needing attention
Any server whose OperState is not Operable/Healthy. Table:
Name | Model | OperState | PowerState | Chassis (or "N/A" for rack units).
If none, write "All servers are healthy."

## Chassis
A table per chassis: Name | Model | OperState | Total Slots | Slots Used |
Slots Free. Compute Slots Used as the count of blades whose Chassis.Moid
matches this chassis's Moid (you have to join the blades data to the
chassis data yourself). Slots Free = Total Slots − Slots Used. Sort by
chassis name.

If there are no chassis (rack-only environment), write
"No chassis present (rack-only environment)."

## Fabric Interconnects
A table: Name | Model | Serial | OperState. If none, say so.

## Server Profiles
- Total: <X>
- Assigned: <X> (profiles with a non-empty AssignedServer reference)
- Unassigned: <X>

If 10 or fewer unassigned, list them by name in a sub-bullet. Otherwise
give the count and the first 10 names.

## Active Alarms
A bullet list of counts per severity from get_alarm_summary. If no active
alarms, write "No active alarms."

## HCL Compliance
A bullet list of counts per HCL status from get_hcl_status (Validated,
Incomplete, Not-Validated, Not-Listed, etc.). If get_hcl_status returns
nothing, write "HCL data unavailable."

---

Format rules:
- Use markdown tables (pipe syntax) for all tabular data — never prose.
- Round percentages to whole numbers.
- If a tool returns no data, write "None" or "0" rather than omitting the
  whole section.
- Do NOT add a "Summary" or "Conclusion" section at the end — the executive
  summary at the top is sufficient.
"""


PRESET_PROMPTS: dict[str, str] = {
    "📦 Inventory Report": INVENTORY_REPORT_PROMPT,
}


def render_preset_chips() -> None:
    """Row of preset-prompt chips just above the chat input.

    Click queues the preset's prompt in session state and triggers a rerun;
    main() picks it up and feeds it through the same handle_user_message
    path as a typed prompt. Disabled until the user has selected a model
    and entered credentials, so clicks don't fall through to a confusing
    "pick a model" error.
    """
    if not PRESET_PROMPTS:
        return

    ss = st.session_state
    ready = bool(ss.selected_model) and credentials_ready()
    help_text = (
        None
        if ready
        else "Pick a model and add Intersight credentials in the sidebar "
        "to enable presets."
    )

    # Lay out chips left-aligned with a spacer column so they take their
    # natural width on wide screens instead of stretching across the page.
    n = len(PRESET_PROMPTS)
    chip_w = 3
    spacer_w = max(1, 12 - chip_w * n)
    cols = st.columns([chip_w] * n + [spacer_w])

    for (label, preset_prompt), col in zip(PRESET_PROMPTS.items(), cols):
        with col:
            if st.button(
                label,
                key=f"preset-{label}",
                use_container_width=True,
                disabled=not ready,
                help=help_text,
            ):
                ss.queued_prompt = preset_prompt
                st.rerun()


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

    # Run any prompt queued by a preset chip on the previous rerun. This
    # appends user + assistant messages to the chat in the natural document
    # flow, so they appear directly under the existing chat history.
    queued = st.session_state.pop("queued_prompt", None)
    if queued:
        handle_user_message(queued)

    render_preset_chips()

    prompt = st.chat_input("Ask about your Intersight environment…")
    if prompt:
        handle_user_message(prompt)


if __name__ == "__main__":
    main()
