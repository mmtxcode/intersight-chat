"""Streamlit chat app for Cisco Intersight, backed by a local Ollama model
and an Intersight MCP server.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import atexit
import base64
import dataclasses
import json
import os
import threading
import time
from datetime import datetime
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from mcp_client import IntersightMCPClient, default_client_from_env
from orchestrator import Orchestrator, TurnEvent, TurnMetrics
from reports import FORMATTER_SYSTEM_PROMPT, PRESET_REPORTS, ReportSpec

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
    ss.setdefault("warmed_model", None)


# ---------------------------------------------------------------- model pre-warm

def _ollama_native_base() -> str:
    """Strip the OpenAI-compat `/v1` suffix off OLLAMA_BASE_URL."""
    base = OLLAMA_BASE_URL.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def prewarm_model_async(model: str) -> None:
    """Kick off model loading in a daemon thread.

    Sends an empty generation to Ollama's native /api/generate with
    keep_alive=24h, which loads the model into VRAM and pins it. The first
    user prompt then skips the ~30s cold-load cost. Best-effort: failures
    (Ollama still booting, network blip, etc.) are silent — the user's
    first prompt just pays the load cost normally.

    Ollama dedupes loads internally, so re-firing this against an
    already-loaded model is a fast no-op.
    """
    url = f"{_ollama_native_base()}/api/generate"

    def _run() -> None:
        try:
            requests.post(
                url,
                json={"model": model, "prompt": "", "keep_alive": "24h"},
                timeout=180,
            )
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


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


_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"


@st.cache_data(show_spinner=False, max_entries=200)
def _markdown_to_pdf_bytes(text: str) -> bytes:
    """Render assistant markdown to PDF using fpdf2.

    Pure-Python — no native deps. fpdf2's write_html supports the markdown
    subset we care about (headings, paragraphs, lists, tables, emphasis,
    code spans). Cached on the markdown text so re-rendering chat history
    on every Streamlit rerun doesn't re-pay generation cost.

    We prefer DejaVu Sans (installed via fonts-dejavu-core in the Docker
    image) for full Unicode coverage, and fall back to fpdf2's built-in
    Helvetica core font when running outside Docker on a host without the
    font (e.g. local dev). In the Helvetica fallback we strip non-Latin-1
    chars so write_html doesn't choke on em-dashes, smart quotes, etc.
    """
    import markdown as md
    from fpdf import FPDF

    body_html = md.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    # fpdf2's HTML renderer doesn't draw table borders unless asked.
    body_html = body_html.replace("<table>", '<table border="1">')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    regular = os.path.join(_DEJAVU_DIR, "DejaVuSans.ttf")
    bold = os.path.join(_DEJAVU_DIR, "DejaVuSans-Bold.ttf")
    italic = os.path.join(_DEJAVU_DIR, "DejaVuSans-Oblique.ttf")
    bold_italic = os.path.join(_DEJAVU_DIR, "DejaVuSans-BoldOblique.ttf")

    if os.path.exists(regular):
        # Register all four style slots fpdf2 may switch into. If a variant
        # file is missing (e.g. fonts-dejavu-extra not installed), fall back
        # to the regular file under that slot so emphasis renders in regular
        # weight instead of crashing with "Undefined font: dejavuI". Bold
        # falls back to bold-or-regular, bold-italic to bold-or-regular.
        pdf.add_font("DejaVu", "", regular)
        pdf.add_font("DejaVu", "B", bold if os.path.exists(bold) else regular)
        pdf.add_font("DejaVu", "I", italic if os.path.exists(italic) else regular)
        pdf.add_font(
            "DejaVu",
            "BI",
            bold_italic if os.path.exists(bold_italic)
            else (bold if os.path.exists(bold) else regular),
        )
        pdf.set_font("DejaVu", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
        body_html = body_html.encode("latin-1", errors="replace").decode("latin-1")

    pdf.write_html(body_html)
    return bytes(pdf.output())


def render_action_buttons(text: str, key: str) -> None:
    """Per-message action row: 📋 Copy + 📄 View PDF.

    Both buttons live in a single `components.html` iframe so they share
    styling and lay out cleanly side-by-side. Each is JS-driven:

      * Copy uses the Clipboard API with a `document.execCommand('copy')`
        fallback so it also works over plain HTTP (Clipboard API requires
        a secure context).
      * View PDF builds a `Blob` from the cached PDF bytes (base64-decoded
        in the iframe), calls `URL.createObjectURL`, and `window.open`s
        the blob URL in a new tab — so the user sees the report rendered
        in the browser's built-in PDF viewer, where they can save/print
        from the viewer's toolbar. If the popup is blocked, we fall back
        to a programmatic <a download> click so the user still gets the
        file (just as a download instead of a view).

    We used to render PDF via `st.download_button`, which is reliable but
    always forces a download — no way to make it open in the viewer first.
    The blob+window.open approach gives us that and only loses the safety
    net for the (rare) popup-blocked case, which the JS fallback handles.
    """
    if not text:
        return

    # PDF generation is cached on `text`, so re-rendering history is cheap.
    pdf_b64: str
    pdf_available: bool
    try:
        pdf_bytes = _markdown_to_pdf_bytes(text)
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        pdf_available = True
    except Exception as exc:  # noqa: BLE001 — surface failure as a label
        _ = exc
        pdf_b64 = ""
        pdf_available = False

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"intersight-{ts}.pdf"

    # `</` inside a JS string literal would close the surrounding <script>
    # tag, so escape it before embedding the JSON.
    text_js = json.dumps(text).replace("</", "<\\/")
    pdf_js = json.dumps(pdf_b64)
    filename_js = json.dumps(filename)

    btn_style = (
        "background:transparent;"
        "border:1px solid rgba(128,128,128,0.25);"
        "border-radius:6px;"
        "padding:3px 10px;"
        "font-size:12px;"
        "cursor:pointer;"
        "color:rgba(140,140,140,1);"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    )
    disabled_style = btn_style + "opacity:0.45;cursor:not-allowed;"

    pdf_html = (
        f'<button id="pdf" type="button" style="{btn_style}">📄 View PDF</button>'
        if pdf_available
        else f'<span style="{disabled_style}">📄 PDF unavailable</span>'
    )

    html = f"""
    <div style="display:flex;gap:6px;margin:2px 0 0 0">
      <button id="cp" type="button" style="{btn_style}">📋 Copy</button>
      {pdf_html}
    </div>
    <script>
      const TEXT = {text_js};
      const PDF_B64 = {pdf_js};
      const FILENAME = {filename_js};

      // ---- Copy ----
      const cp = document.getElementById("cp");
      cp.addEventListener("click", () => {{
        const flash = (ok) => {{
          const orig = cp.innerText;
          cp.innerText = ok ? "✓ Copied" : "× Failed";
          setTimeout(() => {{ cp.innerText = orig; }}, 1500);
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

      // ---- View PDF ----
      const pdfBtn = document.getElementById("pdf");
      if (pdfBtn) {{
        pdfBtn.addEventListener("click", () => {{
          try {{
            const bytes = Uint8Array.from(atob(PDF_B64), c => c.charCodeAt(0));
            const blob = new Blob([bytes], {{type: "application/pdf"}});
            const url = URL.createObjectURL(blob);
            const win = window.open(url, "_blank", "noopener");
            if (!win) {{
              // Popup blocked — fall back to triggering a download so the
              // user at least gets the file.
              const a = document.createElement("a");
              a.href = url;
              a.download = FILENAME;
              a.style.display = "none";
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
            }}
            // Give the new tab time to load before revoking the blob URL.
            setTimeout(() => URL.revokeObjectURL(url), 60_000);
          }} catch (e) {{
            console.error("PDF open failed:", e);
            const orig = pdfBtn.innerText;
            pdfBtn.innerText = "× Failed";
            setTimeout(() => {{ pdfBtn.innerText = orig; }}, 2000);
          }}
        }});
      }}
    </script>
    """
    components.html(html, height=40)


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


# ---------------------------------------------------------------- preset chips

def render_preset_chips() -> None:
    """Row of preset-report chips just above the chat input.

    Each chip maps to a ReportSpec in PRESET_REPORTS. Clicking one queues
    the report's label in session state and triggers a rerun; main() picks
    it up and routes through handle_preset_report (deterministic data
    gather in Python → LLM formats the result). Disabled until the user
    has selected a model and entered credentials, so clicks don't fall
    through to a confusing "pick a model" error.
    """
    if not PRESET_REPORTS:
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
    n = len(PRESET_REPORTS)
    chip_w = 3
    spacer_w = max(1, 12 - chip_w * n)
    cols = st.columns([chip_w] * n + [spacer_w])

    for (label, _spec), col in zip(PRESET_REPORTS.items(), cols):
        with col:
            if st.button(
                label,
                key=f"preset-{label}",
                use_container_width=True,
                disabled=not ready,
                help=help_text,
            ):
                ss.queued_report = label
                st.rerun()


def handle_preset_report(spec: ReportSpec) -> None:
    """Run a deterministic report: gather data in Python, then ask the LLM
    only to format it as markdown.

    Mirrors handle_user_message's UI shell (user bubble + assistant bubble
    with streaming text, status box, metrics caption, Copy/PDF actions) so
    the result blends into chat history like any other turn.
    """
    ss = st.session_state

    if not ss.selected_model:
        st.error("Pick a model in the sidebar first.")
        return
    if not credentials_ready():
        st.error("Configure your Intersight Key ID and PEM in the sidebar first.")
        return

    ss.display.append({"role": "user", "content": spec.user_message})
    with st.chat_message("user"):
        st.markdown(spec.user_message)

    orchestrator = get_orchestrator()
    mcp = get_mcp_client()

    with st.chat_message("assistant"):
        status_box = st.status("Preparing report…", expanded=False)
        text_placeholder = st.empty()
        text_buf: list[str] = []

        # Phase 1 — Python gathers Intersight data deterministically.
        gather_started = time.perf_counter()

        def progress(label: str) -> None:
            status_box.update(label=label, state="running")

        try:
            data = spec.gather(mcp, progress)
        except Exception as exc:
            status_box.update(label="Error", state="error")
            st.error(f"Data gathering failed: {exc}")
            return

        gather_elapsed = time.perf_counter() - gather_started
        status_box.update(
            label=f"Formatting report ({gather_elapsed:.1f}s gathered)…",
            state="running",
        )

        # Phase 2 — LLM formats the pre-computed data.
        format_user_message = spec.format_prompt(data)

        def on_event(ev: TurnEvent) -> None:
            if ev.kind == "round_start":
                text_buf.clear()
                text_placeholder.empty()
            elif ev.kind == "assistant_delta":
                text_buf.append(ev.text)
                text_placeholder.markdown("".join(text_buf) + "▌")
            elif ev.kind == "assistant_text":
                text_placeholder.markdown(ev.text)
            elif ev.kind == "error":
                status_box.update(label="Error", state="error")

        try:
            final_text, record = orchestrator.run_format_turn(
                model=ss.selected_model,
                history=ss.messages,
                user_history_message=spec.user_message,
                format_system_prompt=FORMATTER_SYSTEM_PROMPT,
                format_user_message=format_user_message,
                on_event=on_event,
            )
        except Exception as exc:
            status_box.update(label="Error", state="error")
            st.error(f"Format step failed: {exc}")
            return

        status_box.update(
            label=f"Done — gathered in {gather_elapsed:.1f}s, formatted in "
            f"{record.metrics.model_seconds:.1f}s",
            state="complete",
        )
        text_placeholder.markdown(final_text)
        render_metrics_caption(record.metrics)
        render_action_buttons(final_text, key=f"live-{len(ss.display)}")

    ss.display.append(
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": [],
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

    # Pre-warm the selected model into VRAM in the background so the user's
    # first prompt doesn't pay the cold-load cost (~30s for qwen2.5:14b on
    # an L40S). Fires once per (session, model) — Ollama dedupes anyway.
    ss = st.session_state
    if ss.selected_model and ss.selected_model != ss.warmed_model:
        prewarm_model_async(ss.selected_model)
        ss.warmed_model = ss.selected_model

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

    # Run any preset-report queued by a chip click on the previous rerun.
    # This appends user + assistant messages to the chat in the natural
    # document flow, so they appear directly under the existing chat history.
    queued_report_label = st.session_state.pop("queued_report", None)
    if queued_report_label:
        spec = PRESET_REPORTS.get(queued_report_label)
        if spec is not None:
            handle_preset_report(spec)

    render_preset_chips()

    prompt = st.chat_input("Ask about your Intersight environment…")
    if prompt:
        handle_user_message(prompt)


if __name__ == "__main__":
    main()
