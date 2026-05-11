"""Deterministic report generation.

Each ReportSpec wires a chip label → a Python data-gathering function →
a markdown formatting prompt. The chip-click handler in app.py runs the
gather step, embeds the resulting summary dict in the formatting prompt,
and feeds the prompt through the orchestrator's `run_format_turn`, which
streams a single completion *without* exposing tools. The model can only
format, not call APIs — which eliminates the failure mode where small
tool-calling models narrate JSON tool calls as text or hit MAX_TOOL_ROUNDS.

To add a new report:
  1. Write a `gather_<name>_data(mcp, progress)` function that returns a
     compact summary dict (counts, joins, tallies — all done in Python).
  2. Write a `format_<name>_prompt(data)` function that produces a prompt
     the model can render as markdown.
  3. Add a `ReportSpec` entry to `PRESET_REPORTS`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from mcp_client import IntersightMCPClient


# A tiny callback that the gather step calls between tools so the UI can
# update its status indicator. `None` means "no UI; ignore."
ProgressCb = Callable[[str], None] | None


@dataclass
class ReportSpec:
    label: str          # Chip button label.
    slug: str           # Stable key used in PDF filenames and component keys.
    user_message: str   # Short, friendly user-turn message shown in chat.
    gather: Callable[[IntersightMCPClient, ProgressCb], dict[str, Any]]
    format_prompt: Callable[[dict[str, Any]], str]


# A focused system prompt for the format-only step. The default chat
# SYSTEM_PROMPT is about tool-call discipline, OData rules, etc. — none of
# which apply when the model is just rendering pre-built data.
FORMATTER_SYSTEM_PROMPT = """\
You are a report formatter for a Cisco Intersight administration tool.
Your ONLY job is to render the JSON data in the user message as a clean
markdown report.

Rules:
- Reply in English only.
- Output ONLY the markdown report — no preamble, no commentary, no sign-off.
- Use markdown tables (pipe syntax) for tabular data.
- Use the exact section structure and order given in the user message.
- Do NOT invent fields, counts, or values. Every number you write must
  come from the JSON data in the user message.
- Do NOT call any tools — none are available. All data is already present.
"""


# ---------------------------------------------------------------- helpers

def _call_tool(
    mcp: IntersightMCPClient,
    name: str,
    args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Call an MCP list-style tool and return its results array.

    Returns an empty list on any error (tool failure, JSON parse failure,
    unexpected shape). Report generation never blows up because one tool
    happened to return nothing.
    """
    try:
        res = mcp.call_tool(name, args or {})
        if res.is_error or not res.text:
            return []
        parsed = json.loads(res.text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "results" in parsed and isinstance(parsed["results"], list):
            return parsed["results"]
        # Tools sometimes return a top-level error envelope rather than
        # raising — treat those as "no data" for the report.
        if parsed.get("ok") is False:
            return []
    return []


def _tally(items: list[dict[str, Any]], field: str, *, default: str = "Unknown") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        value = item.get(field)
        counter[value if value else default] += 1
    # Preserve highest-count-first order; the formatter shows them in that order.
    return dict(counter.most_common())


def _percent(part: int, total: int) -> int:
    return round(100 * part / total) if total else 0


# ---------------------------------------------------------------- inventory: gather

# OperState values that count as "healthy" for the needs-attention table.
# Intersight uses different casings across resource types, so cover both.
_HEALTHY_OPER_STATES = {
    "Operable", "operable", "Healthy", "healthy", "ok", "Ok", "OK",
}


def gather_inventory_data(mcp: IntersightMCPClient, progress: ProgressCb = None) -> dict[str, Any]:
    """Pull every Intersight resource needed for the inventory report and
    roll the raw data into a compact summary dict.

    All counts, joins, and percentages happen here in Python, so the LLM's
    only job is to render the dict as markdown.
    """
    def step(label: str) -> None:
        if progress is not None:
            progress(label)

    step("Querying chassis…")
    chassis = _call_tool(mcp, "get_chassis", {"top": 200})
    step("Querying blade servers…")
    blades = _call_tool(mcp, "get_compute_blades", {"top": 500})
    step("Querying rack servers…")
    rack_units = _call_tool(mcp, "get_compute_rack_units", {"top": 500})
    step("Querying fabric interconnects…")
    fis = _call_tool(mcp, "get_fabric_interconnects", {"top": 200})
    step("Querying server profiles…")
    profiles = _call_tool(mcp, "get_server_profiles", {"top": 500})
    step("Querying alarm summary…")
    alarms_raw = _call_tool(mcp, "get_alarm_summary", {})
    step("Querying HCL status…")
    hcl = _call_tool(mcp, "get_hcl_status", {"top": 500})

    # ---- Aggregations
    all_servers = list(blades) + list(rack_units)
    total = len(all_servers)

    # blade → chassis Moid for slot-occupancy join
    slots_used = Counter()
    for b in blades:
        ch_ref = b.get("Chassis")
        moid = ch_ref.get("Moid") if isinstance(ch_ref, dict) else None
        if moid:
            slots_used[moid] += 1

    chassis_rows = []
    for c in chassis:
        moid = c.get("Moid")
        num_slots = int(c.get("NumSlots") or 0)
        used = slots_used.get(moid, 0)
        chassis_rows.append({
            "name": c.get("Name") or "(unnamed)",
            "model": c.get("Model") or "Unknown",
            "oper_state": c.get("OperState") or "Unknown",
            "num_slots": num_slots,
            "slots_used": used,
            "slots_free": max(num_slots - used, 0),
        })
    chassis_rows.sort(key=lambda x: x["name"])

    power_tally = _tally(all_servers, "OperPowerState")
    oper_tally = _tally(all_servers, "OperState")
    model_tally = _tally(all_servers, "Model")

    needs_attention = []
    for s in all_servers:
        if (s.get("OperState") or "Unknown") not in _HEALTHY_OPER_STATES:
            ch_ref = s.get("Chassis")
            ch_name = ch_ref.get("Name") if isinstance(ch_ref, dict) else None
            needs_attention.append({
                "name": s.get("Name") or "(unnamed)",
                "model": s.get("Model") or "Unknown",
                "oper_state": s.get("OperState") or "Unknown",
                "power_state": s.get("OperPowerState") or "Unknown",
                "chassis": ch_name or "N/A",
            })

    fi_rows = [
        {
            "name": f.get("Name") or "(unnamed)",
            "model": f.get("Model") or "Unknown",
            "serial": f.get("Serial") or "",
            "oper_state": f.get("OperState") or "Unknown",
        }
        for f in fis
    ]

    assigned = [p for p in profiles if p.get("AssignedServer")]
    unassigned = [p for p in profiles if not p.get("AssignedServer")]

    # get_alarm_summary returns [{"Severity": "Critical", "Count": N}, …]
    alarm_counts: dict[str, int] = {}
    for entry in alarms_raw:
        sev = entry.get("Severity") or "Unknown"
        cnt = entry.get("Count") or entry.get("count") or 0
        alarm_counts[sev] = int(cnt)

    hcl_counts = _tally(hcl, "Status")

    return {
        "servers": {
            "total": total,
            "blades": len(blades),
            "rack_units": len(rack_units),
            "by_power_state": [
                {"state": k, "count": v, "percent": _percent(v, total)}
                for k, v in power_tally.items()
            ],
            "by_oper_state": [
                {"state": k, "count": v, "percent": _percent(v, total)}
                for k, v in oper_tally.items()
            ],
            "top_models": [
                {"model": k, "count": v}
                for k, v in list(model_tally.items())[:5]
            ],
            "needs_attention": needs_attention,
        },
        "chassis": {
            "total": len(chassis_rows),
            "rows": chassis_rows,
            "total_slots": sum(r["num_slots"] for r in chassis_rows),
            "slots_used": sum(r["slots_used"] for r in chassis_rows),
            "slots_free": sum(r["slots_free"] for r in chassis_rows),
        },
        "fabric_interconnects": {
            "total": len(fi_rows),
            "rows": fi_rows,
        },
        "server_profiles": {
            "total": len(profiles),
            "assigned": len(assigned),
            "unassigned": len(unassigned),
            "unassigned_names": [p.get("Name") or "(unnamed)" for p in unassigned[:10]],
            "unassigned_truncated": max(len(unassigned) - 10, 0),
        },
        "alarms": alarm_counts,
        "hcl": hcl_counts,
    }


# ---------------------------------------------------------------- inventory: format

# We pre-compute the executive-summary one-liners in Python so the model
# doesn't have to interpolate them from the JSON — that's the most error-prone
# part for smaller models.
_INVENTORY_FORMAT_TEMPLATE = """\
You will format the following Intersight inventory data as a clear, scannable
markdown report. The data is already gathered and pre-computed; you do NOT
need to call any tools and you do NOT need to compute totals or percentages
yourself. Just render the data using the structure below.

PRE-COMPUTED DATA (use this as the source of truth):
```json
{data_json}
```

OUTPUT — produce markdown matching this structure exactly. Use the values
from the JSON data above. No commentary, no preamble, no sign-off.

# Intersight Inventory Report

## Executive Summary
- **Servers:** {total} total ({blades} blades, {rack_units} rack units)
- **Power state:** {power_summary}
- **Health:** {health_summary}
- **Chassis:** {chassis_total} chassis with {slots_used}/{total_slots} slots used ({slots_free} free)
- **Fabric interconnects:** {fi_total}
- **Server profiles:** {assigned}/{profile_total} assigned ({unassigned} unassigned)
- **Active alarms:** {alarms_summary}
- **HCL compliance:** {hcl_summary}

## Servers

### By power state
Render `servers.by_power_state` as a markdown table with columns: State | Count | %.

### By operational state
Render `servers.by_oper_state` as a markdown table with columns: State | Count | %.

### Top server models
Render `servers.top_models` as a markdown table with columns: Model | Count.

### Servers needing attention
If `servers.needs_attention` is non-empty, render it as a markdown table with
columns: Name | Model | OperState | PowerState | Chassis. Otherwise write
the single line: **All servers are healthy.**

## Chassis
If `chassis.total` > 0, render `chassis.rows` as a markdown table with
columns: Name | Model | OperState | Total Slots | Slots Used | Slots Free.
Otherwise write the single line: **No chassis present (rack-only environment).**

## Fabric Interconnects
If `fabric_interconnects.total` > 0, render `fabric_interconnects.rows` as a
markdown table with columns: Name | Model | Serial | OperState. Otherwise
write the single line: **No fabric interconnects present.**

## Server Profiles
- Total: {profile_total}
- Assigned: {assigned}
- Unassigned: {unassigned}

If `server_profiles.unassigned_names` is non-empty, list each name as a
bullet under an indented sub-list. If `server_profiles.unassigned_truncated`
is greater than zero, add a final bullet: *…and N more (first 10 shown)*
where N is that value.

## Active Alarms
Render `alarms` as a bullet list of `severity: count`. If empty write:
**No active alarms.**

## HCL Compliance
Render `hcl` as a bullet list of `status: count`. If empty write:
**HCL data unavailable.**
"""


def format_inventory_prompt(data: dict[str, Any]) -> str:
    s = data["servers"]
    c = data["chassis"]
    p = data["server_profiles"]
    alarms = data["alarms"]
    hcl = data["hcl"]

    def _join(pairs: list[tuple[str, int]]) -> str:
        return ", ".join(f"{count} {label}" for label, count in pairs)

    power_summary = _join(
        [(r["state"], r["count"]) for r in s["by_power_state"]]
    ) or "n/a"
    health_summary = _join(
        [(r["state"], r["count"]) for r in s["by_oper_state"]]
    ) or "n/a"
    alarms_summary = _join(
        [(sev.lower(), cnt) for sev, cnt in alarms.items()]
    ) or "none"
    hcl_summary = _join(
        [(status.lower(), cnt) for status, cnt in hcl.items()]
    ) or "n/a"

    return _INVENTORY_FORMAT_TEMPLATE.format(
        data_json=json.dumps(data, indent=2),
        total=s["total"],
        blades=s["blades"],
        rack_units=s["rack_units"],
        power_summary=power_summary,
        health_summary=health_summary,
        chassis_total=c["total"],
        slots_used=c["slots_used"],
        total_slots=c["total_slots"],
        slots_free=c["slots_free"],
        fi_total=data["fabric_interconnects"]["total"],
        profile_total=p["total"],
        assigned=p["assigned"],
        unassigned=p["unassigned"],
        alarms_summary=alarms_summary,
        hcl_summary=hcl_summary,
    )


# ---------------------------------------------------------------- registry

INVENTORY_REPORT = ReportSpec(
    label="📦 Inventory Report",
    slug="inventory",
    user_message="Generate an Intersight inventory report.",
    gather=gather_inventory_data,
    format_prompt=format_inventory_prompt,
)


PRESET_REPORTS: dict[str, ReportSpec] = {
    INVENTORY_REPORT.label: INVENTORY_REPORT,
}
