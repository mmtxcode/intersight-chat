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
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from mcp_client import IntersightMCPClient


def _log(msg: str) -> None:
    """Diagnostic logging surfaced in `make logs`."""
    print(f"[reports] {msg}", file=sys.stderr, flush=True)


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

    The Intersight MCP server wraps responses as:
      {"ok": true, "status": 200, "data": {"Results": [...], ...}}
    Errors come back as:
      {"ok": false, "status": 4xx, "error": "...", "data": {...}}

    Returns an empty list on any error (tool failure, JSON parse failure,
    `ok: false`, unexpected shape) and logs a diagnostic line so blank
    reports are easy to debug from `make logs`.
    """
    try:
        res = mcp.call_tool(name, args or {})
        if res.is_error or not res.text:
            _log(f"{name}: tool error or empty response")
            return []
        parsed = json.loads(res.text)
    except Exception as exc:
        _log(f"{name}: call failed: {exc}")
        return []

    if not isinstance(parsed, dict):
        _log(f"{name}: unexpected top-level shape {type(parsed).__name__}")
        return []
    if parsed.get("ok") is False:
        _log(f"{name}: ok=false error={parsed.get('error')!r}")
        return []

    data = parsed.get("data")
    if isinstance(data, dict) and isinstance(data.get("Results"), list):
        return data["Results"]
    if isinstance(data, list):
        return data
    _log(f"{name}: no Results array in response (data keys: {list(data) if isinstance(data, dict) else type(data).__name__})")
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


# Total slots per chassis model. Intersight does NOT expose total slot count
# on equipment.Chassis (NumSlots is often empty or 0 for X-Series), so we keep
# our own table here. If a chassis model isn't in this table, slot math falls
# back to NumSlots (if present) then to the highest SlotId observed across
# the fleet. Extend this table as new chassis models are encountered.
KNOWN_CAPACITY = {
    "UCSX-9508":     8,   # X-Series chassis
    "UCSB-5108-AC2": 8,   # B-Series chassis (8 half-width or 4 full-width)
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

    # The default $select for get_compute_blades / get_compute_rack_units
    # does NOT include OperState (the health field), so we override it
    # explicitly. Without this the operational-state table would be 100%
    # "Unknown" because the field would be absent from every row.
    BLADE_SELECT = "Name,Moid,Model,Chassis,OperState,OperPowerState"
    RACK_SELECT = "Name,Moid,Model,OperState,OperPowerState"

    step("Querying chassis…")
    chassis = _call_tool(mcp, "get_chassis", {"top": 200})
    step("Querying blade servers…")
    blades = _call_tool(mcp, "get_compute_blades", {"top": 500, "select": BLADE_SELECT})
    step("Querying rack servers…")
    rack_units = _call_tool(mcp, "get_compute_rack_units", {"top": 500, "select": RACK_SELECT})
    step("Querying PCIe nodes…")
    # X-Series PCIe nodes (e.g. UCSX-440P) occupy chassis slots but don't
    # reference a chassis directly — they reference their paired blade via
    # ComputeBlade. Two-hop join: pci.Node -> compute.Blade -> chassis.
    pcie_nodes = _call_tool(mcp, "get_pci_nodes", {"top": 500})
    step("Querying fabric interconnects…")
    fis = _call_tool(mcp, "get_fabric_interconnects", {"top": 200})
    step("Querying server profiles…")
    profiles = _call_tool(mcp, "get_server_profiles", {"top": 500})
    step("Querying alarm summary…")
    alarms_raw = _call_tool(mcp, "get_alarm_summary", {})
    step("Querying HCL status…")
    hcl = _call_tool(mcp, "get_hcl_status", {"top": 500})

    _log(
        f"inventory gather: chassis={len(chassis)} blades={len(blades)} "
        f"rack_units={len(rack_units)} pcie_nodes={len(pcie_nodes)} "
        f"fis={len(fis)} profiles={len(profiles)} "
        f"alarm_buckets={len(alarms_raw)} hcl={len(hcl)}"
    )

    # ---- Aggregations
    all_servers = list(blades) + list(rack_units)
    total = len(all_servers)

    # ---- Slot-occupancy join (blades + PCIe nodes).
    #
    # 1. Blades reference a chassis directly. Modern Intersight uses
    #    `EquipmentChassis`; older payloads use `Chassis`. Check both.
    # 2. PCIe nodes (UCSX-440P etc.) do NOT reference a chassis directly.
    #    They reference their paired blade via `ComputeBlade` (with `Parent`
    #    as a fallback). Two-hop join: PCIe node -> blade -> chassis.
    # 3. Both types share the same physical slot pool, so used = blades + PCIe.
    def _ref_moid(item: dict[str, Any], *keys: str) -> str | None:
        for k in keys:
            ref = item.get(k)
            if isinstance(ref, dict) and ref.get("Moid"):
                return ref["Moid"]
        return None

    blades_per_chassis: Counter[str] = Counter()
    blade_to_chassis: dict[str, str] = {}
    for b in blades:
        chassis_moid = _ref_moid(b, "EquipmentChassis", "Chassis")
        if chassis_moid:
            blades_per_chassis[chassis_moid] += 1
            blade_moid = b.get("Moid")
            if blade_moid:
                blade_to_chassis[blade_moid] = chassis_moid

    pcie_per_chassis: Counter[str] = Counter()
    for node in pcie_nodes:
        # Resolve PCIe node -> paired blade -> chassis.
        paired_blade_moid = _ref_moid(node, "ComputeBlade", "Parent")
        chassis_moid = blade_to_chassis.get(paired_blade_moid) if paired_blade_moid else None
        if chassis_moid:
            pcie_per_chassis[chassis_moid] += 1

    # Capacity per chassis model: KNOWN_CAPACITY first (authoritative), then
    # NumSlots from the chassis MO (often missing for X-Series, hence the
    # table), then highest observed SlotId for that model in the fleet as a
    # last-resort lower bound.
    observed_max_slot: dict[str, int] = {}
    for c in chassis:
        model = c.get("Model") or "Unknown"
        moid = c.get("Moid")
        # Walk every occupant of this chassis. SlotId is int on blades and
        # str on PCIe nodes — coerce defensively.
        occupant_slots: list[int] = []
        for b in blades:
            if _ref_moid(b, "EquipmentChassis", "Chassis") == moid:
                try:
                    occupant_slots.append(int(b.get("SlotId") or 0))
                except (TypeError, ValueError):
                    pass
        for node in pcie_nodes:
            paired = _ref_moid(node, "ComputeBlade", "Parent")
            if blade_to_chassis.get(paired) == moid:
                try:
                    occupant_slots.append(int(node.get("SlotId") or 0))
                except (TypeError, ValueError):
                    pass
        if occupant_slots:
            observed_max_slot[model] = max(
                observed_max_slot.get(model, 0), max(occupant_slots)
            )

    def _capacity_for(model: str, num_slots_field: int) -> int:
        if model in KNOWN_CAPACITY:
            return KNOWN_CAPACITY[model]
        if num_slots_field > 0:
            return num_slots_field
        return observed_max_slot.get(model, 0)

    chassis_rows = []
    for c in chassis:
        moid = c.get("Moid")
        model = c.get("Model") or "Unknown"
        num_slots_field = int(c.get("NumSlots") or 0)
        blade_count = blades_per_chassis.get(moid, 0)
        pcie_count = pcie_per_chassis.get(moid, 0)
        used = blade_count + pcie_count
        total = _capacity_for(model, num_slots_field)
        # Defensive: if reality exceeds our table, trust reality.
        if total and used > total:
            total = used
        chassis_rows.append({
            "name": c.get("Name") or "(unnamed)",
            "model": model,
            "oper_state": c.get("OperState") or "Unknown",
            "num_slots": total,
            "slots_used": used,
            "slots_used_blades": blade_count,
            "slots_used_pcie": pcie_count,
            "slots_free": max(total - used, 0) if total else 0,
            "capacity_known": total > 0,
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
            "slots_used_blades": sum(r["slots_used_blades"] for r in chassis_rows),
            "slots_used_pcie": sum(r["slots_used_pcie"] for r in chassis_rows),
            "slots_free": sum(r["slots_free"] for r in chassis_rows),
            "any_unknown_capacity": any(not r["capacity_known"] for r in chassis_rows),
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
- **Chassis:** {chassis_total} chassis with {slots_used}/{total_slots} slots used ({slots_free} free; {slots_used_blades} blades + {slots_used_pcie} PCIe nodes)
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
columns: Name | Model | OperState | Total Slots | Blades | PCIe Nodes |
Slots Free. The Blades column comes from `slots_used_blades`, PCIe Nodes
from `slots_used_pcie`. Both occupy chassis slots; their sum is `slots_used`.
If `chassis.any_unknown_capacity` is true, add this italic line under the
table: *Total Slots = "0" indicates the chassis model isn't in the
KNOWN_CAPACITY table; extend reports.py to fix.*
Otherwise (no chassis at all), write the single line:
**No chassis present (rack-only environment).**

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
        slots_used_blades=c["slots_used_blades"],
        slots_used_pcie=c["slots_used_pcie"],
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
