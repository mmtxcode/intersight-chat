import {
  callIntersight,
  configureCredentials,
  getBaseUrl,
  isConfigured,
  trimResults,
} from "./intersight-api.js";

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handler: (args: Record<string, unknown>) => Promise<unknown>;
}

const odataParams = {
  filter: {
    type: "string",
    description:
      "OData $filter expression, e.g. \"Name eq 'my-profile'\" or \"contains(Name,'web')\".",
  },
  select: {
    type: "string",
    description: "OData $select. Comma-separated list of fields to return.",
  },
  top: {
    type: "integer",
    description: "OData $top. Max number of results to return (default 50).",
    minimum: 1,
    maximum: 1000,
  },
  skip: {
    type: "integer",
    description: "OData $skip. Number of results to skip for pagination.",
    minimum: 0,
  },
  orderby: {
    type: "string",
    description: "OData $orderby expression, e.g. \"CreateTime desc\".",
  },
} as const;

function odataQueryFromArgs(
  args: Record<string, unknown>,
): Record<string, string | number | undefined> {
  const top = args.top !== undefined ? Number(args.top) : 50;
  return {
    $filter: args.filter as string | undefined,
    $select: args.select as string | undefined,
    $top: top,
    $skip: args.skip !== undefined ? Number(args.skip) : undefined,
    $orderby: args.orderby as string | undefined,
  };
}

function listTool(opts: {
  name: string;
  description: string;
  endpoint: string;
  defaultSelect?: string[];
}): ToolDefinition {
  return {
    name: opts.name,
    description: opts.description,
    inputSchema: {
      type: "object",
      properties: { ...odataParams },
      additionalProperties: false,
    },
    handler: async (args) => {
      const result = await callIntersight({
        method: "GET",
        path: opts.endpoint,
        query: odataQueryFromArgs(args),
      });
      if (!result.ok) return result;
      // If caller asked for specific fields, respect that — otherwise trim.
      const trimmed =
        args.select === undefined
          ? trimResults(result.data, opts.defaultSelect)
          : result.data;
      return { ...result, data: trimmed };
    },
  };
}

export const tools: ToolDefinition[] = [
  {
    name: "configure_credentials",
    description:
      "Configure the Intersight v3 API Key ID and PEM private key for the current MCP session. Must be called before any Intersight tool. Credentials are held in memory only.",
    inputSchema: {
      type: "object",
      properties: {
        key_id: {
          type: "string",
          description: "Intersight API Key ID (looks like XXXX/YYYY/ZZZZ).",
        },
        pem: {
          type: "string",
          description: "Full PEM-encoded private key contents (including BEGIN/END lines).",
        },
        base_url: {
          type: "string",
          description:
            "Optional base URL. Defaults to https://intersight.com. Override for appliance deployments.",
        },
      },
      required: ["key_id", "pem"],
      additionalProperties: false,
    },
    handler: async (args) => {
      try {
        configureCredentials({
          keyId: String(args.key_id ?? ""),
          pem: String(args.pem ?? ""),
          baseUrl: String(args.base_url ?? "https://intersight.com"),
        });
        return {
          ok: true,
          message: "Credentials configured.",
          base_url: getBaseUrl(),
        };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    },
  },

  {
    name: "test_connection",
    description:
      "Verify the configured Intersight credentials by making a small probe call to /api/v1/iam/Accounts.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    handler: async () => {
      if (!isConfigured()) {
        return {
          ok: false,
          error:
            "Credentials not configured. Call configure_credentials first.",
        };
      }
      const res = await callIntersight({
        method: "GET",
        path: "/api/v1/iam/Accounts",
        query: { $top: 1 },
      });
      return res;
    },
  },

  listTool({
    name: "get_server_profiles",
    description:
      "List Intersight server profiles. Supports OData $filter/$select/$top/$skip/$orderby.",
    endpoint: "/api/v1/server/Profiles",
    defaultSelect: [
      "Name",
      "Moid",
      "Description",
      "ConfigContext",
      "AssignedServer",
      "TargetPlatform",
      "Type",
      "ModTime",
    ],
  }),

  {
    name: "get_server_profile_by_name",
    description: "Get a specific server profile by exact Name.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Exact server profile Name." },
        select: odataParams.select,
      },
      required: ["name"],
      additionalProperties: false,
    },
    handler: async (args) => {
      const name = String(args.name ?? "");
      return await callIntersight({
        method: "GET",
        path: "/api/v1/server/Profiles",
        query: {
          $filter: `Name eq '${name.replace(/'/g, "''")}'`,
          $select: args.select as string | undefined,
          $top: 1,
        },
      });
    },
  },

  listTool({
    name: "get_physical_servers",
    description:
      "List physical compute summaries (rack & blade servers managed by Intersight).",
    endpoint: "/api/v1/compute/PhysicalSummaries",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "ManagementMode",
      "OperPowerState",
      "AdminPowerState",
      "Firmware",
      "NumCpus",
      "NumThreads",
      "TotalMemory",
      "Vendor",
    ],
  }),

  listTool({
    name: "get_chassis",
    description:
      "List chassis inventory. Each chassis has a Moid you can join with " +
      "compute Blades' EquipmentChassis.Moid (or Chassis.Moid for older " +
      "payloads — check both) to compute slot occupancy. NumSlots is often " +
      "empty for X-Series chassis; use known capacities by model when " +
      "missing (UCSX-9508 = 8, UCSB-5108-AC2 = 8).",
    // Intersight uses the irregular plural `Chasses` for this endpoint (verified
    // against CiscoDevNet/intersight-python's equipment_api.py). The MO type is
    // still `equipment.Chassis` and the field name on Blades is still `Chassis.Moid`.
    endpoint: "/api/v1/equipment/Chasses",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "Vendor",
      "OperState",
      "ManagementMode",
      "NumSlots",
    ],
  }),

  listTool({
    name: "get_compute_blades",
    description:
      "List blade servers. Each blade has SlotId (its slot in the parent " +
      "chassis) and a chassis reference. Modern Intersight uses " +
      "EquipmentChassis (canonical); older payloads use Chassis — check both. " +
      "To compute available/free slots in a chassis: capacity minus the count " +
      "of blades AND PCIe nodes (see get_pci_nodes) attached to that chassis.",
    endpoint: "/api/v1/compute/Blades",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "SlotId",
      // Both possible chassis-ref field names. Newer Intersight uses
      // EquipmentChassis; some older payloads still use Chassis. Including
      // both in defaultSelect means trimResults preserves whichever is
      // populated — without it, the blade -> chassis join silently breaks.
      "EquipmentChassis",
      "Chassis",
      "OperState",
      "OperPowerState",
      "AdminPowerState",
      "TotalMemory",
      "NumCpus",
      "NumThreads",
    ],
  }),

  listTool({
    name: "get_compute_rack_units",
    description: "List rack-mount servers (standalone or UCS-managed rack units).",
    endpoint: "/api/v1/compute/RackUnits",
    defaultSelect: [
      "Name",
      "Moid",
      "Serial",
      "Model",
      "OperState",
      "OperPowerState",
      "AdminPowerState",
      "TotalMemory",
      "NumCpus",
      "NumThreads",
    ],
  }),

  listTool({
    name: "get_pci_nodes",
    description:
      "List PCIe nodes (pci.Node MOs, e.g., UCSX-440P GPU nodes in X-Series " +
      "chassis). PCIe nodes occupy chassis slots just like blades but do NOT " +
      "reference a chassis directly — they reference their paired blade via " +
      "ComputeBlade (with Parent as a fallback). To compute true slot " +
      "occupancy, join pci.Node -> compute.Blade -> equipment.Chassis. The " +
      "total used slots in a chassis is the count of blades PLUS the count " +
      "of PCIe nodes whose paired blade lives in that chassis.",
    endpoint: "/api/v1/pci/Nodes",
    defaultSelect: [
      "Name",
      "Moid",
      "Model",
      "SlotId",
      "ComputeBlade",
      "Parent",
      "ObjectType",
    ],
  }),

  listTool({
    name: "get_fabric_interconnects",
    description:
      "List fabric interconnects (network elements). The admin-configured " +
      "Name is often empty — when it is, identify the FI by Switchid " +
      "(always 'A' or 'B', commonly referred to as 'FI-A' / 'FI-B'), " +
      "Hostname, or Dn (e.g., 'sys/switch-A'), in that order of preference.",
    endpoint: "/api/v1/network/Elements",
    defaultSelect: [
      "Name",
      "Hostname",
      "Dn",
      "Moid",
      "Serial",
      "Model",
      "Vendor",
      "OperState",
      "ManagementMode",
      "OutOfBandIpAddress",
      "Switchid",
      "Version",
    ],
  }),

  listTool({
    name: "get_alarms",
    description:
      "List active alarms. Defaults to most recent first; use $filter to narrow by Severity, e.g. \"Severity eq 'Critical'\".",
    endpoint: "/api/v1/cond/Alarms",
    defaultSelect: [
      "Name",
      "Moid",
      "Severity",
      "Description",
      "Code",
      "AffectedMoDisplayName",
      "AffectedObjectType",
      "CreationTime",
      "LastTransitionTime",
      "Acknowledge",
    ],
  }),

  listTool({
    name: "get_hcl_status",
    description: "List HCL (Hardware Compatibility List) compatibility statuses.",
    endpoint: "/api/v1/cond/HclStatuses",
    defaultSelect: [
      "Moid",
      "Status",
      "Reason",
      "ServerReason",
      "InvalidReasons",
      "HardwareStatus",
      "SoftwareStatus",
      "ManagedObject",
    ],
  }),

  listTool({
    name: "get_running_firmware",
    description: "List running firmware versions across managed components.",
    endpoint: "/api/v1/firmware/RunningFirmwares",
    defaultSelect: [
      "Moid",
      "Component",
      "Version",
      "PackageVersion",
      "Type",
      "Vendor",
      "Model",
    ],
  }),

  {
    name: "get_alarm_summary",
    description:
      "Get a fleet-wide alarm count rolled up by severity (Critical, Warning, Info, Cleared). " +
      "Use this for questions like 'how many critical alarms?' or 'is anything broken?' — " +
      "much cheaper than enumerating individual alarms with get_alarms.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    handler: async () => {
      // Intersight doesn't expose a dedicated /cond/AlarmSummary endpoint;
      // we build the summary using OData $apply=groupby on /cond/Alarms.
      return await callIntersight({
        method: "GET",
        path: "/api/v1/cond/Alarms",
        query: {
          $apply: "groupby((Severity),aggregate($count as Count))",
        },
      });
    },
  },

  listTool({
    name: "get_organizations",
    description:
      "List Intersight organizations. Use this when the user asks about a specific " +
      "tenant/org, or to scope a follow-up query (most resources have an Organization " +
      "reference you can $filter on).",
    endpoint: "/api/v1/organization/Organizations",
    defaultSelect: ["Name", "Moid", "Description", "AccountMoid"],
  }),

  listTool({
    name: "get_advisories",
    description:
      "List active advisory instances (PSIRTs, field notices) affecting your fleet. " +
      "Each instance has a State (active/cleared), an AffectedObject reference, and a " +
      "Definition reference. To see advisory titles/descriptions, follow up with " +
      "generic_api_call to /api/v1/tam/AdvisoryDefinitions for the relevant Moids, or " +
      "pass select='*' here for the full record.",
    endpoint: "/api/v1/tam/AdvisoryInstances",
    defaultSelect: ["Moid", "State", "AffectedObject", "Definition", "LastVisibleTime"],
  }),

  listTool({
    name: "get_contracts",
    description:
      "List device contract / service-coverage information. Use this for questions " +
      "like 'what contracts are expiring?' or 'show me servers without coverage'. " +
      "Pass orderby='ServiceEndDate asc' to see soonest-to-expire first; filter on " +
      "ContractStatus eq 'Expired' or 'Active' to narrow.",
    endpoint: "/api/v1/asset/DeviceContractInformations",
    defaultSelect: [
      "Moid",
      "ContractStatus",
      "ContractStatusReason",
      "ServiceEndDate",
      "ServiceLevel",
      "ServiceSku",
      "ProductId",
      "DeviceId",
      "DeviceType",
      "Source",
    ],
  }),

  {
    name: "generic_api_call",
    description:
      "Make an arbitrary Intersight REST API call. Use this for endpoints not covered by a dedicated tool. Supports any HTTP method, any path under /api/v1/, query parameters, and optional JSON body. Prefer dedicated tools when one fits.",
    inputSchema: {
      type: "object",
      properties: {
        method: {
          type: "string",
          enum: ["GET", "POST", "PATCH", "DELETE", "PUT"],
          description: "HTTP method.",
        },
        path: {
          type: "string",
          description:
            "Path under the Intersight base URL, e.g. /api/v1/asset/DeviceRegistrations.",
        },
        query: {
          type: "object",
          description:
            "Query parameters. For OData, use keys like $filter, $select, $top, $skip, $orderby.",
          additionalProperties: { type: ["string", "number", "boolean"] },
        },
        body: {
          description: "Optional JSON body for POST/PATCH/PUT.",
        },
      },
      required: ["method", "path"],
      additionalProperties: false,
    },
    handler: async (args) => {
      return await callIntersight({
        method: args.method as "GET" | "POST" | "PATCH" | "DELETE" | "PUT",
        path: String(args.path),
        query: args.query as Record<string, string | number | boolean> | undefined,
        body: args.body,
      });
    },
  },
];

// ---------------------------------------------------------------- list_tools
//
// Pushed onto `tools` after the array is declared so the handler can close
// over `tools` (referencing it inside an array literal at module top-level
// would hit a TDZ). Exposed to the model so it can answer "what tools do
// you have?" by calling a function rather than enumerating from memory —
// smaller models tend to truncate self-described tool lists, and the
// missing entries can mislead users about what the app can do.

const HIDDEN_FROM_LISTING = new Set([
  // configure_credentials is host-only (the Streamlit sidebar wires it).
  // It's already hidden from the model in orchestrator.py's HIDDEN_TOOLS;
  // exclude it here too so list_tools output matches the model's actual
  // tool surface.
  "configure_credentials",
]);

tools.push({
  name: "list_tools",
  description:
    "Return the complete list of MCP tools available on this server, with " +
    "each tool's name and description. ALWAYS call this tool when the user " +
    "asks what tools, commands, capabilities, MCPs, or functions you have — " +
    "the result is the authoritative list. Do not summarize from memory; " +
    "always call this tool for these questions and render every entry.",
  inputSchema: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
  handler: async () => {
    return {
      ok: true,
      count: tools.filter((t) => !HIDDEN_FROM_LISTING.has(t.name)).length,
      tools: tools
        .filter((t) => !HIDDEN_FROM_LISTING.has(t.name))
        .map((t) => ({ name: t.name, description: t.description })),
    };
  },
});

export const toolMap = new Map(tools.map((t) => [t.name, t]));
