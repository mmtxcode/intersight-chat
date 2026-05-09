import { IntersightSigner } from "./auth.js";

export interface IntersightConfig {
  baseUrl: string; // e.g. https://intersight.com
  keyId: string;
  pem: string;
}

export interface ApiCallOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE" | "PUT";
  path: string; // e.g. /api/v1/server/Profiles
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
}

export interface ApiCallResult {
  status: number;
  ok: boolean;
  data: unknown;
  error?: string;
}

let activeConfig: IntersightConfig | null = null;
let activeSigner: IntersightSigner | null = null;

export function configureCredentials(cfg: IntersightConfig): void {
  activeConfig = cfg;
  activeSigner = new IntersightSigner(cfg.keyId, cfg.pem);
}

export function isConfigured(): boolean {
  return activeConfig !== null && activeSigner !== null;
}

export function getBaseUrl(): string {
  return activeConfig?.baseUrl ?? "https://intersight.com";
}

function buildQueryString(query?: ApiCallOptions["query"]): string {
  if (!query) return "";
  const params: string[] = [];
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    params.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return params.length ? "?" + params.join("&") : "";
}

export async function callIntersight(
  opts: ApiCallOptions,
): Promise<ApiCallResult> {
  if (!activeConfig || !activeSigner) {
    return {
      status: 0,
      ok: false,
      data: null,
      error:
        "Intersight credentials not configured. Call configure_credentials first.",
    };
  }

  const method = (opts.method ?? "GET").toUpperCase() as ApiCallOptions["method"];
  const path = opts.path.startsWith("/") ? opts.path : "/" + opts.path;
  const qs = buildQueryString(opts.query);
  const fullPath = path + qs;

  const baseUrl = new URL(activeConfig.baseUrl);
  const host = baseUrl.host;
  const url = `${baseUrl.protocol}//${host}${fullPath}`;

  const bodyString =
    method === "GET" || method === "DELETE" || opts.body === undefined
      ? ""
      : typeof opts.body === "string"
        ? opts.body
        : JSON.stringify(opts.body);

  const headers = activeSigner.sign({
    method: method as string,
    host,
    path: fullPath,
    body: bodyString,
    contentType: "application/json",
  });

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: headers as unknown as HeadersInit,
      body: bodyString.length > 0 ? bodyString : undefined,
    });
  } catch (err) {
    return {
      status: 0,
      ok: false,
      data: null,
      error: `Network error calling Intersight: ${(err as Error).message}`,
    };
  }

  const text = await response.text();
  let data: unknown = text;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      // leave as text if not JSON
    }
  } else {
    data = null;
  }

  if (!response.ok) {
    // Intersight returns structured error bodies like
    //   { code: "InvalidUrl", message: "...", messageId: "..." }
    // Surface that into the error string so the model sees the *actual*
    // reason, not just "403 Forbidden". InvalidUrl in particular is a
    // request-validation error (bad $select / path / method), NOT auth.
    let errorMsg = `Intersight returned HTTP ${response.status} ${response.statusText}`;
    if (data && typeof data === "object") {
      const obj = data as Record<string, unknown>;
      const code = typeof obj.code === "string" ? obj.code : undefined;
      const message = typeof obj.message === "string" ? obj.message : undefined;
      if (code || message) {
        errorMsg += ` — ${code ?? ""}${code && message ? ": " : ""}${message ?? ""}`;
      }
      if (code === "InvalidUrl") {
        errorMsg +=
          " (request-validation error: check the path, method, and $select " +
          "field names — this is NOT a permissions issue)";
      }
    }
    return {
      status: response.status,
      ok: false,
      data,
      error: errorMsg,
    };
  }

  return {
    status: response.status,
    ok: true,
    data,
  };
}

/**
 * Pulls only the most useful fields when callers don't request a $select.
 * The full Intersight payload can be hundreds of KB; trimming keeps token use
 * sane for the LLM. Pass selectFields=null to get the full object.
 */
export function trimResults(
  data: unknown,
  defaultSelect?: string[] | null,
): unknown {
  if (!defaultSelect || defaultSelect.length === 0) return data;
  if (!data || typeof data !== "object") return data;
  const obj = data as Record<string, unknown>;
  if (!Array.isArray(obj.Results)) return data;
  const trimmed = (obj.Results as Record<string, unknown>[]).map((r) => {
    const out: Record<string, unknown> = {};
    for (const f of defaultSelect) {
      if (f in r) out[f] = r[f];
    }
    return out;
  });
  return { ...obj, Results: trimmed };
}
