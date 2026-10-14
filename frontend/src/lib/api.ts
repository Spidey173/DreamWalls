import { AnalysisAndRecolorResponse, PaintItem, RepaintedShadeResult, SchemeResult, SelectZonePayload } from "./types";

/**
 * Resolves the backend API base URL:
 * - In all environments: uses process.env.NEXT_PUBLIC_API_URL if configured.
 * - In production: strictly requires NEXT_PUBLIC_API_URL to be set, NEVER falling back to localhost.
 * - In development only: falls back to http://localhost:8000 if NEXT_PUBLIC_API_URL is unset.
 */
export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

  if (envUrl && envUrl !== "") {
    return envUrl.replace(/\/+$/, "");
  }

  // Default to local backend server
  return "http://localhost:8000";
}

function getSafeBaseUrl(): string {
  try {
    return getApiBaseUrl();
  } catch {
    return "unconfigured URL";
  }
}

/**
 * Wrapper around window.fetch providing diagnostic error messages:
 * In development, reports "Server unreachable at <URL>" if the network request fails.
 */
async function fetchWithDiagnostics(url: string, options?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (err: any) {
    throw new Error(
      `Failed to connect to backend at ${url}. Ensure the local backend server is running on http://127.0.0.1:8000.`
    );
  }
}

export function getFullMediaUrl(pathOrUrl: string): string {
  if (!pathOrUrl) return "";
  if (pathOrUrl.startsWith("data:") || pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  const cleanPath = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${getApiBaseUrl()}${cleanPath}`;
}

export async function analyzeRoomImage(file: File): Promise<AnalysisAndRecolorResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/analyze`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = `Server error (${response.status})`;
    try {
      const errJson = await response.json();
      errorMsg = errJson.detail || errorMsg;
    } catch {
      // fallback
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function updateMaskAndRecolor(
  sessionId: string,
  customMaskBase64?: string,
  layer: string = "main",
  customShades?: Record<string, any>,
  schemeIndex: number = 1
): Promise<{
  session_id: string;
  mask_url: string;
  masks?: Record<string, string>;
  coverage_pct?: Record<string, number>;
  results: RepaintedShadeResult[];
  schemes?: SchemeResult[];
}> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/recolor-mask`;

  const payload: any = {
    session_id: sessionId,
    layer,
  };
  if (customMaskBase64) {
    payload.custom_mask_base64 = customMaskBase64;
  }
  if (customShades) {
    payload.custom_shades = customShades;
    payload.scheme_index = schemeIndex;
  }

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMsg = `Error updating mask (${response.status})`;
    try {
      const err = await response.json();
      errorMsg = err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function resetZones(
  sessionId: string,
  layer?: string
): Promise<{
  session_id: string;
  mask_url: string;
  masks?: Record<string, string>;
  coverage_pct: number;
  coverage_by_layer?: Record<string, number>;
  schemes?: SchemeResult[];
  results?: RepaintedShadeResult[];
}> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/reset-zones`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      layer,
    }),
  });

  if (!response.ok) {
    let errorMsg = `Error resetting zones (${response.status})`;
    try {
      const err = await response.json();
      errorMsg = err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function getCatalogShades(colorFamily?: string, search?: string): Promise<PaintItem[]> {
  const baseUrl = getApiBaseUrl();
  const params = new URLSearchParams();
  if (colorFamily) params.append("color_family", colorFamily);
  if (search) params.append("search", search);
  const endpoint = `${baseUrl}/api/v1/catalog-shades?${params.toString()}`;

  const response = await fetchWithDiagnostics(endpoint);
  if (!response.ok) throw new Error("Failed to fetch catalog shades");
  return response.json();
}

export async function selectZone(payload: SelectZonePayload): Promise<{
  session_id: string;
  active_layer: string;
  mask_url: string;
  masks: Record<string, string>;
  coverage_by_layer: Record<string, number>;
  coverage_pct: number;
  sky_fraction_by_layer?: Record<string, number>;
  protected_mask_url?: string;
  protected_hatched_url?: string;
  auto_region_count?: number;
  selected_region_id?: number | null;
  auto_region_compute_time_ms?: number;
  schemes?: SchemeResult[];
  results?: RepaintedShadeResult[];
}> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/select-zone`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMsg = `Error in zone selection (${response.status})`;
    try {
      const err = await response.json();
      errorMsg = err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function regenerateScheme(
  sessionId: string,
  schemeIndex: number
): Promise<SchemeResult> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/schemes/regenerate`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      scheme_index: schemeIndex,
    }),
  });

  if (!response.ok) {
    let errorMsg = `Error regenerating scheme (${response.status})`;
    try {
      const err = await response.json();
      errorMsg = err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export interface TapPointPayload {
  x: number;
  y: number;
  label: number; // 1 = positive, 0 = negative
}

export async function tapSegment(
  sessionId: string,
  points: TapPointPayload[],
  layer: string = "main"
): Promise<{
  session_id: string;
  active_layer?: string;
  mask_url: string;
  masks?: Record<string, string>;
  coverage_pct: number | Record<string, number>;
}> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/tap-segment`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      layer,
      points,
    }),
  });

  if (!response.ok) {
    let errorMsg = `Error in tap segment (${response.status})`;
    try {
      const err = await response.json();
      errorMsg = err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function verifyAdmin(password: string): Promise<boolean> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/verify`;

  try {
    const response = await fetchWithDiagnostics(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getAdminPaints(password: string, search?: string, colorFamily?: string): Promise<PaintItem[]> {
  const params = new URLSearchParams();
  if (search) params.append("search", search);
  if (colorFamily) params.append("color_family", colorFamily);

  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/paints?${params.toString()}`;

  const response = await fetchWithDiagnostics(endpoint, {
    headers: {
      "X-Admin-Password": password,
    },
  });

  if (!response.ok) throw new Error("Unauthorized or failed to fetch paints");
  return response.json();
}

export async function createAdminPaint(password: string, paint: Omit<PaintItem, "id">): Promise<PaintItem> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/paints`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Password": password,
    },
    body: JSON.stringify(paint),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Failed to create paint");
  }
  return response.json();
}

export async function updateAdminPaint(password: string, id: number, paint: Partial<PaintItem>): Promise<PaintItem> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/paints/${id}`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Password": password,
    },
    body: JSON.stringify(paint),
  });

  if (!response.ok) throw new Error("Failed to update paint");
  return response.json();
}

export async function deleteAdminPaint(password: string, id: number): Promise<void> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/paints/${id}`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "DELETE",
    headers: {
      "X-Admin-Password": password,
    },
  });

  if (!response.ok) throw new Error("Failed to delete paint");
}

export async function uploadCatalogCSV(password: string, file: File): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);

  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/admin/paints/import-csv`;

  const response = await fetchWithDiagnostics(endpoint, {
    method: "POST",
    headers: {
      "X-Admin-Password": password,
    },
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Failed to import CSV");
  }
  return response.json();
}
