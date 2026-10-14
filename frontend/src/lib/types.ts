export interface RoomAnalysis {
  scene_type?: "interior" | "exterior";
  lighting: "bright" | "dim" | "mixed";
  current_wall_color_hex: string;
  style: string;
  mood: string;
  // Interior fields
  dominant_furniture_colors?: string[];
  room_type?: string;
  floor_color_hex?: string;
  // Exterior fields
  accent_colors?: string[];
  building_type?: string;
  light_conditions?: "sunny" | "overcast" | "golden_hour" | "mixed";
  surroundings?: string;
}


export interface RepaintedShadeResult {
  shade_id: number;
  shade_name: string;
  shade_code: string;
  hex: string;
  finish: string;
  color_family: string;
  reason: string;
  image_url: string;
  price?: number | null;
}

export interface SchemeShadeInfo {
  role: "main" | "accent" | "trim_gate";
  shade_id: number;
  shade_name: string;
  shade_code: string;
  hex: string;
  finish: string;
  color_family: string;
}

export interface SchemeResult {
  scheme_index: number;
  reason: string;
  shades: SchemeShadeInfo[];
  image_url: string;
  status?: "success" | "failed" | "fallback";
  mode?: "ai" | "precise";
  mock?: boolean;
  delta_e?: number;
  drift_outside_pct?: number;
  error_message?: string;
}

export interface AnalysisAndRecolorResponse {
  session_id: string;
  original_image_url: string;
  mask_url: string;
  masks?: Record<string, string>;
  coverage_pct?: Record<string, number>;
  coverage_by_layer?: Record<string, number>;
  sky_fraction_by_layer?: Record<string, number>;
  protected_mask_url?: string;
  protected_hatched_url?: string;
  room_analysis: RoomAnalysis;
  results: RepaintedShadeResult[];
  schemes?: SchemeResult[];
  mode?: "ai" | "precise";
  mock?: boolean;
  fallback_reason?: string;
}

export interface SelectZonePayload {
  session_id: string;
  layer: "main" | "accent" | "trim_gate";
  tool: "magic_wand" | "polygon" | "rectangle" | "brush" | "sam" | "mask" | "auto_region";
  tolerance?: number;
  operation?: "add" | "subtract" | "replace";
  point?: { x: number; y: number; label?: number };
  points?: Array<{ x: number; y: number; label?: number }>;
  box?: { x1: number; y1: number; x2: number; y2: number };
  brush_radius?: number;
  custom_mask_base64?: string;
  custom_shades?: Record<string, any>;
  scheme_index?: number;
}

export interface PaintItem {
  id: number;
  brand: string;
  name: string;
  shade_code: string;
  hex: string;
  finish: string;
  color_family: string;
  price?: number | null;
}
