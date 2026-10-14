"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  getFullMediaUrl,
  updateMaskAndRecolor,
  selectZone,
  resetZones,
  getCatalogShades,
} from "@/lib/api";
import { RepaintedShadeResult, PaintItem, SelectZonePayload } from "@/lib/types";
import {
  X,
  Paintbrush,
  Eraser,
  RotateCcw,
  Check,
  Loader2,
  Sliders,
  HelpCircle,
  Eye,
  EyeOff,
  Crosshair,
  PlusCircle,
  MinusCircle,
  Trash2,
  Sparkles,
  Building2,
  PaintBucket,
  Shield,
  Palette,
  RefreshCw,
  Wand2,
  Square,
  Pentagon,
  ShieldCheck,
  ShieldAlert,
  LayoutGrid,
} from "lucide-react";

interface MaskEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  originalImageUrl: string;
  maskUrl: string;
  masks?: Record<string, string>;
  coveragePct?: Record<string, number>;
  isExterior?: boolean;
  initialSchemes?: any[];
  protectedHatchedUrl?: string;
  protectedMaskUrl?: string;
  skyFractionByLayer?: Record<string, number>;
  onRecolorComplete: (updatedResults: RepaintedShadeResult[], updatedSchemes?: any) => void;
}

const DEFAULT_POPULAR_SHADES: PaintItem[] = [
  { id: 1, brand: "Asian Paints", name: "Magnolia White", shade_code: "0915", hex: "#FAF6EE", finish: "Matte", color_family: "White" },
  { id: 2, brand: "Asian Paints", name: "Canvas Cream", shade_code: "L102", hex: "#F8F2E4", finish: "Matte", color_family: "Cream" },
  { id: 3, brand: "Asian Paints", name: "Misty Grey", shade_code: "8297", hex: "#E0E2E1", finish: "Matte", color_family: "Grey" },
  { id: 4, brand: "Asian Paints", name: "Terracotta", shade_code: "0444", hex: "#C0583A", finish: "Satin", color_family: "Earthy" },
  { id: 5, brand: "Asian Paints", name: "Rich Navy", shade_code: "7246", hex: "#1E3A5F", finish: "Satin", color_family: "Accent" },
  { id: 6, brand: "Asian Paints", name: "Steel Grey", shade_code: "8298", hex: "#6B7280", finish: "Matte", color_family: "Grey" },
  { id: 7, brand: "Asian Paints", name: "Raw Charcoal", shade_code: "8299", hex: "#2B2D42", finish: "Matte", color_family: "Grey" },
  { id: 8, brand: "Asian Paints", name: "Sea Breeze", shade_code: "7348", hex: "#D0E8EB", finish: "Matte", color_family: "Blue" },
  { id: 9, brand: "Asian Paints", name: "Olive Tint", shade_code: "7692", hex: "#556B2F", finish: "Matte", color_family: "Green" },
  { id: 10, brand: "Asian Paints", name: "Walnut Bark", shade_code: "8632", hex: "#4A2E18", finish: "Satin", color_family: "Brown" },
];

export default function MaskEditorModal({
  isOpen,
  onClose,
  sessionId,
  originalImageUrl,
  maskUrl,
  masks,
  coveragePct,
  isExterior = false,
  initialSchemes,
  protectedHatchedUrl,
  protectedMaskUrl,
  skyFractionByLayer,
  onRecolorComplete,
}: MaskEditorModalProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const bgImageRef = useRef<HTMLImageElement | null>(null);

  const [activeLayer, setActiveLayer] = useState<"main" | "accent" | "trim_gate">("main");
  
  // Selection tools: Magic Wand (Default), Regions (Auto), Rectangle, Polygon, Brush, Eraser, SAM (Optional)
  const [tool, setTool] = useState<"magic_wand" | "auto_region" | "rect" | "polygon" | "brush" | "eraser" | "sam">(
    isExterior ? "magic_wand" : "brush"
  );
  const [operation, setOperation] = useState<"add" | "subtract">("add");
  const [wandTolerance, setWandTolerance] = useState<number>(12); // Default tuned tolerance
  const [autoRegionCount, setAutoRegionCount] = useState<number | null>(null);

  // Polygon state
  const [polygonPoints, setPolygonPoints] = useState<Array<{ x: number; y: number }>>([]);
  // Rectangle state
  const [rectStart, setRectStart] = useState<{ x: number; y: number } | null>(null);
  const [rectCurrent, setRectCurrent] = useState<{ x: number; y: number } | null>(null);

  // Brush state
  const [brushSize, setBrushSize] = useState<number>(25);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [strokePoints, setStrokePoints] = useState<Array<{ x: number; y: number }>>([]);

  // Auto-protect hatched overlay toggle
  const [showHatched, setShowHatched] = useState<boolean>(true);
  const [hatchedImgUrl, setHatchedImgUrl] = useState<string | undefined>(protectedHatchedUrl);
  const [skyFractions, setSkyFractions] = useState<Record<string, number>>(skyFractionByLayer || {});

  // Progress states
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isResetting, setIsResetting] = useState<boolean>(false);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [history, setHistory] = useState<ImageData[]>([]);

  // Stored masks per layer
  const [layerMasks, setLayerMasks] = useState<Record<string, string>>(masks || {});
  const [coverages, setCoverages] = useState<Record<string, number>>(coveragePct || {});

  // Catalog shades & zone color assignment
  const [catalog, setCatalog] = useState<PaintItem[]>(DEFAULT_POPULAR_SHADES);
  const [selectedFamily, setSelectedFamily] = useState<string>("All");
  const [zoneShades, setZoneShades] = useState<Record<string, { shade_code: string; name: string; hex: string }>>({
    main: { shade_code: "0915", name: "Magnolia White", hex: "#FAF6EE" },
    accent: { shade_code: "0444", name: "Terracotta", hex: "#C0583A" },
    trim_gate: { shade_code: "8299", name: "Raw Charcoal", hex: "#2B2D42" },
  });

  // Initialize zone shades from current scheme if available
  useEffect(() => {
    if (initialSchemes && initialSchemes.length > 0 && initialSchemes[0].shades) {
      const s = initialSchemes[0].shades;
      const m = s.find((sh: any) => sh.role === "main") || s[0];
      const a = s.find((sh: any) => sh.role === "accent") || s[1] || m;
      const t = s.find((sh: any) => sh.role === "trim_gate") || s[2] || a;
      setZoneShades({
        main: { shade_code: m.shade_code, name: m.shade_name, hex: m.hex },
        accent: { shade_code: a.shade_code, name: a.shade_name, hex: a.hex },
        trim_gate: { shade_code: t.shade_code, name: t.shade_name, hex: t.hex },
      });
    }
  }, [initialSchemes]);

  // Fetch full catalog shades on open
  useEffect(() => {
    if (isOpen) {
      getCatalogShades()
        .then((items) => {
          if (items && items.length > 0) {
            setCatalog(items);
          }
        })
        .catch(() => {});
    }
  }, [isOpen]);

  // Load background image and current layer mask into canvas
  const initCanvasForLayer = useCallback(
    (layerKey: "main" | "accent" | "trim_gate") => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      setPolygonPoints([]);
      setRectStart(null);
      setRectCurrent(null);
      setStrokePoints([]);

      const bgImg = new Image();
      bgImg.crossOrigin = "anonymous";
      bgImg.src = getFullMediaUrl(originalImageUrl);

      bgImg.onload = () => {
        bgImageRef.current = bgImg;
        canvas.width = bgImg.naturalWidth || 800;
        canvas.height = bgImg.naturalHeight || 600;

        const targetMaskUrl = layerMasks[layerKey] || maskUrl;
        const maskImg = new Image();
        maskImg.crossOrigin = "anonymous";
        maskImg.src = getFullMediaUrl(targetMaskUrl);

        maskImg.onload = () => {
          const offCanvas = document.createElement("canvas");
          offCanvas.width = canvas.width;
          offCanvas.height = canvas.height;
          const offCtx = offCanvas.getContext("2d");
          if (offCtx) {
            offCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
            const maskData = offCtx.getImageData(0, 0, canvas.width, canvas.height);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const targetData = ctx.createImageData(canvas.width, canvas.height);

            for (let i = 0; i < maskData.data.length; i += 4) {
              const val = maskData.data[i] > 120 ? 255 : 0;
              targetData.data[i] = val;
              targetData.data[i + 1] = val;
              targetData.data[i + 2] = val;
              targetData.data[i + 3] = val > 0 ? 255 : 0;
            }

            ctx.putImageData(targetData, 0, 0);
            setHistory([ctx.getImageData(0, 0, canvas.width, canvas.height)]);
          }
        };

        maskImg.onerror = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          setHistory([]);
        };
      };
    },
    [originalImageUrl, maskUrl, layerMasks]
  );

  useEffect(() => {
    if (isOpen) {
      setTool(isExterior ? "magic_wand" : "brush");
      setTimeout(() => initCanvasForLayer(activeLayer), 60);
    }
  }, [isOpen, isExterior, activeLayer, initCanvasForLayer]);

  const saveHistoryState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const currentState = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setHistory((prev) => [...prev.slice(-10), currentState]);
  };

  const handleUndo = () => {
    if (history.length <= 1) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const newHistory = [...history];
    newHistory.pop();
    const prevState = newHistory[newHistory.length - 1];
    ctx.putImageData(prevState, 0, 0);
    setHistory(newHistory);

    setPolygonPoints([]);
    setRectStart(null);
    setRectCurrent(null);
  };

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = 0;
    let clientY = 0;

    if ("touches" in e) {
      if (e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      }
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  // Helper to execute server-side selection
  const executeSelection = async (selectionPayload: Partial<SelectZonePayload>) => {
    setIsProcessing(true);
    try {
      const fullPayload: SelectZonePayload = {
        session_id: sessionId,
        layer: activeLayer,
        tool: selectionPayload.tool || "magic_wand",
        tolerance: wandTolerance,
        operation: operation,
        custom_shades: {
          main: zoneShades.main.shade_code,
          accent: zoneShades.accent.shade_code,
          trim_gate: zoneShades.trim_gate.shade_code,
        },
        ...selectionPayload,
      };

      const resp = await selectZone(fullPayload);
      if (resp.masks) {
        setLayerMasks(resp.masks);
      }
      if (resp.coverage_by_layer) {
        setCoverages(resp.coverage_by_layer);
      }
      if (resp.sky_fraction_by_layer) {
        setSkyFractions(resp.sky_fraction_by_layer);
      }
      if (resp.protected_hatched_url) {
        setHatchedImgUrl(resp.protected_hatched_url);
      }
      if (resp.auto_region_count !== undefined && resp.auto_region_count !== null) {
        setAutoRegionCount(resp.auto_region_count);
      }

      // Load active layer's mask into canvas
      const targetMask = resp.masks ? resp.masks[activeLayer] : resp.mask_url;
      if (targetMask && canvasRef.current) {
        const maskImg = new Image();
        maskImg.crossOrigin = "anonymous";
        maskImg.src = targetMask;
        maskImg.onload = () => {
          const ctx = canvasRef.current?.getContext("2d");
          if (!ctx || !canvasRef.current) return;
          const off = document.createElement("canvas");
          off.width = canvasRef.current.width;
          off.height = canvasRef.current.height;
          const offCtx = off.getContext("2d");
          if (offCtx) {
            offCtx.drawImage(maskImg, 0, 0, canvasRef.current.width, canvasRef.current.height);
            const mData = offCtx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            const tData = ctx.createImageData(canvasRef.current.width, canvasRef.current.height);
            for (let i = 0; i < mData.data.length; i += 4) {
              const val = mData.data[i] > 120 ? 255 : 0;
              tData.data[i] = val;
              tData.data[i + 1] = val;
              tData.data[i + 2] = val;
              tData.data[i + 3] = val > 0 ? 255 : 0;
            }
            ctx.putImageData(tData, 0, 0);
            saveHistoryState();
          }
        };
      }
    } catch (err: any) {
      console.error("Selection failed:", err);
    } finally {
      setIsProcessing(false);
      setPolygonPoints([]);
      setRectStart(null);
      setRectCurrent(null);
      setStrokePoints([]);
    }
  };

  // Canvas Click / Mouse Interactions
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || isProcessing) return;
    const { x, y } = getCanvasCoords(e);
    const normX = x / canvas.width;
    const normY = y / canvas.height;

    if (tool === "magic_wand") {
      executeSelection({
        tool: "magic_wand",
        point: { x: normX, y: normY },
      });
    } else if (tool === "auto_region") {
      executeSelection({
        tool: "auto_region",
        point: { x: normX, y: normY },
      });
    } else if (tool === "sam") {
      executeSelection({
        tool: "sam",
        points: [{ x: normX, y: normY, label: operation === "add" ? 1 : 0 }],
      });
    } else if (tool === "polygon") {
      // Check if clicking near first point to close polygon
      if (polygonPoints.length >= 3) {
        const first = polygonPoints[0];
        const dist = Math.hypot(first.x - normX, first.y - normY);
        if (dist < 0.03) {
          executeSelection({
            tool: "polygon",
            points: polygonPoints,
          });
          return;
        }
      }
      setPolygonPoints((prev) => [...prev, { x: normX, y: normY }]);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isProcessing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(e);
    const normX = x / canvas.width;
    const normY = y / canvas.height;

    if (tool === "rect") {
      setRectStart({ x: normX, y: normY });
      setRectCurrent({ x: normX, y: normY });
    } else if (tool === "brush" || tool === "eraser") {
      setIsDrawing(true);
      setStrokePoints([{ x: normX, y: normY }]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(e);
    const normX = x / canvas.width;
    const normY = y / canvas.height;

    if (tool === "rect" && rectStart) {
      setRectCurrent({ x: normX, y: normY });
    } else if ((tool === "brush" || tool === "eraser") && isDrawing) {
      setStrokePoints((prev) => [...prev, { x: normX, y: normY }]);
    }
  };

  const handleMouseUp = () => {
    if (tool === "rect" && rectStart && rectCurrent) {
      const x1 = Math.min(rectStart.x, rectCurrent.x);
      const x2 = Math.max(rectStart.x, rectCurrent.x);
      const y1 = Math.min(rectStart.y, rectCurrent.y);
      const y2 = Math.max(rectStart.y, rectCurrent.y);

      if (Math.abs(x2 - x1) > 0.01 && Math.abs(y2 - y1) > 0.01) {
        executeSelection({
          tool: "rectangle",
          box: { x1, y1, x2, y2 },
        });
      } else {
        setRectStart(null);
        setRectCurrent(null);
      }
    } else if ((tool === "brush" || tool === "eraser") && isDrawing) {
      setIsDrawing(false);
      if (strokePoints.length > 0) {
        executeSelection({
          tool: "brush",
          points: strokePoints,
          brush_radius: brushSize,
          operation: tool === "eraser" ? "subtract" : operation,
        });
      }
    }
  };

  const handleFinishPolygon = () => {
    if (polygonPoints.length >= 3) {
      executeSelection({
        tool: "polygon",
        points: polygonPoints,
      });
    }
  };

  const handleResetLayer = async () => {
    setIsResetting(true);
    try {
      const resp = await resetZones(sessionId, activeLayer);
      if (resp.masks) setLayerMasks(resp.masks);
      if (resp.coverage_by_layer) setCoverages(resp.coverage_by_layer);
      initCanvasForLayer(activeLayer);
    } catch (err: any) {
      console.error("Reset layer failed:", err);
    } finally {
      setIsResetting(false);
    }
  };

  const handleResetAll = async () => {
    if (!confirm("Reset all 3 zones to default auto-detected facade?")) return;
    setIsResetting(true);
    try {
      const resp = await resetZones(sessionId);
      if (resp.masks) setLayerMasks(resp.masks);
      if (resp.coverage_by_layer) setCoverages(resp.coverage_by_layer);
      initCanvasForLayer(activeLayer);
      if (resp.schemes) {
        onRecolorComplete(resp.results || [], resp.schemes);
      }
    } catch (err: any) {
      alert(`Reset failed: ${err.message || err}`);
    } finally {
      setIsResetting(false);
    }
  };

  const handleSelectShade = (paint: PaintItem) => {
    setZoneShades((prev) => ({
      ...prev,
      [activeLayer]: {
        shade_code: paint.shade_code,
        name: paint.name,
        hex: paint.hex,
      },
    }));
  };

  const handleApplyRecolor = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsSaving(true);
    try {
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const expCtx = exportCanvas.getContext("2d");

      if (expCtx) {
        expCtx.fillStyle = "#000000";
        expCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
        expCtx.drawImage(canvas, 0, 0);

        const maskBase64 = exportCanvas.toDataURL("image/png");
        const customShadesPayload = isExterior
          ? {
              main: zoneShades.main.shade_code,
              accent: zoneShades.accent.shade_code,
              trim_gate: zoneShades.trim_gate.shade_code,
            }
          : undefined;

        const resp = await updateMaskAndRecolor(
          sessionId,
          maskBase64,
          activeLayer,
          customShadesPayload,
          1
        );
        onRecolorComplete(resp.results || [], resp.schemes);
        onClose();
      }
    } catch (err: any) {
      alert(`Failed to update recolored building: ${err.message || err}`);
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  const families = ["All", "White", "Cream", "Grey", "Earthy", "Brown", "Accent", "Blue", "Green"];
  const filteredCatalog =
    selectedFamily === "All"
      ? catalog
      : catalog.filter((p) => (p.color_family || "").toLowerCase().includes(selectedFamily.toLowerCase()));

  const layerMeta = {
    main: { label: "Main Wall", icon: Building2, badgeColor: "bg-blue-600 text-white", tintFilter: "drop-shadow(0px 0px 2px #3b82f6)" },
    accent: { label: "Borders & Slabs", icon: PaintBucket, badgeColor: "bg-amber-600 text-white", tintFilter: "drop-shadow(0px 0px 2px #f59e0b)" },
    trim_gate: { label: "Gate & Trims", icon: Shield, badgeColor: "bg-emerald-600 text-white", tintFilter: "drop-shadow(0px 0px 2px #10b981)" },
  };

  const currentMeta = layerMeta[activeLayer];
  const activeZoneShade = zoneShades[activeLayer];
  const currentSkyLeak = skyFractions[activeLayer] ?? 0.0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="glass-panel-glow border border-white/[0.1] rounded-3xl w-full max-w-5xl max-h-[96vh] flex flex-col shadow-2xl overflow-hidden text-neutral-100 relative">
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-white/[0.08] bg-neutral-900/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center">
              <Wand2 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm sm:text-base text-white flex items-center gap-2">
                <span>{isExterior ? "Refine Facade & Accent Zones" : "Adjust Wall Mask"}</span>
                <span className="text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/20 flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3" /> Protected Sky &amp; Ground
                </span>
              </h3>
              <p className="text-xs text-neutral-400 font-sans mt-0.5">
                Tap wall surfaces to flood-fill or use the brush to fine-tune where paint is applied.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-neutral-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 3 Zone Roles Selector Bar */}
        {isExterior && (
          <div className="px-4 py-2.5 bg-neutral-950/60 border-b border-white/[0.08] flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-neutral-400 font-semibold flex items-center gap-1">
                <Palette className="w-3.5 h-3.5 text-orange-400" /> Active Layer:
              </span>
              <div className="flex items-center gap-1.5 bg-white/[0.04] p-1 rounded-2xl border border-white/[0.08]">
                {(["main", "accent", "trim_gate"] as const).map((layerKey) => {
                  const meta = layerMeta[layerKey];
                  const shade = zoneShades[layerKey];
                  const coverage = coverages[layerKey];
                  const isActive = activeLayer === layerKey;
                  const Icon = meta.icon;

                  return (
                    <button
                      key={layerKey}
                      onClick={() => setActiveLayer(layerKey)}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-xl font-bold transition-all ${
                        isActive
                          ? "btn-primary text-white shadow-sm"
                          : "text-neutral-400 hover:text-white hover:bg-white/5"
                      }`}
                    >
                      <span
                        className="w-3 h-3 rounded-full border border-white/60 shadow-xs shrink-0"
                        style={{ backgroundColor: shade.hex }}
                      />
                      <Icon className="w-3.5 h-3.5 shrink-0" />
                      <span>{meta.label}</span>
                      {coverage !== undefined && (
                        <span className="text-[10px] opacity-80">
                          ({coverage.toFixed(1)}%)
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <button
                onClick={handleResetLayer}
                disabled={isResetting}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-neutral-300 hover:text-white border border-white/[0.08] transition-all"
                title={`Reset ${currentMeta.label} mask`}
              >
                <RefreshCw className={`w-3 h-3 text-orange-400 ${isResetting ? "animate-spin" : ""}`} />
                <span>Reset Layer</span>
              </button>
              <button
                onClick={handleResetAll}
                disabled={isResetting}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-white/[0.04] hover:bg-red-500/20 text-red-300 hover:text-white border border-white/[0.08] hover:border-red-500/40 transition-all"
                title="Reset all zones to initial auto-detected wall"
              >
                <span>Reset All</span>
              </button>
            </div>
          </div>
        )}

        {/* Catalog Palette Strip for Active Zone */}
        {isExterior && (
          <div className="px-4 py-2.5 bg-[#09070e] border-b border-[#ff3b0022] space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="text-[#a0a0b5] font-bold">SHADE FOR {currentMeta.label.toUpperCase()}:</span>
                <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-[#180d15] border border-[#ff440044] text-white font-bold shadow-[0_0_12px_rgba(255,60,0,0.25)]">
                  <span
                    className="w-3 h-3 rounded-full border border-white/40 shadow-xs"
                    style={{ backgroundColor: activeZoneShade.hex }}
                  />
                  <span>{activeZoneShade.name}</span>
                  <span className="text-[#ffaa77] text-[10px]">({activeZoneShade.shade_code})</span>
                  <span className="text-[#888899] text-[10px]">{activeZoneShade.hex}</span>
                </span>
              </div>

              {/* Family Filters */}
              <div className="flex items-center gap-1.5 overflow-x-auto max-w-[50%] scrollbar-none">
                {families.slice(0, 6).map((fam) => (
                  <button
                    key={fam}
                    onClick={() => setSelectedFamily(fam)}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold tracking-wider transition-all ${
                      selectedFamily === fam
                        ? "bg-gradient-to-r from-[#d82500] to-[#ff5500] text-white shadow-[0_0_12px_#ff4400] border border-[#ffa066]/50"
                        : "bg-[#140e1a] text-[#8e8e9f] hover:text-white border border-white/5"
                    }`}
                  >
                    {fam.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Scrollable Swatch Palette */}
            <div className="flex items-center gap-2 overflow-x-auto pb-1.5 scrollbar-thin">
              {filteredCatalog.map((paint) => {
                const isChosen = activeZoneShade.shade_code === paint.shade_code;
                return (
                  <button
                    key={paint.shade_code}
                    onClick={() => handleSelectShade(paint)}
                    className={`shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all ${
                      isChosen
                        ? "bg-[#220d14] border-[#ff4400] text-white shadow-[0_0_16px_rgba(255,68,0,0.4)]"
                        : "bg-[#0f0b16] border-[#ff3b001a] hover:border-[#ff440055] text-[#b0b0c5]"
                    }`}
                  >
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-white/20 shadow-xs shrink-0"
                      style={{ backgroundColor: paint.hex }}
                    />
                    <span className="font-semibold text-[11px] whitespace-nowrap">{paint.name}</span>
                    <span className="text-[10px] text-[#ff8855]">{paint.shade_code}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Selection Tools Bar */}
        <div className="p-3 bg-[#0a0710] border-b border-[#ff3b0022] flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          {/* Main Tools (Magic Wand is Default) */}
          <div className="flex items-center gap-1 bg-[#120a16] p-1 rounded-2xl border border-[#ff3b0022]">
            <button
              onClick={() => setTool("magic_wand")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "magic_wand" ? "btn-magma text-white" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Magic Wand: tap to flood-fill surface in LAB"
            >
              <Wand2 className="w-3.5 h-3.5" />
              <span>WAND</span>
            </button>
            <button
              onClick={() => setTool("auto_region")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "auto_region" ? "bg-gradient-to-r from-[#d97706] to-[#ffaa00] text-black shadow-[0_0_12px_#ffaa00]" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Auto-Regions: tap pre-segmented architectural parts"
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              <span>REGIONS</span>
              {autoRegionCount !== null && autoRegionCount > 0 && (
                <span className="ml-1 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded-full font-mono">
                  {autoRegionCount}
                </span>
              )}
            </button>
            <button
              onClick={() => setTool("rect")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "rect" ? "btn-magma text-white" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Rectangle: click corners or drag box"
            >
              <Square className="w-3.5 h-3.5" />
              <span>RECT</span>
            </button>
            <button
              onClick={() => setTool("polygon")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "polygon" ? "btn-magma text-white" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Polygon: click corners, close shape"
            >
              <Pentagon className="w-3.5 h-3.5" />
              <span>POLYGON</span>
            </button>
            <button
              onClick={() => setTool("brush")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "brush" ? "btn-magma text-white" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Brush: freehand draw"
            >
              <Paintbrush className="w-3.5 h-3.5" />
              <span>BRUSH</span>
            </button>
            <button
              onClick={() => setTool("eraser")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "eraser" ? "btn-magma text-white" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="Eraser: freehand erase"
            >
              <Eraser className="w-3.5 h-3.5" />
              <span>ERASE</span>
            </button>
            <button
              onClick={() => setTool("sam")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                tool === "sam" ? "bg-gradient-to-r from-[#7c3aed] to-[#a855f7] text-white shadow-[0_0_12px_#a855f7]" : "text-[#8e8e9f] hover:text-white"
              }`}
              title="MobileSAM (Optional AI tap segmentation)"
            >
              <Crosshair className="w-3.5 h-3.5" />
              <span>AI SAM</span>
            </button>
          </div>

          {/* Operation (Add / Subtract) */}
          <div className="flex items-center gap-1 bg-[#120a16] p-1 rounded-2xl border border-[#ff3b0022]">
            <button
              onClick={() => setOperation("add")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                operation === "add" ? "bg-[#008844] text-white shadow-[0_0_12px_rgba(0,255,100,0.5)] border border-[#00ff88]/40" : "text-[#8e8e9f] hover:text-white"
              }`}
            >
              <PlusCircle className="w-3.5 h-3.5 text-[#00ff88]" />
              <span>+ ADD</span>
            </button>
            <button
              onClick={() => setOperation("subtract")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold transition-all ${
                operation === "subtract" ? "bg-[#991111] text-white shadow-[0_0_12px_rgba(255,40,40,0.5)] border border-[#ff4444]/40" : "text-[#8e8e9f] hover:text-white"
              }`}
            >
              <MinusCircle className="w-3.5 h-3.5 text-[#ff6666]" />
              <span>− SUBTRACT</span>
            </button>
          </div>

          {/* Tool specific controls (Sliders) */}
          {tool === "magic_wand" && (
            <div className="flex items-center gap-2 bg-[#120a16] px-3.5 py-1.5 rounded-2xl border border-[#ff3b0022]">
              <Sliders className="w-3.5 h-3.5 text-[#ff5500]" />
              <span className="text-[#a0a0b5]">TOLERANCE: {wandTolerance}</span>
              <input
                type="range"
                min="3"
                max="35"
                value={wandTolerance}
                onChange={(e) => setWandTolerance(Number(e.target.value))}
                className="w-20 accent-[#ff4400] cursor-pointer"
              />
            </div>
          )}

          {tool === "brush" && (
            <div className="flex items-center gap-2 bg-[#120a16] px-3.5 py-1.5 rounded-2xl border border-[#ff3b0022]">
              <Sliders className="w-3.5 h-3.5 text-[#ff5500]" />
              <span className="text-[#a0a0b5]">BRUSH: {brushSize}px</span>
              <input
                type="range"
                min="8"
                max="80"
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="w-20 accent-[#ff4400] cursor-pointer"
              />
            </div>
          )}

          {tool === "polygon" && polygonPoints.length >= 3 && (
            <button
              onClick={handleFinishPolygon}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl btn-magma text-white font-bold tracking-wider shadow-md transition-all"
            >
              <Check className="w-3.5 h-3.5 stroke-[3]" />
              <span>COMMIT POLYGON ({polygonPoints.length} PTS)</span>
            </button>
          )}

          {/* Hatched Overlay Toggle & Undo */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHatched((v) => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-bold transition-all ${
                showHatched
                  ? "bg-[#281508] text-[#ffaa77] border-[#ff770066] shadow-[0_0_12px_rgba(255,120,0,0.3)]"
                  : "bg-[#120a16] text-[#8e8e9f] border-white/5 hover:text-white"
              }`}
              title="Toggle diagonal hatched overlay for auto-protected sky, road, and vegetation"
            >
              <ShieldAlert className="w-3.5 h-3.5 text-[#ffaa00]" />
              <span>PROTECT GRID</span>
            </button>

            <button
              onClick={handleUndo}
              disabled={history.length <= 1}
              className="flex items-center gap-1.5 bg-[#120a16] hover:bg-[#1f1024] disabled:opacity-30 px-3 py-1.5 rounded-xl border border-[#ff3b0022] text-[#c0c0d5] hover:text-white font-bold transition-all"
              title="Undo last action"
            >
              <RotateCcw className="w-3.5 h-3.5 text-[#ff5500]" />
              <span>UNDO</span>
            </button>
          </div>
        </div>

        {/* Canvas Workspace */}
        <div
          ref={containerRef}
          className="relative flex-1 min-h-[300px] max-h-[50vh] overflow-hidden bg-black/50 flex items-center justify-center p-3"
        >
          <div className="relative max-w-full max-h-full inline-block rounded-2xl overflow-hidden shadow-2xl border border-white/[0.1]">
            {/* Background Image */}
            <img
              src={getFullMediaUrl(originalImageUrl)}
              alt="Background"
              className="block max-w-full max-h-[48vh] object-contain pointer-events-none select-none"
            />

            {/* Hatched Overlay for Protected Areas */}
            {showHatched && hatchedImgUrl && (
              <img
                src={getFullMediaUrl(hatchedImgUrl)}
                alt="Protected Hatched Overlay"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none opacity-60 z-10"
              />
            )}

            {/* Active Mask Overlay Canvas */}
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              className={`absolute inset-0 w-full h-full cursor-crosshair transition-opacity z-20 ${
                showOverlay ? "opacity-60" : "opacity-0"
              }`}
              style={{
                filter: currentMeta.tintFilter,
              }}
            />

            {/* Interactive Polygon Points & Preview Lines */}
            {tool === "polygon" && polygonPoints.length > 0 && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none z-30">
                <polyline
                  points={polygonPoints.map((p) => `${p.x * 100}%,${p.y * 100}%`).join(" ")}
                  fill="none"
                  stroke="#ff3b00"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                />
                {polygonPoints.map((p, idx) => (
                  <circle
                    key={idx}
                    cx={`${p.x * 100}%`}
                    cy={`${p.y * 100}%`}
                    r="4"
                    fill={idx === 0 ? "#00ff88" : "#ff3b00"}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                ))}
              </svg>
            )}

            {/* Interactive Rectangle Preview Box */}
            {tool === "rect" && rectStart && rectCurrent && (
              <div
                className="absolute border-2 border-dashed border-[#ff4400] bg-[#ff440022] pointer-events-none z-30 shadow-[0_0_15px_rgba(255,60,0,0.5)]"
                style={{
                  left: `${Math.min(rectStart.x, rectCurrent.x) * 100}%`,
                  top: `${Math.min(rectStart.y, rectCurrent.y) * 100}%`,
                  width: `${Math.abs(rectCurrent.x - rectStart.x) * 100}%`,
                  height: `${Math.abs(rectCurrent.y - rectStart.y) * 100}%`,
                }}
              />
            )}
          </div>

          {/* Quick Guide Pill */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-neutral-900/90 backdrop-blur-md px-4 py-1.5 rounded-full text-xs text-neutral-300 border border-white/10 flex items-center gap-2 pointer-events-none shadow-lg z-40">
            {isProcessing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 text-orange-400 animate-spin" />
                <span>Processing selection...</span>
              </>
            ) : tool === "magic_wand" ? (
              <>
                <Wand2 className="w-3.5 h-3.5 text-orange-400" />
                <span>Tap any wall surface to select it automatically.</span>
              </>
            ) : tool === "rect" ? (
              <>
                <Square className="w-3.5 h-3.5 text-amber-400" />
                <span>Drag a box over architectural areas, slabs, or gates.</span>
              </>
            ) : tool === "polygon" ? (
              <>
                <Pentagon className="w-3.5 h-3.5 text-blue-400" />
                <span>Click points around an area, then click the start point to close.</span>
              </>
            ) : (
              <>
                <HelpCircle className="w-3.5 h-3.5 text-emerald-400" />
                <span>Brush or erase freely. Sky and plants are protected.</span>
              </>
            )}
          </div>
        </div>

        {/* Footer Actions with 3 Swatches Summary */}
        <div className="p-4 border-t border-white/[0.08] bg-neutral-900/70 flex flex-wrap items-center justify-between gap-3">
          {isExterior ? (
            <div className="flex items-center gap-3 text-xs">
              <span className="text-neutral-400 font-semibold hidden sm:inline">Active Swatches:</span>
              <div className="flex items-center gap-2">
                {(["main", "accent", "trim_gate"] as const).map((roleKey) => {
                  const m = layerMeta[roleKey];
                  const s = zoneShades[roleKey];
                  return (
                    <div
                      key={roleKey}
                      className="flex items-center gap-2 bg-white/[0.04] px-3 py-1.5 rounded-xl border border-white/[0.06] text-xs"
                    >
                      <span
                        className="w-3 h-3 rounded-full border border-white/30"
                        style={{ backgroundColor: s.hex }}
                      />
                      <span className="font-semibold text-white">{m.label}:</span>
                      <span className="text-neutral-300">{s.name}</span>
                      <span className="text-orange-400 text-[11px]">({s.shade_code})</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div />
          )}

          <div className="flex items-center gap-3 ml-auto">
            <button
              onClick={onClose}
              className="px-4 py-2.5 rounded-xl text-neutral-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-all text-xs font-semibold"
            >
              Cancel
            </button>

            <button
              onClick={handleApplyRecolor}
              disabled={isSaving}
              className="px-6 py-2.5 rounded-xl btn-primary text-white font-bold text-xs sm:text-sm tracking-wide flex items-center gap-2 shadow-md disabled:opacity-50 transition-all"
            >
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Updating previews...</span>
                </>
              ) : (
                <>
                  <Check className="w-4 h-4 stroke-[3]" />
                  <span>Apply &amp; Recolor Walls</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
