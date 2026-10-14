"use client";

import { SchemeResult } from "@/lib/types";
import { getFullMediaUrl } from "@/lib/api";
import { Check, MessageCircle, Sparkles, Building2, PaintBucket, Shield, RotateCw, AlertCircle } from "lucide-react";

interface SchemeCardProps {
  scheme: SchemeResult;
  isSelected: boolean;
  onSelect: () => void;
  whatsappNumber?: string;
  onRegenerate?: (schemeIndex: number) => void;
  isRegenerating?: boolean;
}

export default function SchemeCard({
  scheme,
  isSelected,
  onSelect,
  whatsappNumber = "+919876543210",
  onRegenerate,
  isRegenerating = false,
}: SchemeCardProps) {
  const cleanNumber = whatsappNumber.replace(/[^0-9]/g, "");

  const mainShade = scheme.shades.find((s) => s.role === "main") || scheme.shades[0];
  const accentShade = scheme.shades.find((s) => s.role === "accent") || scheme.shades[1] || mainShade;
  const trimShade = scheme.shades.find((s) => s.role === "trim_gate") || scheme.shades[2] || accentShade;

  const whatsappText = `Hi! I tried DreamWalls for my building exterior and loved Scheme #${scheme.scheme_index}:
- Main Facade: ${mainShade.shade_name} (${mainShade.shade_code}, ${mainShade.hex})
- Accents & Slabs: ${accentShade.shade_name} (${accentShade.shade_code}, ${accentShade.hex})
- Gate & Trim: ${trimShade.shade_name} (${trimShade.shade_code}, ${trimShade.hex})
Could you share the physical shade cards and pricing for these?`;

  const whatsappUrl = `https://wa.me/${cleanNumber}?text=${encodeURIComponent(whatsappText)}`;

  const roleMeta = {
    main: { label: "Main Facade", icon: Building2 },
    accent: { label: "Accents & Slabs", icon: PaintBucket },
    trim_gate: { label: "Gate & Trims", icon: Shield },
  };

  const isFailed = scheme.status === "failed";

  return (
    <div
      onClick={onSelect}
      className={`group relative flex flex-col rounded-3xl overflow-hidden transition-all duration-300 cursor-pointer ${
        isSelected
          ? "border-2 border-orange-500 bg-orange-500/10 shadow-xl shadow-orange-500/10"
          : "border border-white/[0.08] hover:border-orange-500/40 bg-white/[0.03] hover:bg-white/[0.05]"
      }`}
    >
      {/* Active Indicator Badge */}
      {isSelected && (
        <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-orange-500 text-white text-[11px] font-bold px-3 py-1 rounded-full shadow-md">
          <Check className="w-3.5 h-3.5 stroke-[3]" />
          <span>Active Selection</span>
        </div>
      )}

      {/* Demo Mode Badge if any */}
      {scheme.mock && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-1 bg-amber-600/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-xs">
          <AlertCircle className="w-3 h-3" />
          <span>Preview Mode</span>
        </div>
      )}

      {/* Repainted Preview Image */}
      <div className="relative w-full aspect-[4/3] bg-black/40 overflow-hidden">
        <img
          src={getFullMediaUrl(scheme.image_url)}
          alt={`Exterior Scheme ${scheme.scheme_index}`}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#090a0f] via-transparent to-transparent opacity-80" />

        {/* 3 Swatch Dots Floating on Image */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1.5 bg-neutral-900/80 backdrop-blur-md px-2.5 py-1.5 rounded-xl border border-white/10 shadow-md">
          <span
            className="w-3.5 h-3.5 rounded-full border border-white/40 shadow-xs"
            style={{ backgroundColor: mainShade.hex }}
            title={`Main: ${mainShade.shade_name} (${mainShade.hex})`}
          />
          <span
            className="w-3.5 h-3.5 rounded-full border border-white/40 shadow-xs"
            style={{ backgroundColor: accentShade.hex }}
            title={`Accent: ${accentShade.shade_name} (${accentShade.hex})`}
          />
          <span
            className="w-3.5 h-3.5 rounded-full border border-white/40 shadow-xs"
            style={{ backgroundColor: trimShade.hex }}
            title={`Trim: ${trimShade.shade_name} (${trimShade.hex})`}
          />
        </div>
      </div>

      {/* Card Content */}
      <div className="p-4 sm:p-5 flex-1 flex flex-col justify-between space-y-4">
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className="text-[11px] font-semibold text-orange-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-orange-400" /> Scheme #{scheme.scheme_index}
              </span>
              <h4 className="text-base font-bold text-white group-hover:text-orange-300 transition-colors mt-0.5 tracking-tight">
                {mainShade.shade_name} &amp; Coordinated Trims
              </h4>
            </div>
            <span className="text-[10px] font-semibold bg-white/[0.06] text-neutral-300 px-2.5 py-1 rounded-md border border-white/[0.08]">
              3 Coordinated Shades
            </span>
          </div>

          {/* Designer Rationale */}
          {scheme.reason && (
            <p className="text-xs text-neutral-300 leading-relaxed bg-white/[0.03] p-2.5 rounded-xl border border-white/[0.06] italic">
              "{scheme.reason}"
            </p>
          )}

          {/* 3 Labeled Role Swatches */}
          <div className="space-y-2 pt-1">
            {[
              { shade: mainShade, role: "main" as const },
              { shade: accentShade, role: "accent" as const },
              { shade: trimShade, role: "trim_gate" as const },
            ].map(({ shade, role }) => {
              const meta = roleMeta[role];
              return (
                <div
                  key={role}
                  className="flex items-center justify-between p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.05] text-xs"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className="w-4 h-4 rounded-full border border-white/40 shrink-0 shadow-xs"
                      style={{ backgroundColor: shade.hex }}
                    />
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-semibold text-white text-xs">{shade.shade_name}</span>
                        <span className="text-[11px] text-orange-400 font-medium">({shade.shade_code})</span>
                      </div>
                      <div className="flex items-center gap-1 text-[11px] text-neutral-400">
                        <span>{meta.label}</span>
                        <span>· {shade.hex}</span>
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-white/[0.06] text-neutral-300 border border-white/[0.08]">
                    {shade.finish}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-2 border-t border-white/[0.08] space-y-2">
          {onRegenerate && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onRegenerate(scheme.scheme_index);
              }}
              disabled={isRegenerating}
              className={`w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl border text-xs font-semibold transition-all ${
                isFailed
                  ? "bg-red-600 hover:bg-red-500 text-white border-red-500"
                  : "bg-white/[0.04] hover:bg-white/[0.08] text-neutral-200 border-white/[0.08]"
              } ${isRegenerating ? "opacity-60 cursor-not-allowed" : ""}`}
            >
              <RotateCw className={`w-3.5 h-3.5 ${isRegenerating ? "animate-spin text-orange-400" : ""}`} />
              <span>{isRegenerating ? "Generating preview..." : "Regenerate Scheme"}</span>
            </button>
          )}

          <a
            href={whatsappUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-orange-500 hover:bg-orange-600 text-white text-xs font-bold transition-all shadow-md"
          >
            <MessageCircle className="w-3.5 h-3.5" />
            <span>Order Swatches via WhatsApp</span>
          </a>
        </div>
      </div>
    </div>
  );
}
