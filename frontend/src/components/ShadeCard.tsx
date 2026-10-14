"use client";

import { RepaintedShadeResult } from "@/lib/types";
import { getFullMediaUrl } from "@/lib/api";
import { Check, MessageCircle, SlidersHorizontal, Sparkles } from "lucide-react";

interface ShadeCardProps {
  result: RepaintedShadeResult;
  isSelected: boolean;
  onSelect: () => void;
  whatsappNumber?: string;
}

export default function ShadeCard({
  result,
  isSelected,
  onSelect,
  whatsappNumber = "+919876543210",
}: ShadeCardProps) {
  const cleanNumber = whatsappNumber.replace(/[^0-9]/g, "");
  const whatsappText = `Hi! I tried the DreamWalls app and would like to see the physical shade card for Asian Paints "${result.shade_name}" (Code: ${result.shade_code}, Finish: ${result.finish}). Is this in stock?`;
  const whatsappUrl = `https://wa.me/${cleanNumber}?text=${encodeURIComponent(whatsappText)}`;

  return (
    <div
      onClick={onSelect}
      className={`group relative flex flex-col rounded-3xl overflow-hidden transition-all duration-300 cursor-pointer ${
        isSelected
          ? "border-2 border-orange-500 bg-orange-500/10 shadow-xl shadow-orange-500/10"
          : "border border-white/[0.08] hover:border-orange-500/40 bg-white/[0.03] hover:bg-white/[0.05]"
      }`}
    >
      {/* Selection indicator pill */}
      {isSelected && (
        <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-orange-500 text-white text-[11px] font-bold px-3 py-1 rounded-full shadow-md">
          <Check className="w-3.5 h-3.5 stroke-[3]" />
          <span>Active Selection</span>
        </div>
      )}

      {/* Repainted Preview Image */}
      <div className="relative w-full aspect-[4/3] bg-black/40 overflow-hidden">
        <img
          src={getFullMediaUrl(result.image_url)}
          alt={result.shade_name}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#090a0f] via-transparent to-transparent opacity-80" />

        {/* Floating Shade Swatch on image */}
        <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-neutral-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 shadow-md">
          <span
            className="w-4 h-4 rounded-full border border-white/40 shadow-xs"
            style={{ backgroundColor: result.hex }}
          />
          <span className="text-xs font-semibold text-white">{result.hex}</span>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 sm:p-5 flex-1 flex flex-col justify-between space-y-4">
        <div>
          {/* Header & Badges */}
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className="text-[11px] font-semibold text-orange-400 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-orange-400" /> Asian Paints
              </span>
              <h4 className="text-base sm:text-lg font-bold text-white group-hover:text-orange-300 transition-colors mt-0.5 tracking-tight">
                {result.shade_name}
              </h4>
            </div>
            <span className="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-bold bg-white/[0.06] text-orange-400 border border-white/[0.08]">
              {result.shade_code}
            </span>
          </div>

          {/* Details pills */}
          <div className="flex flex-wrap items-center gap-1.5 mt-2.5">
            <span className="text-[11px] bg-white/[0.04] text-neutral-300 border border-white/[0.06] px-2 py-0.5 rounded-md">
              Finish: {result.finish}
            </span>
            <span className="text-[11px] bg-white/[0.04] text-neutral-300 border border-white/[0.06] px-2 py-0.5 rounded-md">
              {result.color_family}
            </span>
            {result.price && (
              <span className="text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-md">
                ₹{result.price}/L
              </span>
            )}
          </div>

          {/* AI Design Reason */}
          {result.reason && (
            <div className="mt-3.5 p-3 rounded-2xl bg-white/[0.03] border border-white/[0.05] text-xs text-neutral-300 leading-relaxed flex items-start gap-2">
              <Sparkles className="w-3.5 h-3.5 text-orange-400 shrink-0 mt-0.5" />
              <p className="italic font-sans">"{result.reason}"</p>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="pt-2 border-t border-white/[0.08] flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelect();
            }}
            className={`flex-1 py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
              isSelected
                ? "bg-orange-500/20 text-orange-300 border border-orange-500/40"
                : "bg-white/[0.04] hover:bg-white/[0.08] text-neutral-200 border border-white/[0.08]"
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>{isSelected ? "Active in Viewport" : "Compare"}</span>
          </button>

          <a
            href={whatsappUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="py-2 px-3.5 rounded-xl text-xs font-bold bg-orange-500 hover:bg-orange-600 text-white flex items-center justify-center gap-1.5 transition-all shadow-md"
            title="Ask dealer on WhatsApp"
          >
            <MessageCircle className="w-3.5 h-3.5 text-white" />
            <span>Swatch</span>
          </a>
        </div>
      </div>
    </div>
  );
}
