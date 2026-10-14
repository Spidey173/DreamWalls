"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface BeforeAfterSliderProps {
  beforeImage: string;
  afterImage: string;
  beforeLabel?: string;
  afterLabel?: string;
  afterHex?: string;
  className?: string;
}

export default function BeforeAfterSlider({
  beforeImage,
  afterImage,
  beforeLabel = "Original",
  afterLabel = "Repainted Wall",
  afterHex = "#FAF6EE",
  className = "max-h-[72vh]",
}: BeforeAfterSliderProps) {
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage 0-100
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedX = Math.max(0, Math.min(x, rect.width));
    const percentage = (clampedX / rect.width) * 100;
    setSliderPosition(percentage);
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!isDragging || e.touches.length === 0) return;
    handleMove(e.touches[0].clientX);
  }, [isDragging, handleMove]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  }, [isDragging, handleMove]);

  const handleEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleEnd);
      window.addEventListener("touchmove", handleTouchMove, { passive: true });
      window.addEventListener("touchend", handleEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleEnd);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleEnd);
    };
  }, [isDragging, handleMouseMove, handleTouchMove, handleEnd]);

  const [aspectRatio, setAspectRatio] = useState<number | null>(null);

  return (
    <div
      ref={containerRef}
      className={`relative w-full rounded-2xl sm:rounded-3xl overflow-hidden select-none glass-panel cursor-ew-resize touch-none mx-auto flex items-center justify-center group shadow-2xl ${className}`}
      style={{
        aspectRatio: aspectRatio ? `${aspectRatio}` : "4 / 3",
      }}
      onMouseDown={() => setIsDragging(true)}
      onTouchStart={() => setIsDragging(true)}
    >
      {/* Background (After / Repainted Image) */}
      <img
        src={afterImage}
        alt={afterLabel}
        onLoad={(e) => {
          if (e.currentTarget.naturalWidth && e.currentTarget.naturalHeight) {
            setAspectRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight);
          }
        }}
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        draggable={false}
      />

      {/* Foreground (Before / Original Image) with clip-path */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{
          clipPath: `polygon(0% 0%, ${sliderPosition}% 0%, ${sliderPosition}% 100%, 0% 100%)`,
        }}
      >
        <img
          src={beforeImage}
          alt={beforeLabel}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          draggable={false}
        />
      </div>

      {/* Elegant Divider Line */}
      <div
        className="absolute top-0 bottom-0 w-[2px] bg-white shadow-[0_0_12px_rgba(255,255,255,0.7)] pointer-events-none z-20"
        style={{ left: `${sliderPosition}%` }}
      >
        {/* Minimalist Circular Grip Handle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-white text-neutral-900 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
          <div className="flex items-center -space-x-1">
            <ChevronLeft className="w-3.5 h-3.5 stroke-[3] text-neutral-800" />
            <ChevronRight className="w-3.5 h-3.5 stroke-[3] text-neutral-800" />
          </div>
        </div>
      </div>

      {/* Before Label Badge */}
      <div className="absolute top-4 left-4 z-10 pointer-events-none flex items-center gap-2 bg-neutral-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 shadow-md">
        <span className="w-2 h-2 rounded-full bg-neutral-400" />
        <span className="text-xs font-semibold text-neutral-200">
          {beforeLabel}
        </span>
      </div>

      {/* After Label Badge */}
      <div className="absolute top-4 right-4 z-10 pointer-events-none flex items-center gap-2 bg-neutral-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 shadow-md">
        {afterHex && (
          <span
            className="w-3.5 h-3.5 rounded-full border border-white/40 shadow-xs"
            style={{ backgroundColor: afterHex }}
          />
        )}
        <span className="text-xs font-semibold text-white">
          {afterLabel}
        </span>
      </div>

      {/* Subtle Hint Bar at Bottom */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 pointer-events-none bg-neutral-900/70 backdrop-blur-md px-3.5 py-1 rounded-full border border-white/10 text-[11px] text-neutral-300 font-medium">
        Drag slider to compare
      </div>
    </div>
  );
}
