"use client";

import Link from "next/link";
import { Shield, MessageCircle, Sparkles, Palette } from "lucide-react";

export default function Navbar() {
  const whatsappNumber = process.env.NEXT_PUBLIC_SHOP_WHATSAPP_NUMBER || "+919876543210";
  const cleanNumber = whatsappNumber.replace(/[^0-9]/g, "");

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/[0.08] bg-[#090a0f]/85 backdrop-blur-xl">
      <div className="w-full px-4 sm:px-8 xl:px-12 h-16 flex items-center justify-between">
        {/* Brand: DreamWalls */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-[0_2px_12px_rgba(249,115,22,0.35)] group-hover:scale-105 transition-transform">
            <Palette className="w-5 h-5 text-white" />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-extrabold text-xl tracking-tight text-white">
                DREAM<span className="text-orange-500">WALLS</span>
              </span>
              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-orange-300 bg-orange-500/10 px-2 py-0.5 rounded-full border border-orange-500/20">
                <Sparkles className="w-3 h-3 text-orange-400" /> AI Visualizer
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 font-sans hidden sm:block">
              Asian Paints Color &amp; Recolor Studio
            </p>
          </div>
        </Link>

        {/* Navigation & CTAs */}
        <div className="flex items-center gap-3 sm:gap-4">
          <Link
            href="/admin"
            className="flex items-center gap-1.5 text-xs font-semibold text-neutral-300 hover:text-white px-3.5 py-2 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] transition-all"
          >
            <Shield className="w-3.5 h-3.5 text-orange-400" />
            <span className="hidden sm:inline">Paint Catalog</span>
          </Link>

          <a
            href={`https://wa.me/${cleanNumber}?text=${encodeURIComponent("Hello! I used DreamWalls AI to preview Asian Paints colors on my home. Could you assist me with authentic swatches and pricing?")}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs font-bold text-white px-4 py-2 rounded-xl btn-primary shadow-md hover:shadow-lg transition-all"
          >
            <MessageCircle className="w-4 h-4 text-white" />
            <span>WhatsApp Inquiry</span>
          </a>
        </div>
      </div>
    </header>
  );
}
