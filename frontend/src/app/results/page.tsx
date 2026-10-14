"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import ShadeCard from "@/components/ShadeCard";
import SchemeCard from "@/components/SchemeCard";
import MaskEditorModal from "@/components/MaskEditorModal";
import { AnalysisAndRecolorResponse, RepaintedShadeResult, SchemeResult } from "@/lib/types";
import { getFullMediaUrl, regenerateScheme } from "@/lib/api";
import {
  Paintbrush,
  MessageCircle,
  ArrowLeft,
  Sparkles,
  SunMedium,
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  Sliders,
} from "lucide-react";

export default function ResultsPage() {
  const router = useRouter();
  const [data, setData] = useState<AnalysisAndRecolorResponse | null>(null);
  const [activeShadeIndex, setActiveShadeIndex] = useState<number>(0);
  const [isMaskEditorOpen, setIsMaskEditorOpen] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [regeneratingSchemeIndex, setRegeneratingSchemeIndex] = useState<number | null>(null);

  const whatsappNumber = process.env.NEXT_PUBLIC_SHOP_WHATSAPP_NUMBER || "+919876543210";
  const cleanNumber = whatsappNumber.replace(/[^0-9]/g, "");

  useEffect(() => {
    const raw = sessionStorage.getItem("dreamwalls_results") || sessionStorage.getItem("paintmywall_results");
    if (!raw) {
      router.push("/");
      return;
    }
    try {
      const parsed: AnalysisAndRecolorResponse = JSON.parse(raw);
      setData(parsed);
    } catch {
      router.push("/");
    }
  }, [router]);

  const handleRecolorComplete = (
    updatedResults: RepaintedShadeResult[],
    updatedSchemes?: SchemeResult[]
  ) => {
    if (!data) return;
    const updatedData: AnalysisAndRecolorResponse = {
      ...data,
      results: updatedResults.length > 0 ? updatedResults : data.results,
      schemes: updatedSchemes && updatedSchemes.length > 0 ? updatedSchemes : data.schemes,
    };
    setData(updatedData);
    sessionStorage.setItem("dreamwalls_results", JSON.stringify(updatedData));
    sessionStorage.setItem("paintmywall_results", JSON.stringify(updatedData));

    setToastMessage("Wall colors updated successfully!");
    setTimeout(() => setToastMessage(null), 4000);
  };

  const handleRegenerateScheme = async (schemeIndex: number) => {
    if (!data) return;
    setRegeneratingSchemeIndex(schemeIndex);
    try {
      const updatedScheme = await regenerateScheme(data.session_id, schemeIndex);
      const updatedSchemes = (data.schemes || []).map((s) =>
        s.scheme_index === schemeIndex ? updatedScheme : s
      );
      const updatedData: AnalysisAndRecolorResponse = {
        ...data,
        schemes: updatedSchemes,
      };
      setData(updatedData);
      sessionStorage.setItem("dreamwalls_results", JSON.stringify(updatedData));
      sessionStorage.setItem("paintmywall_results", JSON.stringify(updatedData));
      setToastMessage(`Scheme #${schemeIndex} preview refreshed!`);
      setTimeout(() => setToastMessage(null), 4000);
    } catch (err: any) {
      setToastMessage(err.message || "Failed to regenerate scheme");
      setTimeout(() => setToastMessage(null), 5000);
    } finally {
      setRegeneratingSchemeIndex(null);
    }
  };

  if (!data) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#090a0f] text-white">
        <div className="w-12 h-12 rounded-full border-2 border-orange-500/20 border-t-orange-500 animate-spin" />
        <p className="mt-4 text-xs font-medium text-neutral-400">Loading your visualization...</p>
      </div>
    );
  }

  const hasSchemes = Boolean(data.schemes && data.schemes.length > 0);
  const activeScheme = hasSchemes ? data.schemes![activeShadeIndex] || data.schemes![0] : null;
  const activeResult = data.results[activeShadeIndex] || data.results[0];

  // WhatsApp text
  let whatsappUrl = "";
  if (hasSchemes && activeScheme) {
    const m = activeScheme.shades.find((s) => s.role === "main") || activeScheme.shades[0];
    const a = activeScheme.shades.find((s) => s.role === "accent") || activeScheme.shades[1] || m;
    const t = activeScheme.shades.find((s) => s.role === "trim_gate") || activeScheme.shades[2] || a;
    const schemeText = `Hello! I used DreamWalls to design my exterior building. I loved Scheme #${activeScheme.scheme_index}: Main Facade ${m.shade_name} (${m.shade_code}), Accent ${a.shade_name} (${a.shade_code}), Gate/Trim ${t.shade_name} (${t.shade_code}). Could you help me with pricing and shade cards?`;
    whatsappUrl = `https://wa.me/${cleanNumber}?text=${encodeURIComponent(schemeText)}`;
  } else {
    const shadesSummary = data.results.map((r) => `${r.shade_name} (${r.shade_code})`).join(", ");
    const allShadesWhatsappText = `Hello! I used DreamWalls to design my ${data.room_analysis.room_type || "room"}. The recommended Asian Paints shades are: ${shadesSummary}. Could you help me with pricing and shade cards?`;
    whatsappUrl = `https://wa.me/${cleanNumber}?text=${encodeURIComponent(allShadesWhatsappText)}`;
  }

  const isExterior = data.room_analysis.scene_type === "exterior";
  const accentEmpty = isExterior && (!data.coverage_pct || (data.coverage_pct.accent ?? 0) < 0.1);
  const trimEmpty = isExterior && (!data.coverage_pct || (data.coverage_pct.trim_gate ?? 0) < 0.1);

  return (
    <div className="min-h-screen flex flex-col selection:bg-orange-500 selection:text-white bg-[#090a0f]">
      <Navbar />

      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 bg-neutral-900 border border-orange-500/30 text-white px-5 py-3 rounded-2xl shadow-xl flex items-center gap-2.5 text-xs font-semibold animate-in slide-in-from-bottom duration-200">
          <CheckCircle className="w-4 h-4 text-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Full-Screen Studio Workspace */}
      <main className="flex-1 w-full px-4 sm:px-8 xl:px-12 py-5 sm:py-6 space-y-6">
        {/* Navigation & Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-3 border-b border-white/[0.08]">
          <div className="flex items-center gap-3.5">
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-3.5 py-2.5 rounded-2xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-neutral-200 hover:text-white transition-all text-xs font-semibold shadow-xs"
              title="Upload another photo"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Upload New Photo</span>
            </Link>
            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <h1 className="text-2xl sm:text-3xl font-display font-extrabold text-white tracking-tight">
                  {hasSchemes ? "Exterior Color Schemes" : "Room Color Palette"}
                </h1>
                <span className="text-xs font-semibold bg-orange-500/10 text-orange-400 border border-orange-500/20 px-3 py-0.5 rounded-full">
                  {hasSchemes ? "4 Exterior Schemes" : "4 Curated Shades"}
                </span>
              </div>
              <p className="text-xs sm:text-sm text-neutral-400 mt-0.5">
                {hasSchemes
                  ? `Asian Paints 3-coat color palettes for your ${data.room_analysis.building_type || "building"} with natural daylight matching`
                  : `Color palettes for your ${data.room_analysis.room_type || "room"} in ${data.room_analysis.style || "modern"} style`}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setIsMaskEditorOpen(true)}
              className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-neutral-200 border border-white/[0.08] hover:border-orange-500/40 text-xs font-semibold transition-all"
            >
              <Paintbrush className="w-4 h-4 text-orange-400" />
              <span>{isExterior ? "Fine-Tune Wall Areas" : "Adjust Wall Mask"}</span>
            </button>

            <a
              href={whatsappUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl btn-primary text-white text-xs font-bold tracking-wide shadow-md transition-all"
            >
              <MessageCircle className="w-4 h-4" />
              <span>Send Palette to WhatsApp</span>
            </a>
          </div>
        </div>

        {/* Demo Mode Notice */}
        {data.mock && (
          <div className="p-4 rounded-2xl bg-amber-950/30 border border-amber-500/30 text-amber-200 flex items-start gap-3 text-xs">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-white flex items-center gap-2">
                <span className="bg-amber-500 text-neutral-900 text-[10px] px-2 py-0.5 rounded-full font-bold">Preview Mode</span>
                <span>Unpainted Original Photo with Mock Watermark</span>
              </p>
              <p className="text-amber-200/80 leading-relaxed mt-1">
                Configure your API key in the backend environment for full photorealistic AI inpainting.
              </p>
            </div>
          </div>
        )}

        {/* Empty Zone Tip Banner for Exterior */}
        {hasSchemes && (accentEmpty || trimEmpty) && (
          <div className="p-3.5 rounded-2xl bg-blue-950/30 border border-blue-500/30 text-blue-200 flex items-start sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              <Lightbulb className="w-4 h-4 text-blue-400 shrink-0" />
              <span>
                Tip: Only the main facade layer is active. Click{" "}
                <button
                  onClick={() => setIsMaskEditorOpen(true)}
                  className="underline font-bold text-white hover:text-blue-300"
                >
                  "Fine-Tune Wall Areas"
                </button>{" "}
                to assign cantilevered slabs, trims, or gates to preview coordinated 3-color schemes.
              </span>
            </div>
          </div>
        )}

        {/* Clean Space Metadata Ribbon */}
        <div className="flex flex-wrap items-center gap-2.5 p-3 bg-white/[0.03] border border-white/[0.06] rounded-2xl text-xs text-neutral-300">
          <span className="font-semibold text-white flex items-center gap-1.5 px-2">
            <Sparkles className="w-3.5 h-3.5 text-orange-400" /> Space Details:
          </span>
          <span className="bg-white/[0.04] px-3 py-1 rounded-xl border border-white/[0.06]">
            Type: <strong className="text-white capitalize">{isExterior ? (data.room_analysis.building_type || "Exterior") : (data.room_analysis.room_type || "Interior")}</strong>
          </span>
          <span className="bg-white/[0.04] px-3 py-1 rounded-xl border border-white/[0.06] flex items-center gap-1">
            <SunMedium className="w-3.5 h-3.5 text-amber-400" /> Lighting:{" "}
            <strong className="text-white capitalize">{isExterior ? (data.room_analysis.light_conditions || data.room_analysis.lighting) : data.room_analysis.lighting}</strong>
          </span>
          <span className="bg-white/[0.04] px-3 py-1 rounded-xl border border-white/[0.06] flex items-center gap-1.5">
            Base Wall Color:
            <span
              className="w-3 h-3 rounded-full border border-white/30"
              style={{ backgroundColor: data.room_analysis.current_wall_color_hex }}
            />
            <strong className="text-orange-300">{data.room_analysis.current_wall_color_hex}</strong>
          </span>
          <span className="bg-white/[0.04] px-3 py-1 rounded-xl border border-white/[0.06]">
            Style: <strong className="text-white capitalize">{data.room_analysis.style}</strong>
          </span>
          <span className="bg-white/[0.04] px-3 py-1 rounded-xl border border-white/[0.06]">
            Mood: <strong className="text-white capitalize">{data.room_analysis.mood}</strong>
          </span>
        </div>

        {/* Dual-Column Full-Screen Studio Workspace Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
          {/* Left / Center Viewport: Interactive Comparison Stage */}
          <section className="xl:col-span-7 2xl:col-span-8 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-base sm:text-lg font-bold text-white">
                  Before &amp; After Comparison
                </h2>
                <p className="text-xs text-neutral-400">
                  {hasSchemes && activeScheme
                    ? `Compare: Original photo vs Scheme #${activeScheme.scheme_index}`
                    : `Compare: Original photo vs Asian Paints "${activeResult?.shade_name || 'Selected'}"`}
                </p>
              </div>

              {/* Scheme Switcher Tabs */}
              <div className="flex items-center gap-1.5 bg-white/[0.04] p-1 rounded-2xl border border-white/[0.08] overflow-x-auto">
                {hasSchemes
                  ? data.schemes!.map((s, idx) => {
                      const m = s.shades.find((sh) => sh.role === "main") || s.shades[0];
                      return (
                        <button
                          key={s.scheme_index}
                          onClick={() => setActiveShadeIndex(idx)}
                          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 ${
                            activeShadeIndex === idx
                              ? "btn-primary text-white shadow-sm"
                              : "text-neutral-300 hover:text-white hover:bg-white/5"
                          }`}
                        >
                          <span
                            className="w-2.5 h-2.5 rounded-full border border-white/40"
                            style={{ backgroundColor: m.hex }}
                          />
                          <span>Scheme #{s.scheme_index}</span>
                        </button>
                      );
                    })
                  : data.results.map((r, idx) => (
                      <button
                        key={r.shade_code}
                        onClick={() => setActiveShadeIndex(idx)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 ${
                          activeShadeIndex === idx
                            ? "btn-primary text-white shadow-sm"
                            : "text-neutral-300 hover:text-white hover:bg-white/5"
                        }`}
                      >
                        <span
                          className="w-2.5 h-2.5 rounded-full border border-white/40"
                          style={{ backgroundColor: r.hex }}
                        />
                        <span>Shade #{idx + 1}</span>
                      </button>
                    ))}
              </div>
            </div>

            {/* Expansive Before / After Interactive Viewport */}
            <div className="rounded-3xl overflow-hidden shadow-2xl border border-white/[0.08] bg-black/40">
              <BeforeAfterSlider
                beforeImage={getFullMediaUrl(data.original_image_url)}
                afterImage={getFullMediaUrl(hasSchemes && activeScheme ? activeScheme.image_url : activeResult.image_url)}
                beforeLabel="Original Walls"
                afterLabel={
                  hasSchemes && activeScheme
                    ? `Scheme #${activeScheme.scheme_index} (${activeScheme.shades[0]?.shade_name})`
                    : `${activeResult.shade_name} (${activeResult.shade_code})`
                }
                afterHex={hasSchemes && activeScheme ? activeScheme.shades[0]?.hex : activeResult.hex}
              />
            </div>

            {/* Helper Bar */}
            <div className="flex items-center justify-between p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06] text-xs text-neutral-400">
              <div className="flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-orange-400" />
                <span>Drag the slider horizontally to view the seamless color transformation.</span>
              </div>
              <button
                onClick={() => setIsMaskEditorOpen(true)}
                className="text-orange-400 hover:text-orange-300 font-semibold underline text-xs shrink-0"
              >
                Refine Wall Mask &rarr;
              </button>
            </div>
          </section>

          {/* Right Column: Architectural Schemes & Paint Deck */}
          <section className="xl:col-span-5 2xl:col-span-4 space-y-4">
            <div>
              <h2 className="text-base sm:text-lg font-bold text-white">
                {hasSchemes ? "Asian Paints Combinations" : "Asian Paints Palette"}
              </h2>
              <p className="text-xs text-neutral-400">
                {hasSchemes
                  ? "Click any scheme to preview it on the main image above"
                  : "Curated Asian Paints shades matching your room lighting"}
              </p>
            </div>

            {/* Schemes Stacked Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-4 max-h-[78vh] overflow-y-auto pr-1 scrollbar-thin">
              {hasSchemes
                ? data.schemes!.map((scheme, idx) => (
                    <SchemeCard
                      key={scheme.scheme_index}
                      scheme={scheme}
                      isSelected={activeShadeIndex === idx}
                      onSelect={() => setActiveShadeIndex(idx)}
                      whatsappNumber={whatsappNumber}
                      onRegenerate={handleRegenerateScheme}
                      isRegenerating={regeneratingSchemeIndex === scheme.scheme_index}
                    />
                  ))
                : data.results.map((result, idx) => (
                    <ShadeCard
                      key={result.shade_code}
                      result={result}
                      isSelected={activeShadeIndex === idx}
                      onSelect={() => setActiveShadeIndex(idx)}
                      whatsappNumber={whatsappNumber}
                    />
                  ))}
            </div>

            {/* Paint Dealer Physical Swatch Notice */}
            <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] text-neutral-300 flex items-start gap-3 shadow-md">
              <AlertTriangle className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-xs font-bold text-white">Authentic Asian Paints Swatches</h4>
                <p className="text-[11px] text-neutral-400 leading-relaxed">
                  Real wall finishes may vary slightly with wall moisture and sun exposure. Ask your Asian Paints dealer for physical fan deck swatches before final tinting.
                </p>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Mask Editor Modal */}
      <MaskEditorModal
        isOpen={isMaskEditorOpen}
        onClose={() => setIsMaskEditorOpen(false)}
        sessionId={data.session_id}
        originalImageUrl={data.original_image_url}
        maskUrl={data.mask_url}
        masks={data.masks}
        coveragePct={(data.coverage_by_layer as any) || (data.coverage_pct as any)}
        isExterior={isExterior}
        initialSchemes={data.schemes}
        protectedHatchedUrl={data.protected_hatched_url}
        protectedMaskUrl={data.protected_mask_url}
        skyFractionByLayer={data.sky_fraction_by_layer}
        onRecolorComplete={handleRecolorComplete}
      />
    </div>
  );
}
