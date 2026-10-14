"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import ProgressSteps from "@/components/ProgressSteps";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import { analyzeRoomImage } from "@/lib/api";
import {
  Upload,
  Camera,
  Sparkles,
  Palette,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  MessageCircle,
  SunMedium,
} from "lucide-react";

const MAX_SIZE_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"];

const PREVIEW_PALETTES = [
  {
    name: "Midnight Navy Suite",
    code: "7246",
    hex: "#1E3A5F",
    image: "/samples/room_navy.jpg",
    tag: "Deep Luxe",
  },
  {
    name: "Amber Orange",
    code: "7905",
    hex: "#F4A836",
    image: "/samples/room_orange.jpg",
    tag: "Warm Vibrant",
  },
];

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Live showcase state
  const [activePreviewIndex, setActivePreviewIndex] = useState<number>(0);

  // Always reset loading state when landing or navigating back to this page
  useEffect(() => {
    setIsLoading(false);

    const handleReset = () => {
      setIsLoading(false);
    };

    window.addEventListener("pageshow", handleReset);
    window.addEventListener("popstate", handleReset);

    return () => {
      window.removeEventListener("pageshow", handleReset);
      window.removeEventListener("popstate", handleReset);
    };
  }, []);

  const handleValidateAndSetFile = (file: File) => {
    setErrorMessage(null);

    // Validate type
    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
      setErrorMessage("Please upload a valid photo in JPG, PNG, or WEBP format.");
      return;
    }

    // Validate size
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`Image size is ${(file.size / (1024 * 1024)).toFixed(1)} MB. Maximum allowed is 10 MB.`);
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleValidateAndSetFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleValidateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const data = await analyzeRoomImage(selectedFile);
      sessionStorage.setItem("dreamwalls_results", JSON.stringify(data));
      sessionStorage.setItem("paintmywall_results", JSON.stringify(data));
      setIsLoading(false);
      router.push("/results");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to analyze and recolor room. Please check your connection.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col selection:bg-orange-500 selection:text-white bg-[#090a0f]">
      <Navbar />

      <main className="flex-1 w-full px-4 sm:px-8 xl:px-12 py-6 lg:py-10 flex flex-col justify-between space-y-10">
        {/* Loading Overlay */}
        {isLoading ? (
          <div className="py-24 flex-1 flex flex-col items-center justify-center space-y-6 animate-in fade-in duration-300">
            <ProgressSteps />
          </div>
        ) : (
          <div className="space-y-12">
            {/* Top Workspace Grid: Upload & Live Visualizer */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 xl:gap-12 items-start">
              {/* Left Column: Upload Deck */}
              <div className="lg:col-span-7 xl:col-span-7 space-y-6">
                {/* Hero Header */}
                <div className="space-y-3">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-xs font-semibold">
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Asian Paints AI Visualizer</span>
                  </div>

                  <h1 className="text-3xl sm:text-4xl xl:text-5xl font-display font-extrabold text-white leading-[1.12] tracking-tight">
                    See how your walls look <span className="text-brand-gradient">before you paint</span>
                  </h1>

                  <p className="text-sm sm:text-base text-neutral-400 leading-relaxed max-w-xl">
                    Upload a photo of your house or room. Our vision AI detects your walls and renders authentic <strong className="text-white">Asian Paints</strong> shade combinations with realistic lighting and zero bleed.
                  </p>
                </div>

                {/* Clean Drag & Drop Area */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative rounded-3xl p-6 sm:p-8 text-center cursor-pointer transition-all duration-300 glass-panel ${
                    isDragging
                      ? "border-orange-500 bg-orange-500/10 scale-[1.01] shadow-xl"
                      : "hover:border-orange-500/40 hover:bg-white/[0.04] group"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                    onChange={handleFileChange}
                  />

                  {previewUrl ? (
                    <div className="space-y-4 relative z-10">
                      <div className="relative w-full max-h-[360px] rounded-2xl overflow-hidden bg-black/40 border border-white/10 flex items-center justify-center shadow-lg mx-auto group">
                        <img
                          src={previewUrl}
                          alt="Photo preview"
                          className="max-w-full max-h-[360px] object-contain rounded-xl"
                        />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2 text-white text-xs font-medium">
                          <Upload className="w-5 h-5 text-orange-400" />
                          <span>Click to change photo</span>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-neutral-300 px-1">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span className="truncate max-w-[180px] text-white font-medium">{selectedFile?.name}</span>
                          <span className="text-neutral-400 bg-white/[0.06] px-2.5 py-0.5 rounded-lg border border-white/[0.08]">
                            {((selectedFile?.size || 0) / (1024 * 1024)).toFixed(1)} MB
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFile(null);
                            setPreviewUrl(null);
                            if (fileInputRef.current) fileInputRef.current.value = "";
                          }}
                          className="text-orange-400 hover:text-orange-300 font-semibold underline text-xs"
                        >
                          Upload Different Photo
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4 py-8 relative z-10">
                      <div className="w-14 h-14 rounded-2xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center mx-auto shadow-sm group-hover:scale-105 group-hover:border-orange-500/40 transition-all">
                        <Upload className="w-6 h-6" />
                      </div>
                      <div>
                        <h3 className="text-base font-bold text-white tracking-wide">
                          Upload Your Space Photo
                        </h3>
                        <p className="text-xs sm:text-sm text-neutral-400 mt-1 font-sans">
                          Drag and drop your image here, or click to browse
                        </p>
                      </div>
                      <div className="inline-flex items-center gap-2.5 text-xs text-neutral-400 bg-white/[0.04] px-3.5 py-1.5 rounded-full border border-white/[0.06]">
                        <span>JPG, PNG, WEBP</span>
                        <span>•</span>
                        <span>Up to 10 MB</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Error Message */}
                {errorMessage && (
                  <div className="p-4 rounded-2xl bg-red-950/40 border border-red-500/30 text-red-200 text-xs sm:text-sm flex items-start gap-3">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
                    <p className="font-medium">{errorMessage}</p>
                  </div>
                )}

                {/* Action Trigger Button */}
                {previewUrl && (
                  <div>
                    <button
                      type="button"
                      onClick={handleStartAnalysis}
                      className="w-full py-4 px-6 rounded-2xl btn-primary text-white font-bold text-base tracking-wide flex items-center justify-center gap-3 shadow-lg"
                    >
                      <Sparkles className="w-5 h-5" />
                      <span>Visualize Paint Colors</span>
                      <ArrowRight className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>

              {/* Right Column: Compact, Clean Live Sample Visualizer */}
              <div className="lg:col-span-5 xl:col-span-5 space-y-4">
                <div className="glass-panel-glow rounded-3xl p-4 sm:p-5 space-y-3.5 shadow-xl max-w-[460px] mx-auto lg:ml-auto">
                  {/* Visualizer Header */}
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-sm sm:text-base font-bold text-white">
                        Interactive Paint Preview
                      </h2>
                      <p className="text-[11px] text-neutral-400 mt-0.5">
                        Slide horizontally to preview Asian Paints wall finishes.
                      </p>
                    </div>

                    <span className="text-[10px] font-semibold bg-orange-500/10 text-orange-400 px-2.5 py-0.5 rounded-full border border-orange-500/20">
                      Live Sample
                    </span>
                  </div>

                  {/* Compact Interactive Before After Canvas */}
                  <div className="rounded-2xl overflow-hidden border border-white/[0.08] shadow-md">
                    <BeforeAfterSlider
                      beforeImage="/samples/room_original.jpg"
                      afterImage={PREVIEW_PALETTES[activePreviewIndex].image}
                      beforeLabel="Original Wall"
                      afterLabel={PREVIEW_PALETTES[activePreviewIndex].name}
                      afterHex={PREVIEW_PALETTES[activePreviewIndex].hex}
                      className="max-h-[250px] sm:max-h-[280px]"
                    />
                  </div>

                  {/* 2 Clean Swatch Selector Chips */}
                  <div className="space-y-1.5 pt-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-neutral-400">Choose preview color:</span>
                      <span className="text-orange-400 font-semibold">
                        {PREVIEW_PALETTES[activePreviewIndex].name} ({PREVIEW_PALETTES[activePreviewIndex].code})
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      {PREVIEW_PALETTES.map((palette, idx) => {
                        const isSelected = activePreviewIndex === idx;
                        return (
                          <button
                            key={palette.code}
                            onClick={() => setActivePreviewIndex(idx)}
                            className={`p-2.5 rounded-xl border text-left transition-all ${
                              isSelected
                                ? "bg-orange-500/15 border-orange-500 shadow-xs"
                                : "bg-white/[0.03] border-white/[0.06] hover:border-white/[0.15]"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span
                                className="w-3.5 h-3.5 rounded-full border border-white/40 shadow-xs shrink-0"
                                style={{ backgroundColor: palette.hex }}
                              />
                              <span className="font-bold text-xs text-white truncate">
                                {palette.code}
                              </span>
                            </div>
                            <p className="text-[11px] font-medium text-neutral-300 truncate">
                              {palette.name}
                            </p>
                            <span className="text-[10px] text-orange-400/80">
                              {palette.tag}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom 4 Feature Highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-6 border-t border-white/[0.08]">
              <div className="p-4 rounded-2xl glass-panel space-y-2 hover:border-orange-500/30 transition-all">
                <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center">
                  <SunMedium className="w-4 h-4" />
                </div>
                <h4 className="text-sm font-bold text-white">Realistic Lighting</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Preserves natural sunlight, shadow gradients, and wall texture for true-to-life color visualization.
                </p>
              </div>

              <div className="p-4 rounded-2xl glass-panel space-y-2 hover:border-orange-500/30 transition-all">
                <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center">
                  <Camera className="w-4 h-4" />
                </div>
                <h4 className="text-sm font-bold text-white">Smart Wall Detection</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Vision AI precisely detects wall planes while keeping windows, doors, sky, and plants untouched.
                </p>
              </div>

              <div className="p-4 rounded-2xl glass-panel space-y-2 hover:border-orange-500/30 transition-all">
                <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center">
                  <Palette className="w-4 h-4" />
                </div>
                <h4 className="text-sm font-bold text-white">Authentic Asian Paints</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Matched with official Asian Paints Royale, Apex, and Ultima fan deck shades and official codes.
                </p>
              </div>

              <div className="p-4 rounded-2xl glass-panel space-y-2 hover:border-orange-500/30 transition-all">
                <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center">
                  <MessageCircle className="w-4 h-4" />
                </div>
                <h4 className="text-sm font-bold text-white">1-Click Dealer Inquiry</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Export your chosen paint schemes directly to WhatsApp for real swatch samples and store pricing.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
