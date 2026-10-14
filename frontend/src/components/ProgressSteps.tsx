"use client";

import { useEffect, useState } from "react";
import { Eye, Palette, Paintbrush, CheckCircle2, Sparkles } from "lucide-react";

interface ProgressStepsProps {
  currentStep?: number;
}

const STEPS = [
  {
    title: "Analyzing Scene & Lighting",
    description: "Detecting room geometry, natural daylight, and surface textures",
    icon: Eye,
  },
  {
    title: "Isolating Wall Surfaces",
    description: "Accurately mapping wall planes while protecting windows, trim, and rooflines",
    icon: Palette,
  },
  {
    title: "Applying Asian Paints Colors",
    description: "Synthesizing authentic shade codes with realistic lighting and soft shadows",
    icon: Paintbrush,
  },
];

export default function ProgressSteps({ currentStep: propStep }: ProgressStepsProps) {
  const [internalStep, setInternalStep] = useState(0);

  useEffect(() => {
    if (propStep !== undefined) {
      setInternalStep(propStep);
      return;
    }

    const timer1 = setTimeout(() => setInternalStep(1), 2200);
    const timer2 = setTimeout(() => setInternalStep(2), 4800);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [propStep]);

  return (
    <div className="w-full max-w-md mx-auto glass-panel-glow rounded-3xl p-6 sm:p-8 relative overflow-hidden shadow-2xl">
      {/* Top subtle orange accent highlight */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-orange-500 to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/[0.08]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-orange-500 animate-ping" />
            <span className="text-xs font-semibold text-orange-400 uppercase tracking-wider">
              DreamWalls AI
            </span>
          </div>
          <h3 className="text-lg font-bold text-white">
            Visualizing Your Space
          </h3>
        </div>

        <div className="w-10 h-10 rounded-2xl bg-orange-500/10 border border-orange-500/20 flex items-center justify-center text-orange-400">
          <Sparkles className="w-5 h-5" />
        </div>
      </div>

      {/* Steps List */}
      <div className="space-y-3.5">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isDone = internalStep > idx;
          const isCurrent = internalStep === idx;
          const isPending = internalStep < idx;

          return (
            <div
              key={step.title}
              className={`p-3.5 rounded-2xl border transition-all duration-300 ${
                isCurrent
                  ? "bg-white/[0.06] border-orange-500/50 shadow-sm"
                  : isDone
                  ? "bg-white/[0.03] border-white/[0.06] opacity-90"
                  : "bg-transparent border-white/[0.04] opacity-40"
              }`}
            >
              <div className="flex items-start gap-3.5">
                <div
                  className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-all ${
                    isDone
                      ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-400"
                      : isCurrent
                      ? "bg-orange-500 text-white shadow-md shadow-orange-500/30 animate-pulse"
                      : "bg-white/[0.05] text-neutral-400"
                  }`}
                >
                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 stroke-[2.5]" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>

                <div className="flex-1 min-w-0 pt-0.5">
                  <h4 className="text-xs sm:text-sm font-semibold text-white">
                    {step.title}
                  </h4>
                  <p className="text-[11px] text-neutral-400 mt-0.5 leading-snug">
                    {step.description}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 pt-4 border-t border-white/[0.08] text-center">
        <p className="text-xs text-neutral-400">
          Takes about 5–10 seconds. Preparing high-definition previews...
        </p>
      </div>
    </div>
  );
}
