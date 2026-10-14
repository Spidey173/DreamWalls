import type { Metadata } from "next";
import { Geist, Geist_Mono, Outfit, Space_Grotesk } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DreamWalls - Asian Paints AI Wall Recolor & Shade Studio",
  description:
    "Upload your building exterior or room photo to visualize authentic Asian Paints shades on your real walls with DreamWalls AI.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${outfit.variable} ${spaceGrotesk.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-[#090a0f] text-[#f4f4f8] selection:bg-[#f97316] selection:text-white relative">
        {/* Soft, elegant ambient background glow */}
        <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
          <div className="absolute -top-[25%] left-1/2 -translate-x-1/2 w-[1000px] h-[550px] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-orange-500/10 via-amber-600/5 to-transparent blur-[140px]" />
          <div className="absolute top-[60%] -right-[10%] w-[600px] h-[600px] bg-[radial-gradient(circle,_var(--tw-gradient-stops))] from-amber-500/5 via-orange-600/3 to-transparent blur-[160px]" />
        </div>

        {/* Content Container */}
        <div className="relative z-10 flex-1 flex flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
