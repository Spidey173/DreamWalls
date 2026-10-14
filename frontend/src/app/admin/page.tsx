"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import {
  verifyAdmin,
  getAdminPaints,
  createAdminPaint,
  updateAdminPaint,
  deleteAdminPaint,
  uploadCatalogCSV,
} from "@/lib/api";
import { PaintItem } from "@/lib/types";
import {
  Shield,
  KeyRound,
  Plus,
  Search,
  Trash2,
  Edit2,
  Check,
  X,
  Loader2,
  AlertCircle,
  FileSpreadsheet,
} from "lucide-react";

export default function AdminPage() {
  const [password, setPassword] = useState<string>("");
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [paints, setPaints] = useState<PaintItem[]>([]);
  const [search, setSearch] = useState<string>("");
  const [colorFamily, setColorFamily] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Modal states
  const [isAddOpen, setIsAddOpen] = useState<boolean>(false);
  const [editingPaint, setEditingPaint] = useState<PaintItem | null>(null);

  // Form states
  const [formName, setFormName] = useState("");
  const [formCode, setFormCode] = useState("");
  const [formHex, setFormHex] = useState("#FFFFFF");
  const [formFinish, setFormFinish] = useState("Matte");
  const [formFamily, setFormFamily] = useState("Warm Neutral");
  const [formPrice, setFormPrice] = useState<string>("");

  useEffect(() => {
    const saved = sessionStorage.getItem("dreamwalls_admin_pass") || sessionStorage.getItem("paintmywall_admin_pass");
    if (saved) {
      setPassword(saved);
      checkAuth(saved);
    }
  }, []);

  const checkAuth = async (pwd: string) => {
    setIsLoading(true);
    setAuthError(null);
    try {
      const valid = await verifyAdmin(pwd);
      if (valid) {
        setIsAuthenticated(true);
        sessionStorage.setItem("dreamwalls_admin_pass", pwd);
        sessionStorage.setItem("paintmywall_admin_pass", pwd);
        loadPaints(pwd);
      } else {
        setAuthError("Invalid admin password.");
      }
    } catch {
      setAuthError("Failed to authenticate. Ensure backend server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadPaints = async (pwd: string, s = search, fam = colorFamily) => {
    try {
      const list = await getAdminPaints(pwd, s || undefined, fam || undefined);
      setPaints(list);
    } catch (e: any) {
      setStatusMessage(`Error loading paints: ${e.message}`);
    }
  };

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password.trim()) checkAuth(password.trim());
  };

  const handleSearchChange = (val: string) => {
    setSearch(val);
    loadPaints(password, val, colorFamily);
  };

  const handleFamilyChange = (val: string) => {
    setColorFamily(val);
    loadPaints(password, search, val);
  };

  const openAddModal = () => {
    setEditingPaint(null);
    setFormName("");
    setFormCode("");
    setFormHex("#F4EBD9");
    setFormFinish("Matte");
    setFormFamily("Warm Neutral");
    setFormPrice("");
    setIsAddOpen(true);
  };

  const openEditModal = (p: PaintItem) => {
    setEditingPaint(p);
    setFormName(p.name);
    setFormCode(p.shade_code);
    setFormHex(p.hex);
    setFormFinish(p.finish);
    setFormFamily(p.color_family);
    setFormPrice(p.price ? String(p.price) : "");
    setIsAddOpen(true);
  };

  const handleSavePaint = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName || !formCode) return;

    try {
      const payload = {
        brand: "Asian Paints",
        name: formName.trim(),
        shade_code: formCode.trim(),
        hex: formHex.trim().toUpperCase(),
        finish: formFinish,
        color_family: formFamily,
        price: formPrice ? parseFloat(formPrice) : null,
      };

      if (editingPaint) {
        await updateAdminPaint(password, editingPaint.id, payload);
        setStatusMessage(`Updated shade ${formName}`);
      } else {
        await createAdminPaint(password, payload);
        setStatusMessage(`Added new shade ${formName}`);
      }
      setIsAddOpen(false);
      loadPaints(password);
    } catch (err: any) {
      alert(`Error saving paint: ${err.message}`);
    }
  };

  const handleDeletePaint = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to remove "${name}" from the catalog?`)) return;
    try {
      await deleteAdminPaint(password, id);
      setStatusMessage(`Deleted "${name}"`);
      loadPaints(password);
    } catch (err: any) {
      alert(`Error deleting paint: ${err.message}`);
    }
  };

  const handleCSVUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];

    try {
      const res = await uploadCatalogCSV(password, file);
      setStatusMessage(res.message || "CSV Catalog imported successfully!");
      loadPaints(password);
    } catch (err: any) {
      alert(`Failed to import CSV: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  };

  // If not logged in, render clean login card
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex flex-col bg-[#090a0f] text-white">
        <Navbar />
        <main className="flex-1 flex items-center justify-center p-4">
          <div className="w-full max-w-md glass-panel-glow rounded-3xl p-6 sm:p-8 relative overflow-hidden shadow-2xl">
            <div className="w-12 h-12 rounded-2xl bg-orange-500/10 border border-orange-500/20 text-orange-400 flex items-center justify-center mx-auto mb-4">
              <Shield className="w-6 h-6" />
            </div>
            <div className="text-center space-y-1 mb-6">
              <h2 className="text-xl font-bold text-white tracking-tight">Admin Catalog Access</h2>
              <p className="text-xs text-neutral-400 font-sans">
                Enter your password to manage Asian Paints colors and CSV catalog.
              </p>
            </div>

            <form onSubmit={handleLoginSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-neutral-300 mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <KeyRound className="w-4 h-4 text-orange-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter admin password..."
                    className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-sm text-white focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                    autoFocus
                  />
                </div>
              </div>

              {authError && (
                <div className="p-3.5 rounded-xl bg-red-950/40 border border-red-500/30 text-red-200 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
                  <span>{authError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3.5 rounded-xl btn-primary text-white font-bold text-sm tracking-wide shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                <span>Unlock Catalog</span>
              </button>
            </form>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#090a0f] text-white">
      <Navbar />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 sm:py-10 space-y-6">
        {/* Top Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                Asian Paints Catalog <span className="text-orange-400">Database</span>
              </h1>
              <span className="text-xs font-semibold bg-orange-500/10 text-orange-400 border border-orange-500/20 px-3 py-0.5 rounded-full">
                {paints.length} Shades
              </span>
            </div>
            <p className="text-xs sm:text-sm text-neutral-400 font-sans mt-0.5">
              Manage shade codes, finishes, and bulk CSV uploads for DreamWalls.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <label className="cursor-pointer inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-neutral-200 border border-white/[0.08] hover:border-orange-500/40 text-xs font-semibold transition-all">
              <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
              <span>Import CSV Catalog</span>
              <input type="file" accept=".csv" className="hidden" onChange={handleCSVUpload} />
            </label>

            <button
              onClick={openAddModal}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl btn-primary text-white text-xs font-bold shadow-md transition-all"
            >
              <Plus className="w-4 h-4 stroke-[3]" />
              <span>Add New Shade</span>
            </button>
          </div>
        </div>

        {/* Status Toast */}
        {statusMessage && (
          <div className="p-3.5 rounded-2xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-200 text-xs flex items-center justify-between shadow-md">
            <div className="flex items-center gap-2">
              <Check className="w-4 h-4 text-emerald-400" />
              <span>{statusMessage}</span>
            </div>
            <button onClick={() => setStatusMessage(null)} className="text-emerald-300 hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Filter Bar */}
        <div className="flex flex-col sm:flex-row items-center gap-3 p-3.5 rounded-2xl glass-panel shadow-sm">
          <div className="relative flex-1 w-full">
            <Search className="w-4 h-4 text-neutral-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder="Search by shade code or name..."
              className="w-full pl-10 pr-4 py-2 rounded-xl bg-white/[0.04] border border-white/[0.08] focus:border-orange-500 text-xs sm:text-sm text-white focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
            />
          </div>

          <select
            value={colorFamily}
            onChange={(e) => handleFamilyChange(e.target.value)}
            className="w-full sm:w-60 px-3.5 py-2 rounded-xl bg-white/[0.04] border border-white/[0.08] focus:border-orange-500 text-xs sm:text-sm text-neutral-300 focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
          >
            <option value="" className="bg-neutral-900 text-white">All Color Families</option>
            <option value="Warm Neutral" className="bg-neutral-900 text-white">Warm Neutral</option>
            <option value="Cool Neutral" className="bg-neutral-900 text-white">Cool Neutral</option>
            <option value="Cool Blue" className="bg-neutral-900 text-white">Cool Blue</option>
            <option value="Earthy Green" className="bg-neutral-900 text-white">Earthy Green</option>
            <option value="Warm Yellow" className="bg-neutral-900 text-white">Warm Yellow</option>
            <option value="Terracotta & Red" className="bg-neutral-900 text-white">Terracotta & Red</option>
            <option value="Purple & Lilac" className="bg-neutral-900 text-white">Purple & Lilac</option>
          </select>
        </div>

        {/* Paints Table */}
        <div className="rounded-3xl glass-panel shadow-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead className="bg-white/[0.03] border-b border-white/[0.08] text-neutral-400 font-semibold text-xs uppercase tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Swatch</th>
                  <th className="py-3.5 px-4">Code</th>
                  <th className="py-3.5 px-4">Name</th>
                  <th className="py-3.5 px-4">Family</th>
                  <th className="py-3.5 px-4">Finish</th>
                  <th className="py-3.5 px-4">Price / L</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {paints.map((p) => (
                  <tr key={p.id} className="hover:bg-white/[0.02] transition-colors group">
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="w-6 h-6 rounded-lg border border-white/20 shadow-xs shrink-0 group-hover:scale-105 transition-transform"
                          style={{ backgroundColor: p.hex }}
                        />
                        <span className="text-xs text-neutral-400 font-mono">{p.hex}</span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 font-bold text-white tracking-wide">
                      <span className="bg-white/[0.05] text-orange-400 px-2 py-0.5 rounded border border-white/[0.08]">
                        {p.shade_code}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-white">{p.name}</td>
                    <td className="py-3.5 px-4 text-neutral-400">{p.color_family}</td>
                    <td className="py-3.5 px-4">
                      <span className="bg-white/[0.04] text-neutral-300 border border-white/[0.06] px-2 py-0.5 rounded-md text-xs">
                        {p.finish}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-emerald-400">
                      {p.price ? `₹${p.price}` : "—"}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          onClick={() => openEditModal(p)}
                          className="p-1.5 rounded-lg text-neutral-300 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-all"
                          title="Edit"
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleDeletePaint(p.id, p.name)}
                          className="p-1.5 rounded-lg text-red-400 hover:text-white bg-white/[0.04] hover:bg-red-500/20 border border-white/[0.08] hover:border-red-500/40 transition-all"
                          title="Delete"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {paints.length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-neutral-400 text-xs">
                      No shades found matching your search.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Add / Edit Modal */}
      {isAddOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in">
          <div className="glass-panel-glow rounded-3xl max-w-lg w-full p-6 sm:p-7 shadow-2xl relative overflow-hidden">
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
              <h3 className="font-bold text-base text-white tracking-wide">
                {editingPaint ? "Edit Shade Details" : "Add New Asian Paints Shade"}
              </h3>
              <button
                onClick={() => setIsAddOpen(false)}
                className="p-1.5 rounded-xl text-neutral-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSavePaint} className="space-y-4 pt-4 text-xs sm:text-sm">
              <div>
                <label className="block font-semibold text-neutral-300 mb-1">
                  Shade Name
                </label>
                <input
                  type="text"
                  required
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g. Magnolia White"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-white focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-neutral-300 mb-1">
                    Shade Code
                  </label>
                  <input
                    type="text"
                    required
                    value={formCode}
                    onChange={(e) => setFormCode(e.target.value)}
                    placeholder="e.g. 0915"
                    className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-white uppercase focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-neutral-300 mb-1">
                    Hex Color
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="color"
                      value={formHex}
                      onChange={(e) => setFormHex(e.target.value.toUpperCase())}
                      className="w-10 h-10 rounded-xl border border-white/20 p-0.5 cursor-pointer bg-transparent"
                    />
                    <input
                      type="text"
                      required
                      value={formHex}
                      onChange={(e) => setFormHex(e.target.value)}
                      placeholder="#FFFFFF"
                      className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-white uppercase focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-neutral-300 mb-1">
                    Color Family
                  </label>
                  <select
                    value={formFamily}
                    onChange={(e) => setFormFamily(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-neutral-200 focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                  >
                    <option value="Warm Neutral" className="bg-neutral-900 text-white">Warm Neutral</option>
                    <option value="Cool Neutral" className="bg-neutral-900 text-white">Cool Neutral</option>
                    <option value="Cool Blue" className="bg-neutral-900 text-white">Cool Blue</option>
                    <option value="Earthy Green" className="bg-neutral-900 text-white">Earthy Green</option>
                    <option value="Warm Yellow" className="bg-neutral-900 text-white">Warm Yellow</option>
                    <option value="Terracotta & Red" className="bg-neutral-900 text-white">Terracotta & Red</option>
                    <option value="Purple & Lilac" className="bg-neutral-900 text-white">Purple & Lilac</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-neutral-300 mb-1">
                    Finish Sheen
                  </label>
                  <select
                    value={formFinish}
                    onChange={(e) => setFormFinish(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-neutral-200 focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                  >
                    <option value="Matte" className="bg-neutral-900 text-white">Matte</option>
                    <option value="Eggshell" className="bg-neutral-900 text-white">Eggshell</option>
                    <option value="Sheen" className="bg-neutral-900 text-white">Sheen</option>
                    <option value="Satin" className="bg-neutral-900 text-white">Satin</option>
                    <option value="Gloss" className="bg-neutral-900 text-white">Gloss</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block font-semibold text-neutral-300 mb-1">
                  Price per Liter (₹)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={formPrice}
                  onChange={(e) => setFormPrice(e.target.value)}
                  placeholder="e.g. 450.0"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.1] focus:border-orange-500 text-white focus:outline-none focus:ring-1 focus:ring-orange-500 transition-all"
                />
              </div>

              <div className="pt-4 border-t border-white/[0.08] flex items-center justify-end gap-2.5">
                <button
                  type="button"
                  onClick={() => setIsAddOpen(false)}
                  className="px-4 py-2.5 rounded-xl text-neutral-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] font-semibold transition-all"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2.5 rounded-xl btn-primary text-white font-bold tracking-wide shadow-md transition-all"
                >
                  {editingPaint ? "Save Changes" : "Add to Catalog"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
