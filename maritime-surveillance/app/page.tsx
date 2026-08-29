"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Satellite,
  Ship,
  Compass,
  Upload,
  Play,
  CheckCircle2,
  AlertTriangle,
  FileSearch,
  Layers,
  ShieldCheck,
  Gauge,
  Crosshair,
  Activity,
  MapPin,
  Radio
} from "lucide-react";

const SurveillanceMap = dynamic(() => import("../src/SurveillanceMap"), {
  ssr: false,
});

const SAR_ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

function isValidSarUpload(file: File | null) {
  if (!file) return false;
  const fileName = file.name.toLowerCase();
  return SAR_ACCEPTED_EXTENSIONS.some((ext) => fileName.endsWith(ext));
}

async function isAuthenticSarImage(file: File | null): Promise<boolean> {
  if (!file || !isValidSarUpload(file)) return false;

  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to decode image"));
      img.src = objectUrl;
    });

    const width = image.naturalWidth;
    const height = image.naturalHeight;
    if (width < 32 || height < 32 || width > 512 || height > 512) {
      return false;
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return false;

    ctx.drawImage(image, 0, 0);
    const { data } = ctx.getImageData(0, 0, width, height);

    let channelDiffTotal = 0;
    let graySum = 0;
    let squaredSum = 0;
    let pixelCount = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      channelDiffTotal += Math.abs(r - g) + Math.abs(g - b) + Math.abs(r - b);
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      graySum += gray;
      squaredSum += gray * gray;
      pixelCount += 1;
    }

    const rgbDrift = channelDiffTotal / (pixelCount * 3);
    const mean = graySum / pixelCount;
    const variance = squaredSum / pixelCount - mean * mean;
    const std = Math.sqrt(Math.max(variance, 0));

    return rgbDrift < 8 && std >= 5 && std <= 90 && mean >= 10 && mean <= 180;
  } catch {
    return false;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export default function MaritimeDashboard() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [inputMode, setInputMode] = useState<"dataset" | "upload">("dataset");
  const [selectedScene, setSelectedScene] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState<number>(0.5);

  const [showSpill, setShowSpill] = useState(true);
  const [showVessel, setShowVessel] = useState(true);
  const [showMask, setShowMask] = useState(false);

  const [loading, setLoading] = useState(false);
  const [pipelineState, setPipelineState] = useState<any>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!uploadFile) {
      setUploadPreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(uploadFile);
    setUploadPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [uploadFile]);

  useEffect(() => {
    fetch("http://localhost:8000/api/scenes")
      .then((res) => res.json())
      .then((data) => {
        setScenes(data.scenes || []);
        if (data.scenes?.length > 0) setSelectedScene(data.scenes[0]);
      })
      .catch((err) => console.error("API error:", err));

    fetch("http://localhost:8000/api/status")
      .then((res) => res.json())
      .then((data) => setPipelineState(data))
      .catch((err) => console.error("Status error:", err));
  }, []);

  const handleRunPipeline = async () => {
    setLoading(true);
    setValidationError(null);

    if (inputMode === "upload") {
      if (!uploadFile || !isValidSarUpload(uploadFile)) {
        setPipelineState({
          success: false,
          validation_msg: "Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted.",
          has_spill: false,
          geo_stats: null,
          dossier: null,
          spill_geojson: null,
          culprit_geojson: null,
        });
        setValidationError("Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted.");
        setLoading(false);
        return;
      }

      const isRealSar = await isAuthenticSarImage(uploadFile);
      if (!isRealSar) {
        setPipelineState({
          success: false,
          validation_msg: "Invalid SAR image: only authentic SAR scenes matching the dataset are accepted.",
          has_spill: false,
          geo_stats: null,
          dossier: null,
          spill_geojson: null,
          culprit_geojson: null,
        });
        setValidationError("Invalid SAR image: only authentic SAR scenes matching the dataset are accepted.");
        setLoading(false);
        return;
      }
    }

    const formData = new FormData();
    formData.append("mode", inputMode);
    formData.append("threshold", threshold.toString());

    if (inputMode === "dataset") {
      formData.append("selected_scene", selectedScene);
    } else if (uploadFile) {
      formData.append("file", uploadFile);
    }

    try {
      const res = await fetch("http://localhost:8000/api/run-pipeline", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!data.success) {
        setValidationError(data.validation_msg);
      }
      setPipelineState(data);
    } catch (err: any) {
      setValidationError("Failed to communicate with pipeline server.");
    } finally {
      setLoading(false);
    }
  };

  const hasSpill = pipelineState?.has_spill;
  const geoStats = pipelineState?.geo_stats;
  const dossier = pipelineState?.dossier;
  const scenePreviewUrl = inputMode === "upload"
    ? uploadPreview
    : selectedScene
      ? `http://localhost:8000/raw_sar/${selectedScene}`
      : pipelineState?.target_scene_url || null;

  const defaultLocation = { lat: 25.9527, lon: 53.3438, name: "Persian Gulf – Offshore Corridor" };
  const mapCenter: [number, number] = geoStats
    ? [Number(geoStats.centroid_lat), Number(geoStats.centroid_lon)]
    : [defaultLocation.lat, defaultLocation.lon];
  const locationLabel = geoStats
    ? `${geoStats.location_name || "Persian Gulf"} • ${Number(geoStats.centroid_lat).toFixed(4)}°N, ${Number(geoStats.centroid_lon).toFixed(4)}°E`
    : `Persian Gulf • ${defaultLocation.lat.toFixed(4)}°N, ${defaultLocation.lon.toFixed(4)}°E`;

  const statCards = [
    {
      label: "Spill Status",
      value: hasSpill ? "Confirmed" : "Clear",
      tone: hasSpill ? "text-rose-300" : "text-emerald-300",
      accent: hasSpill ? "bg-rose-500/10 border-rose-400/30" : "bg-emerald-500/10 border-emerald-400/30",
      icon: ShieldCheck,
    },
    {
      label: "Detected Slicks",
      value: `${geoStats?.slick_count ?? 0} polygons`,
      tone: "text-slate-100",
      accent: "bg-sky-500/10 border-sky-400/30",
      icon: Activity,
    },
    {
      label: "Top Suspect Vessel",
      value: dossier?.vessel_name ?? "None",
      tone: "text-cyan-300",
      accent: "bg-cyan-500/10 border-cyan-400/30",
      icon: Ship,
    },
    {
      label: "Trajectory Proximity",
      value: dossier ? `${dossier.min_distance_m} m` : "N/A",
      tone: "text-amber-300",
      accent: "bg-amber-500/10 border-amber-400/30",
      icon: Gauge,
    }
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(59,130,246,0.18),_transparent_30%),radial-gradient(circle_at_bottom_right,_rgba(14,165,233,0.12),_transparent_28%),linear-gradient(135deg,#040816_0%,#091120_32%,#0c1424_100%)] text-slate-100 font-sans">
      {/* 1. SIDEBAR CONTROLLER */}
      <aside className="w-[310px] border-r border-white/10 bg-slate-950/60 backdrop-blur-xl p-5 flex flex-col justify-between shadow-[0_0_30px_rgba(15,23,42,0.7)]">
        <div className="space-y-6">
          <div className="flex items-center justify-between rounded-2xl border border-indigo-500/20 bg-indigo-500/5 px-3 py-3">
            <div className="flex items-center gap-3">
              <div className="rounded-xl border border-indigo-400/30 bg-indigo-500/10 p-2 text-indigo-300">
                <Satellite className="h-5 w-5" />
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-indigo-300/80">Mission Control</div>
                <div className="text-sm font-semibold text-white">SAR Intelligence</div>
              </div>
            </div>
            <div className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-emerald-300">
              Online
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800/80 bg-slate-950/70 p-3.5 shadow-inner shadow-slate-950/60">
            <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              <Layers className="h-3.5 w-3.5 text-indigo-300" /> Layer Display
            </div>
            <div className="space-y-2.5">
              {[
                { label: "Slick Vector Layer", checked: showSpill, setter: setShowSpill, color: "text-red-300" },
                { label: "Culprit AIS Trajectory", checked: showVessel, setter: setShowVessel, color: "text-cyan-300" },
                { label: "Binary Segmentation Mask", checked: showMask, setter: setShowMask, color: "text-indigo-300" },
              ].map((item) => (
                <label key={item.label} className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-800 bg-slate-900/60 px-2.5 py-2 text-xs text-slate-200 transition hover:border-slate-700 hover:bg-slate-900">
                  <span className="flex items-center gap-2.5">
                    <span className={`h-2.5 w-2.5 rounded-full ${item.color.replace("text-", "bg-")}`} />
                    {item.label}
                  </span>
                  <input
                    type="checkbox"
                    checked={item.checked}
                    onChange={(e) => item.setter(e.target.checked)}
                    className="h-4 w-4 rounded border-slate-700 bg-slate-800 text-indigo-500 accent-indigo-500 focus:ring-0"
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">SAR Image Ingestion</div>
            <div className="grid grid-cols-2 gap-2 rounded-xl border border-slate-800 bg-slate-950/80 p-1">
              <button
                onClick={() => setInputMode("dataset")}
                className={`rounded-lg px-2 py-2 text-xs font-medium transition ${inputMode === "dataset" ? "bg-gradient-to-r from-indigo-600 to-blue-500 text-white shadow-lg shadow-indigo-900/30" : "text-slate-400 hover:text-slate-200"}`}
              >
                Dataset
              </button>
              <button
                onClick={() => setInputMode("upload")}
                className={`rounded-lg px-2 py-2 text-xs font-medium transition ${inputMode === "upload" ? "bg-gradient-to-r from-indigo-600 to-blue-500 text-white shadow-lg shadow-indigo-900/30" : "text-slate-400 hover:text-slate-200"}`}
              >
                Upload Scene
              </button>
            </div>

            {inputMode === "dataset" ? (
              <select
                value={selectedScene}
                onChange={(e) => setSelectedScene(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2.5 text-xs text-slate-200 outline-none transition focus:border-indigo-400"
              >
                {scenes.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            ) : (
              <div className="relative rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950/80 p-4 text-center transition hover:border-indigo-400/60">
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
                  onChange={async (e) => {
                    const file = e.target.files?.[0] || null;
                    if (!file) {
                      setUploadFile(null);
                      setUploadPreview(null);
                      return;
                    }
                    if (!isValidSarUpload(file)) {
                      setUploadFile(null);
                      setUploadPreview(null);
                      setPipelineState({
                        success: false,
                        validation_msg: "Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted.",
                        has_spill: false,
                        geo_stats: null,
                        dossier: null,
                        spill_geojson: null,
                        culprit_geojson: null,
                      });
                      setValidationError("Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted.");
                      e.target.value = "";
                      return;
                    }

                    const isRealSar = await isAuthenticSarImage(file);
                    if (!isRealSar) {
                      setUploadFile(null);
                      setUploadPreview(null);
                      setPipelineState({
                        success: false,
                        validation_msg: "Invalid SAR image: only authentic SAR scenes matching the dataset are accepted.",
                        has_spill: false,
                        geo_stats: null,
                        dossier: null,
                        spill_geojson: null,
                        culprit_geojson: null,
                      });
                      setValidationError("Invalid SAR image: only authentic SAR scenes matching the dataset are accepted.");
                      e.target.value = "";
                      return;
                    }

                    setValidationError(null);
                    setUploadFile(file);
                  }}
                  className="absolute inset-0 z-10 cursor-pointer opacity-0"
                />
                <div className="relative z-0">
                  <Upload className="mx-auto mb-2 h-5 w-5 text-slate-400" />
                  <p className="truncate text-xs font-medium text-slate-200">
                    {uploadFile ? uploadFile.name : "Select SAR scene"}
                  </p>
                  <span className="mt-1 block text-[10px] text-slate-500">.png, .jpg, .tif accepted</span>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-slate-400">
              <span>Confidence</span>
              <span className="font-mono text-indigo-300">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-indigo-500"
            />
          </div>
        </div>

        <button
          onClick={handleRunPipeline}
          disabled={loading}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 via-blue-500 to-cyan-500 px-4 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-white shadow-[0_18px_30px_rgba(59,130,246,0.35)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white" />
          ) : (
            <Play className="h-4 w-4 fill-current" />
          )}
          {loading ? "Analyzing scene" : "Run pipeline"}
        </button>
      </aside>

      {/* 2. MAIN COCKPIT & MAP */}
      <main className="flex-1 overflow-hidden p-6">
        <div className="flex h-full flex-col gap-4">
          <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/45 px-5 py-4 shadow-[0_10px_40px_rgba(15,23,42,0.35)] backdrop-blur-xl">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-indigo-300/80">Maritime Defense Solution</div>
              <h2 className="mt-1 text-lg font-semibold text-white">Sentinel-1 SAR Oil Spill & AIS Culprit Tracking</h2>
              <div className="mt-1.5 inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] text-slate-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                {locationLabel}
              </div>
            </div>
            
            {/* Header Right Action Items */}
            <div className="flex items-center gap-3">
              <Link
                href="/alerts"
                className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 transition shadow-lg shadow-rose-950/40"
              >
                <Radio className="w-3.5 h-3.5 animate-pulse text-rose-400" />
                Emergency Alert Signal Hub
              </Link>

              {validationError ? (
                <span className="inline-flex items-center gap-2 rounded-full border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300">
                  <AlertTriangle className="h-3.5 w-3.5" /> {validationError}
                </span>
              ) : (
                <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300">
                  <CheckCircle2 className="h-3.5 w-3.5" /> Scene Validated
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4">
            {statCards.map((card) => {
              const Icon = card.icon;
              return (
                <div key={card.label} className={`rounded-2xl border p-3.5 shadow-[0_10px_25px_rgba(15,23,42,0.2)] ${card.accent}`}>
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-slate-400">{card.label}</div>
                    <div className="rounded-lg border border-white/10 bg-slate-950/40 p-1.5 text-slate-200">
                      <Icon className="h-3.5 w-3.5" />
                    </div>
                  </div>
                  <div className={`text-lg font-semibold ${card.tone}`}>{card.value}</div>
                </div>
              );
            })}
          </div>

          <div className="min-h-0 flex-1 overflow-hidden rounded-3xl border border-white/10 bg-slate-950/40 p-2 shadow-[0_20px_50px_rgba(2,6,23,0.55)] backdrop-blur-sm">
            <SurveillanceMap
              spillGeoJson={pipelineState?.spill_geojson}
              culpritGeoJson={pipelineState?.culprit_geojson}
              showSpill={showSpill}
              showVessel={showVessel}
              center={mapCenter}
            />
          </div>
        </div>
      </main>

      {/* 3. INCIDENT DOSSIER PANE */}
      <aside className="w-[320px] border-l border-white/10 bg-slate-950/60 p-5 backdrop-blur-xl">
        <div className="space-y-5">
          <div>
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
              <FileSearch className="h-3.5 w-3.5 text-indigo-300" /> Target SAR Scene
            </div>
            <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-inner shadow-slate-950/80">
              <div className="aspect-square">
                {scenePreviewUrl ? (
                  <img src={scenePreviewUrl} alt="SAR Target" className="h-full w-full object-cover" />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500">No scene loaded</div>
                )}
              </div>
            </div>
          </div>

          {showMask && pipelineState?.mask_url && (
            <div>
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">Segmentation Mask</div>
              <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
                <img src={pipelineState.mask_url} alt="Segmentation Mask" className="h-40 w-full object-cover" />
              </div>
            </div>
          )}

          {/* Incident Geo-Location Block with Location Name */}
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
              <Crosshair className="h-3.5 w-3.5 text-indigo-300" /> Spill Geo-Location
            </div>
            {geoStats ? (
              <div className="space-y-2 text-xs text-slate-300">
                <p>
                  <span className="text-slate-500">Location Name:</span>{" "}
                  <strong className="text-white">{geoStats.location_name || "Persian Gulf TSS"}</strong>
                </p>
                <p>
                  <span className="text-slate-500">Centroid:</span> {geoStats.centroid_lat}° N, {geoStats.centroid_lon}° E
                </p>
                {geoStats.authority && (
                  <p>
                    <span className="text-slate-500">Authority:</span>{" "}
                    <span className="text-slate-300 font-medium">{geoStats.authority}</span>
                  </p>
                )}
                <p>
                  <span className="text-slate-500">Estimated Area:</span>{" "}
                  <span className="font-medium text-amber-300">{geoStats.area_km2} km²</span>
                </p>
                <p className="text-[10px] text-slate-500 pt-1">Footprint: [{geoStats.bounds.join(", ")}]</p>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No active detections.</p>
            )}
          </div>

          {/* Culprit Dossier */}
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
              <Ship className="h-3.5 w-3.5 text-cyan-300" /> Culprit Dossier
            </div>
            {dossier ? (
              <div className="space-y-2 text-xs text-slate-300">
                <div><span className="text-slate-500">Name:</span> <span className="font-semibold text-white">{dossier.vessel_name}</span></div>
                <div><span className="text-slate-500">MMSI:</span> <span className="font-mono text-cyan-300">{dossier.mmsi}</span></div>
                <div><span className="text-slate-500">Type:</span> {dossier.vessel_type}</div>
                <div><span className="text-slate-500">Speed:</span> {dossier.avg_speed_knots} knots</div>
                <div><span className="text-slate-500">Approach:</span> <span className="font-medium text-rose-300">{dossier.min_distance_m} m</span></div>
                <div><span className="text-slate-500">Last ping:</span> <span className="text-[10px] text-slate-400">{dossier.last_seen}</span></div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No vessel correlated yet.</p>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}