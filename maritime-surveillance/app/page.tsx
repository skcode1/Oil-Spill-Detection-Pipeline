"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Satellite,
  Ship,
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
  Radio,
  HelpCircle,
} from "lucide-react";

const SurveillanceMap = dynamic(() => import("../src/SurveillanceMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-slate-950 text-xs text-slate-400">
      Loading Mapping Engine...
    </div>
  ),
});

const SAR_ACCEPTED_EXTENSIONS = [".tif", ".tiff"];
const SAR_ACCEPTED_MIME_TYPES = new Set(["image/tiff", "image/x-tiff", "application/octet-stream"]);
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const PIPELINE_UPLOAD_URL = `${API_BASE_URL}/pipeline/upload`;

function isValidSarUpload(file: File | null): boolean {
  if (!file) return false;
  const fileName = file.name.toLowerCase();
  const hasAllowedExtension = SAR_ACCEPTED_EXTENSIONS.some((ext) => fileName.endsWith(ext));
  const hasAllowedMime = !file.type || SAR_ACCEPTED_MIME_TYPES.has(file.type.toLowerCase());
  return hasAllowedExtension && hasAllowedMime;
}

function isUploadCandidate(file: File | null): boolean {
  if (!file || !isValidSarUpload(file)) return false;
  return file.size > 0;
}

async function isAuthenticSarImage(file: File): Promise<boolean> {
  if (!isUploadCandidate(file)) return false;

  try {
    const slice = file.slice(0, 8);
    const buffer = await slice.arrayBuffer();
    const view = new Uint8Array(buffer);
    const isTiffLE = view[0] === 0x49 && view[1] === 0x49 && view[2] === 0x2a && view[3] === 0x00;
    const isTiffBE = view[0] === 0x4d && view[1] === 0x4d && view[2] === 0x00 && view[3] === 0x2a;
    const isBigTiffLE = view[0] === 0x49 && view[1] === 0x49 && view[2] === 0x2b && view[3] === 0x00;
    const isBigTiffBE = view[0] === 0x4d && view[1] === 0x4d && view[2] === 0x00 && view[3] === 0x2b;

    return isTiffLE || isTiffBE || isBigTiffLE || isBigTiffBE;
  } catch (err) {
    console.error("Error inspecting SAR TIFF header:", err);
    return false;
  }
}

function geojsonBounds(featureCollection: any): number[] | null {
  const coords: Array<[number, number]> = [];
  for (const feature of featureCollection?.features ?? []) {
    const geometry = feature?.geometry;
    const polygons =
      geometry?.type === "Polygon"
        ? [geometry.coordinates]
        : geometry?.type === "MultiPolygon"
          ? geometry.coordinates
          : [];
    for (const ring of polygons.flat()) {
      for (const point of ring) {
        if (Array.isArray(point) && point.length >= 2) {
          coords.push([Number(point[0]), Number(point[1])]);
        }
      }
    }
  }
  if (coords.length === 0) return null;
  const lons = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  return [
    Math.min(...lons),
    Math.min(...lats),
    Math.max(...lons),
    Math.max(...lats),
  ];
}

export default function MaritimeDashboard() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [inputMode, setInputMode] = useState<"dataset" | "upload">("dataset");
  const [selectedScene, setSelectedScene] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState<number>(0.90);

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
    const controller = new AbortController();

    const loadStatus = async () => {
      try {
        const scenesRes = await fetch(`${API_BASE_URL}/scenes`, { signal: controller.signal });
        const scenesData = scenesRes.ok ? await scenesRes.json() : { scenes: [] };
        const scenePaths = (scenesData.scenes || []).map((s: any) => s.path).filter(Boolean);
        setScenes(scenePaths);
        if (scenePaths.length > 0) setSelectedScene(scenePaths[0]);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Scenes load error:", err);
      }
    };

    void loadStatus();
    return () => controller.abort();
  }, []);

  const handleRunPipeline = async () => {
    setLoading(true);
    setValidationError(null);

    if (inputMode === "upload") {
      if (!uploadFile || !isValidSarUpload(uploadFile)) {
        setPipelineState(null);
        setValidationError("Invalid SAR image: only GeoTIFF SAR scenes (.tif, .tiff) are accepted.");
        setLoading(false);
        return;
      }

      const isAcceptedUpload = isUploadCandidate(uploadFile);
      if (!isAcceptedUpload) {
        setPipelineState(null);
        setValidationError("Invalid SAR image: please choose a non-empty GeoTIFF SAR scene (.tif, .tiff).");
        setLoading(false);
        return;
      }
    }

    const pipelineOptions = {
      confidence_threshold: threshold,
      run_drift: true,
      run_forecast: true,
      run_attribution: true,
    };

    const controller = new AbortController();

    try {
      let res: Response;
      if (inputMode === "dataset") {
        if (!selectedScene) {
          setValidationError("Select a scene from the dataset first.");
          setLoading(false);
          return;
        }
        res = await fetch(`${API_BASE_URL}/pipeline`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: selectedScene,
            ...pipelineOptions,
          }),
          signal: controller.signal,
        });
      } else {
        const formData = new FormData();
        formData.append("file", uploadFile!);
        formData.append("options", JSON.stringify(pipelineOptions));
        res = await fetch(PIPELINE_UPLOAD_URL, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      }

      const data = await res.json().catch(() => null);
      if (!res.ok || !data) {
        const errorBlock = typeof data?.error === "object" ? data.error : null;
        const detail =
          typeof data?.detail === "string"
            ? data.detail
            : Array.isArray(data?.detail)
              ? (data.detail[0]?.msg ?? "Invalid request.")
              : null;
        const message =
          errorBlock?.message ||
          (typeof data?.error === "string" ? data.error : null) ||
          detail ||
          (typeof data?.message === "string" ? data.message : null) ||
          `Pipeline request failed (HTTP ${res.status}).`;
        setValidationError(message);
        setPipelineState(null);
        return;
      }
      setValidationError(null);
      setPipelineState(data);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      console.error("Pipeline request error:", err);
      setValidationError("Failed to communicate with pipeline server.");
    } finally {
      controller.abort();
      setLoading(false);
    }
  };

  const detection = pipelineState?.detection ?? null;
  const attribution = pipelineState?.attribution ?? null;
  const topCandidate = attribution?.candidates?.[0] ?? null;
  const sceneMeta = detection?.scene ?? null;
  const stats = detection?.statistics ?? null;

  // Exact 3-Way Classification State
  const rawStatus = pipelineState?.spill_status || detection?.status || "No Oil";
  const isNoOil = rawStatus === "No Oil" || rawStatus === "No Spill Detected";
  const isLookalike = rawStatus === "Lookalike" || Boolean(detection?.is_lookalike);
  const isConfirmed = rawStatus === "Confirmed Spill" || rawStatus === "Confirmed";

  const displayStatus = isNoOil ? "No Oil" : isLookalike ? "Lookalike" : "Confirmed Spill";

  const hasSpill = isConfirmed && (detection?.slicks?.length ?? 0) > 0;

  const geolocatedSlick = (detection?.slicks ?? []).find(
    (s: any) => s.centroid_lat != null && s.centroid_lon != null
  );

  const geoStats =
    detection && (hasSpill || isLookalike)
      ? {
          location_name: sceneMeta?.filename ?? selectedScene ?? "SAR scene",
          centroid_lat: geolocatedSlick?.centroid_lat ?? sceneMeta?.center_lat ?? null,
          centroid_lon: geolocatedSlick?.centroid_lon ?? sceneMeta?.center_lon ?? null,
          area_km2: stats?.total_slick_area_km2 ?? 0,
          bounds: geojsonBounds(detection?.spill_geojson),
          slick_count: detection?.slicks?.length ?? 0,
        }
      : null;

  const dossier = (topCandidate && isConfirmed)
    ? {
        vessel_name: topCandidate.vessel_name ?? `MMSI ${topCandidate.mmsi}`,
        mmsi: topCandidate.mmsi,
        vessel_type: topCandidate.vessel_type ?? "Commercial",
        avg_speed_knots: topCandidate.mean_speed_knots != null ? Number(topCandidate.mean_speed_knots).toFixed(1) : "n/a",
        min_distance_m: Math.round(topCandidate.distance_to_origin_m ?? topCandidate.proximity_m ?? 0),
        last_seen: topCandidate.last_seen_utc ?? "n/a",
        score: topCandidate.attribution_score != null ? Math.round(topCandidate.attribution_score) : null,
      }
    : null;

  const spillGeoJson = (isConfirmed || isLookalike) ? (detection?.spill_geojson ?? null) : null;
  const culpritGeoJson =
    isConfirmed &&
    attribution?.attribution_geojson &&
    Object.keys(attribution.attribution_geojson).length > 0
      ? attribution.attribution_geojson
      : null;

  const maskPreviewUrl = detection?.mask_preview_png_base64
    ? `data:image/png;base64,${detection.mask_preview_png_base64}`
    : null;
  const scenePreviewUrl = inputMode === "upload" ? uploadPreview : null;

  const defaultLocation = { lat: 20.0, lon: 0.0, name: "Open Ocean" };
  const hasCentroid = geoStats?.centroid_lat != null && geoStats?.centroid_lon != null;
  const mapCenter: [number, number] = hasCentroid
    ? [Number(geoStats!.centroid_lat), Number(geoStats!.centroid_lon)]
    : [defaultLocation.lat, defaultLocation.lon];

  const locationTitle = hasCentroid
    ? (geoStats!.location_name || "Persian Gulf")
    : (sceneMeta?.filename || defaultLocation.name);

  const coordinatesSubtitle = hasCentroid
    ? `${Number(geoStats!.centroid_lat).toFixed(4)}°N, ${Number(geoStats!.centroid_lon).toFixed(4)}°E`
    : `${defaultLocation.lat.toFixed(4)}°N, ${defaultLocation.lon.toFixed(4)}°E`;

  // Status-dependent tone and accent mapping
  const getStatusBadgeConfig = () => {
    if (isNoOil) {
      return {
        tone: "text-emerald-400",
        accent: "bg-emerald-500/10 border-emerald-400/30",
        icon: CheckCircle2,
      };
    }
    if (isLookalike) {
      return {
        tone: "text-amber-400",
        accent: "bg-amber-500/10 border-amber-400/30",
        icon: HelpCircle,
      };
    }
    return {
      tone: "text-rose-400 animate-pulse",
      accent: "bg-rose-500/10 border-rose-400/30",
      icon: AlertTriangle,
    };
  };

  const statusConfig = getStatusBadgeConfig();

  const statCards = [
    {
      label: "Spill Status",
      value: displayStatus,
      tone: statusConfig.tone,
      accent: statusConfig.accent,
      icon: statusConfig.icon,
    },
    {
      label: "Detected Slicks",
      value: `${isNoOil ? 0 : (geoStats?.slick_count ?? 0)} polygons`,
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
              <div className="text-2xl font-bold tracking-tight text-white">
                MariScope
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
                  accept=".tif,.tiff,image/tiff,image/x-tiff"
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
                      setPipelineState(null);
                      setValidationError("Invalid SAR image: only GeoTIFF SAR scenes (.tif, .tiff) are accepted.");
                      e.target.value = "";
                      return;
                    }

                    const isRealSar = await isAuthenticSarImage(file);
                    if (!isRealSar) {
                      setUploadFile(null);
                      setUploadPreview(null);
                      setPipelineState(null);
                      setValidationError("Invalid SAR image: only authentic GeoTIFF SAR scenes are accepted.");
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
                  <span className="mt-1 block text-[10px] text-slate-500">GeoTIFF only: .tif, .tiff</span>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-slate-400">
              <span>Confidence Threshold</span>
              <span className="font-mono text-indigo-300">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.95"
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
          {loading ? "Analyzing scene..." : "Run pipeline"}
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
                <span className={`h-2 w-2 rounded-full ${isLookalike ? "bg-amber-400" : isConfirmed ? "bg-rose-400" : "bg-emerald-400"}`} />
                <span>{locationTitle}</span>
                <span className="text-slate-500">•</span>
                <span>{coordinatesSubtitle}</span>
              </div>
            </div>
            
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
              spillGeoJson={spillGeoJson}
              culpritGeoJson={culpritGeoJson}
              showSpill={showSpill}
              showVessel={showVessel}
              center={mapCenter}
              regionLabel={locationTitle}
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

          {showMask && maskPreviewUrl && (
            <div>
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">Segmentation Mask</div>
              <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
                <img src={maskPreviewUrl} alt="Segmentation Mask" className="h-40 w-full object-cover" />
              </div>
            </div>
          )}

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
              <Crosshair className="h-3.5 w-3.5 text-indigo-300" /> Spill Geo-Location
            </div>
            {geoStats && !isNoOil ? (
              <div className="space-y-2 text-xs text-slate-300">
                <p>
                  <span className="text-slate-500">Location Name:</span>{" "}
                  <strong className="text-white">{geoStats.location_name || "Open Ocean"}</strong>
                </p>
                <p>
                  <span className="text-slate-500">Centroid:</span>{" "}
                  {geoStats.centroid_lat != null && geoStats.centroid_lon != null
                    ? `${Number(geoStats.centroid_lat).toFixed(4)}° N, ${Number(geoStats.centroid_lon).toFixed(4)}° E`
                    : "not georeferenced (pixel space)"}
                </p>
                <p>
                  <span className="text-slate-500">Estimated Area:</span>{" "}
                  <span className="font-medium text-amber-300">{geoStats.area_km2 ?? 0} km²</span>
                </p>
                {geoStats.bounds && (
                  <p className="text-[10px] text-slate-500 pt-1">
                    Footprint: [{geoStats.bounds.map((b: number) => b.toFixed(3)).join(", ")}]
                  </p>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                {isNoOil ? "No slicks identified in scene." : "No active detections."}
              </p>
            )}
          </div>

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
                {dossier.score != null && (
                  <div><span className="text-slate-500">Attribution Score:</span> <span className="font-bold text-emerald-400">{dossier.score}/100</span></div>
                )}
                <div><span className="text-slate-500">Last ping:</span> <span className="text-[10px] text-slate-400">{dossier.last_seen}</span></div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                {isNoOil
                  ? "Clean marine scene. Vessel correlation skipped."
                  : isLookalike
                  ? "Lookalike detected; vessel attribution suppressed to prevent false accusation."
                  : "No vessel correlated in spatial-temporal envelope."}
              </p>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}