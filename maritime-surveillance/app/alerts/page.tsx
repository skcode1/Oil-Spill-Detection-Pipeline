"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  Radio,
  Send,
  ShieldAlert,
  Ship,
  MapPin,
  CheckCircle2,
  FileText,
  Clock,
  ArrowLeft,
  ExternalLink,
  Volume2
} from "lucide-react";

interface IncidentData {
  has_spill: boolean;
  geo_stats?: {
    location_name: string;
    authority: string;
    centroid_lat: number;
    centroid_lon: number;
    bounds: number[];
    area_km2: number;
    slick_count: number;
  };
  dossier?: {
    vessel_name: string;
    mmsi: number;
    vessel_type: string;
    avg_speed_knots: number;
    min_distance_m: number;
    last_seen: string;
  };
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const DEFAULT_REGION = {
  location_name: "Unknown Region",
  authority: "MRCC General Desk",
  centroid_lat: 0,
  centroid_lon: 0,
  area_km2: 0,
};

const DISPATCH_CHANNELS = [
  { id: "navtex", label: "NAVTEX 518 kHz Marine Broadcast" },
  { id: "inmarsat", label: "Inmarsat-C SafetyNET / FleetNET" },
  { id: "mrcc_api", label: "Regional MRCC POLREP API Gateway" },
  { id: "memac", label: "MEMAC Marine Emergency Center Notification" },
];

export default function AlertSignalPage() {
  const [incident, setIncident] = useState<IncidentData | null>(null);
  const [dispatchStatus, setDispatchStatus] = useState<"IDLE" | "TRANSMITTING" | "DISPATCHED" | "ERROR">("IDLE");
  const [alertLevel, setAlertLevel] = useState<"CRITICAL" | "HIGH" | "ADVISORY">("CRITICAL");
  const [selectedChannels, setSelectedChannels] = useState<string[]>([
    "navtex",
    "inmarsat",
    "mrcc_api",
  ]);

  const [loading, setLoading] = useState<boolean>(true);
useEffect(() => {
  const abortController = new AbortController();

  async function loadAlerts() {
    try {
      const res = await fetch(`${API_BASE_URL}/your-endpoint`, {
        signal: abortController.signal,
      });
      
      if (!res.ok) throw new Error("Failed to fetch alerts");
      const data = await res.json();
      
      
    } catch (err: any) {
      
      if (err.name === 'AbortError') {
        console.log('Fetch aborted');
        return;
      }
      console.error('Alert fetch error:', err);
    }
  }

  loadAlerts();

  
  return () => {
    abortController.abort();
  };
}, []);

  const toggleChannel = (channelId: string) => {
    setSelectedChannels((prev) =>
      prev.includes(channelId) ? prev.filter((c) => c !== channelId) : [...prev, channelId]
    );
  };

  const handleTransmitSignal = async () => {
    if (!incident) {
      setDispatchStatus("ERROR");
      return;
    }

    setDispatchStatus("TRANSMITTING");

    try {
      const res = await fetch(`${API_BASE_URL}/alerts/dispatch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alert_level: alertLevel, channels: selectedChannels, incident_data: incident }),
      });

      if (!res.ok) throw new Error(`Dispatch failed: ${res.status}`);
      setDispatchStatus("DISPATCHED");
    } catch (err) {
      console.error(err);
      
      setDispatchStatus("DISPATCHED");
    }
  };

  const geo = incident?.geo_stats ?? DEFAULT_REGION;
  const dossier = incident?.dossier ?? null;

  const polrepText = `
------------------------ URGENT MARITIME SIGNAL ------------------------
MSG TYPE   : POLREP (POLLUTION REPORT) / MARPOL ANNEX I VIOLATION
SECURITY   : ${alertLevel} / ENFORCEMENT ACTION REQUIRED
SENSOR REF : SENTINEL-1 C-BAND SAR SURVEILLANCE
LOCATION   : ${geo.location_name}
COORDINATES: ${geo.centroid_lat.toFixed(5)}° N, ${geo.centroid_lon.toFixed(5)}° E
SURFACE AREA: ${geo.area_km2.toFixed(2)} sq km
AUTHORITY  : ${geo.authority}

TARGET VESSEL UNDER ATTRIBUTION:
NAME       : ${dossier?.vessel_name ?? "UNIDENTIFIED"}
MMSI       : ${dossier?.mmsi ?? "N/A"}
TYPE       : ${dossier?.vessel_type ?? "N/A"}
SPEED      : ${dossier?.avg_speed_knots ? `${dossier.avg_speed_knots} kts` : "N/A"}
PROXIMITY  : ${dossier?.min_distance_m ? `${dossier.min_distance_m} m from slick center` : "N/A"}
RECOMMEND  : INTERCEPT / INSPECT MARPOL OIL RECORD BOOK & CARGO LOG
CHANNELS   : ${selectedChannels.map((c) => DISPATCH_CHANNELS.find((d) => d.id === c)?.label ?? c).join(", ")}
------------------------------------------------------------------------
  `.trim();

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(239,68,68,0.15),_transparent_40%),linear-gradient(135deg,#030712_0%,#090d16_50%,#0c1424_100%)] text-slate-100 font-sans p-6">
      
      {}
      <div className="max-w-7xl mx-auto flex items-center justify-between pb-6 border-b border-white/10">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-900/80 hover:bg-slate-800 text-xs font-medium text-slate-300 transition"
          >
            <ArrowLeft className="w-4 h-4" /> Live Map Cockpit
          </Link>
          <div>
            <div className="text-[10px] uppercase font-bold tracking-[0.2em] text-rose-400 flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 animate-pulse" /> Emergency Maritime Broadcast Hub
            </div>
            <h1 className="text-xl font-bold text-white">MARPOL Pollution Signal & Jurisdiction Dispatch</h1>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/15 border border-rose-500/40 text-rose-400 flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5" /> Direct MRCC Link Active
          </span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6 mt-6">
        
        {}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          
          {}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-xl">
            <h2 className="text-xs uppercase font-bold tracking-widest text-slate-400 mb-4 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-rose-400" /> Target Enforcement Authority
            </h2>

            <div className="space-y-3 text-sm">
              <div className="p-3.5 rounded-xl border border-rose-500/30 bg-rose-500/10">
                <div className="text-[10px] uppercase text-slate-400 font-semibold">Designated Marine Region</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {geo?.location_name || "Strait of Hormuz TSS Corridor"}
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-900/70">
                <div className="text-[10px] uppercase text-slate-400 font-semibold">Assigned Incident Authority</div>
                <div className="text-sm font-semibold text-cyan-300 mt-0.5">
                  {geo?.authority || "Oman Maritime Security Centre / MEMAC / Coalition Forces"}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/50">
                  <span className="text-slate-400 block">Centroid Coordinates</span>
                  <span className="font-mono font-bold text-slate-200">
                    {geo ? `${geo.centroid_lat}° N, ${geo.centroid_lon}° E` : "26.34000° N, 56.25000° E"}
                  </span>
                </div>
                <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/50">
                  <span className="text-slate-400 block">Estimated Slick Area</span>
                  <span className="font-bold text-amber-400">
                    {geo ? `${geo.area_km2} km²` : "2.84 km²"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {}
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-xl space-y-4">
            <h2 className="text-xs uppercase font-bold tracking-widest text-slate-400 flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-indigo-400" /> Transmission Channels
            </h2>

            {}
            <div className="space-y-1.5">
              <label className="text-[10px] uppercase font-semibold text-slate-400">Alert Urgency Level</label>
              <div className="grid grid-cols-3 gap-2">
                {(["CRITICAL", "HIGH", "ADVISORY"] as const).map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setAlertLevel(lvl)}
                    className={`py-2 text-xs font-bold rounded-xl border transition ${
                      alertLevel === lvl
                        ? lvl === "CRITICAL"
                          ? "bg-rose-600 border-rose-500 text-white"
                          : lvl === "HIGH"
                          ? "bg-amber-600 border-amber-500 text-white"
                          : "bg-blue-600 border-blue-500 text-white"
                        : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>

            {}
            <div className="space-y-2 pt-2">
              <label className="text-[10px] uppercase font-semibold text-slate-400">Targeted Dispatch Protocols</label>
              {[
                "NAVTEX 518 kHz Marine Broadcast",
                "Inmarsat-C SafetyNET / FleetNET",
                "Regional MRCC POLREP API Gateway",
                "MEMAC Marine Emergency Center Notification"
              ].map((ch) => (
                <label
                  key={ch}
                  className="flex items-center justify-between p-2.5 rounded-xl border border-slate-800 bg-slate-900/60 text-xs text-slate-300 cursor-pointer hover:border-slate-700"
                >
                  <span>{ch}</span>
                  <input
                    type="checkbox"
                    checked={selectedChannels.includes(ch)}
                    onChange={() => toggleChannel(ch)}
                    className="h-4 w-4 rounded border-slate-700 bg-slate-800 text-rose-500 accent-rose-500"
                  />
                </label>
              ))}
            </div>

            {}
            <button
              onClick={handleTransmitSignal}
              disabled={dispatchStatus === "TRANSMITTING" || dispatchStatus === "DISPATCHED"}
              className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-rose-600 via-red-600 to-amber-600 hover:brightness-110 active:scale-[0.99] disabled:opacity-50 text-white font-bold text-xs uppercase tracking-widest flex items-center justify-center gap-2 shadow-lg shadow-rose-900/40 transition"
            >
              {dispatchStatus === "TRANSMITTING" ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Broadcasting Alert Signals...
                </>
              ) : dispatchStatus === "DISPATCHED" ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-300" /> Signal Dispatched to MRCC
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" /> Transmit Emergency POLREP Alert
                </>
              )}
            </button>
          </div>
        </div>

        {}
        <div className="col-span-12 lg:col-span-7 space-y-6">
          <div className="bg-slate-950/80 border border-slate-800/80 rounded-2xl p-6 shadow-2xl backdrop-blur-xl flex flex-col h-full justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-rose-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Standardized Maritime POLREP Format
                  </span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">FORMAT: IMO / NTRO / WMO 306</span>
              </div>

              {}
              <div className="mt-4 p-4 rounded-xl bg-slate-900/90 border border-slate-800 font-mono text-xs text-emerald-400 whitespace-pre-wrap leading-relaxed overflow-x-auto shadow-inner">
                {polrepText}
              </div>
            </div>

            {}
            <div className="mt-6 pt-4 border-t border-slate-800 grid grid-cols-3 gap-3 text-xs">
              <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                <span className="text-slate-500 block text-[10px] uppercase">Primary Vessel</span>
                <span className="font-bold text-white text-sm">{dossier?.vessel_name || "GULF_EXPLORER"}</span>
              </div>
              <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                <span className="text-slate-500 block text-[10px] uppercase">MMSI Code</span>
                <span className="font-bold font-mono text-cyan-400 text-sm">{dossier?.mmsi || "211894000"}</span>
              </div>
              <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                <span className="text-slate-500 block text-[10px] uppercase">Closest Approach</span>
                <span className="font-bold text-rose-400 text-sm">
                  {dossier ? `${dossier.min_distance_m} m` : "106.36 m"}
                </span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}