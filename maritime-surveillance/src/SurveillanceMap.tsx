"use client";

import { useEffect, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  LayersControl,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { Compass, Maximize2 } from "lucide-react";

interface MapProps {
  spillGeoJson: any;
  culpritGeoJson: any;
  showSpill: boolean;
  showVessel: boolean;
  center?: [number, number];
  regionLabel?: string;
  sceneBounds?: [number, number, number, number];
  sceneCenter?: [number, number];
}

const createPulseIcon = (color: string) => {
  return L.divIcon({
    className: "custom-pulse-marker",
    html: `
      <div class="relative flex items-center justify-center w-6 h-6">
        <span class="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style="background-color: ${color};"></span>
        <span class="relative inline-flex rounded-full h-3 w-3 border-2 border-slate-900" style="background-color: ${color};"></span>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
};


function MapControlsManager() {
  const map = useMap();

  useEffect(() => {
    map.invalidateSize();
    const scaleControl = L.control.scale({ position: "bottomleft", imperial: false });
    scaleControl.addTo(map);

    return () => {
      scaleControl.remove();
    };
  }, [map]);

  return null;
}

function TacticalMapController({
  center,
  spillGeoJson,
  culpritGeoJson,
}: {
  center?: [number, number];
  spillGeoJson: any;
  culpritGeoJson: any;
}) {
  const map = useMap();

  useEffect(() => {
    const group = new L.FeatureGroup();

    if (spillGeoJson && spillGeoJson.features?.length > 0) {
      const spillLayer = L.geoJSON(spillGeoJson);
      group.addLayer(spillLayer);
    }

    if (culpritGeoJson && culpritGeoJson.features?.length > 0) {
      const vesselLayer = L.geoJSON(culpritGeoJson);
      group.addLayer(vesselLayer);
    }

    if (group.getLayers().length > 0) {
      map.flyToBounds(group.getBounds(), {
        padding: [60, 60],
        maxZoom: 12,
        duration: 1.5,
        easeLinearity: 0.25,
      });
    } else if (center) {
      map.flyTo(center, 8, { duration: 1.2 });
    }

    return () => {
      group.clearLayers();
    };
  }, [center, spillGeoJson, culpritGeoJson, map]);

  return null;
}

export default function SurveillanceMap({
  spillGeoJson,
  culpritGeoJson,
  showSpill,
  showVessel,
  center = [20.0, 0.0],
  regionLabel = "MARITIME SURVEILLANCE",
  sceneBounds,
  sceneCenter,
}: MapProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const mapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  
  useEffect(() => {
    if (!mapRef.current || !sceneBounds) return;
    try {
      mapRef.current.fitBounds(
        [
          [sceneBounds[1], sceneBounds[0]], 
          [sceneBounds[3], sceneBounds[2]], 
        ],
        { padding: [40, 40] }
      );
    } catch {
      
    }
  }, [sceneBounds]);

  if (!isMounted) {
    return <div className="w-full h-full bg-slate-950 min-h-[550px]" />;
  }

  const spillStyle = () => ({
    color: "#ff334b",
    weight: 2,
    opacity: 0.9,
    fillColor: "#e63946",
    fillOpacity: 0.65,
    dashArray: "1, 0",
  });

  const vesselStyle = () => ({
    color: "#00f5d4",
    weight: 3.5,
    opacity: 0.9,
    dashArray: "8, 6",
    lineCap: "round" as L.LineCapShape,
    lineJoin: "round" as L.LineJoinShape,
  });

  const onEachSpillFeature = (feature: any, layer: L.Layer) => {
    layer.bindTooltip(
      `
      <div class="p-1.5 font-sans">
        <div class="flex items-center gap-1.5 text-[11px] font-bold text-red-400 uppercase tracking-wide">
          <span class="w-2 h-2 rounded-full bg-red-500"></span> Oil Slick Polygon
        </div>
        <div class="text-[10px] text-slate-300 mt-1">Class: Hydrocarbon Discharge</div>
      </div>
      `,
      { sticky: true, className: "tactical-tooltip" }
    );

    layer.on({
      mouseover: (e) => {
        const l = e.target;
        l.setStyle({ fillOpacity: 0.85, weight: 3, color: "#ffffff" });
      },
      mouseout: (e) => {
        const l = e.target;
        l.setStyle(spillStyle());
      },
    });
  };

  const onEachVesselFeature = (feature: any, layer: L.Layer) => {
    const props = feature.properties || {};
    layer.bindTooltip(
      `
      <div class="p-1.5 font-sans min-w-[150px]">
        <div class="text-[11px] font-bold text-cyan-400 uppercase tracking-wide">
          🚢 ${props.vessel_name || "Suspect Trajectory"}
        </div>
        <div class="text-[10px] text-slate-300 mt-1">MMSI: <span class="font-mono text-cyan-300">${props.mmsi || "N/A"}</span></div>
        <div class="text-[10px] text-slate-300">Speed: <span class="font-semibold">${props.avg_speed_knots || "N/A"} kts</span></div>
        <div class="text-[10px] text-rose-400 font-semibold mt-0.5">Min Distance: ${props.min_distance_m || "0"} m</div>
      </div>
      `,
      { sticky: true, className: "tactical-tooltip" }
    );
  };

  const pointToLayerVessel = (_feature: any, latlng: L.LatLng) => {
    return L.marker(latlng, { icon: createPulseIcon("#00f5d4") });
  };

  return (
    <div
      id="tactical-map-viewport"
      className={`relative w-full h-full min-h-[550px] rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950 transition-all ${
        isFullscreen ? "fixed inset-0 z-50 rounded-none border-none" : ""
      }`}
    >
      {}
      <div className="absolute top-3 left-3 z-[1000] flex items-center gap-2 bg-slate-900/90 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-slate-700/60 shadow-lg pointer-events-auto">
        <Compass className="w-4 h-4 text-indigo-400 animate-spin" />
        <span className="text-xs font-mono font-semibold tracking-wider text-slate-200">
          {sceneCenter ? (
            <>
              {sceneCenter[1] >= 0 ? "N" : "S"}
              {Math.abs(sceneCenter[1]).toFixed(4)}°{" "}
              {sceneCenter[0] >= 0 ? "E" : "W"}
              {Math.abs(sceneCenter[0]).toFixed(4)}°
            </>
          ) : (
            regionLabel
          )}
        </span>
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse ml-1"></span>
      </div>

      {}
      <div className="absolute top-3 right-3 z-[1000] flex items-center gap-1.5 bg-slate-900/90 backdrop-blur-md p-1 rounded-lg border border-slate-700/60 shadow-lg pointer-events-auto">
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-1.5 hover:bg-slate-800 rounded text-slate-300 hover:text-white transition"
          title="Toggle Fullscreen"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      <MapContainer
        key={`map-${isFullscreen ? "fullscreen" : "inline"}`}
        ref={mapRef}
        center={center}
        zoom={4}
        minZoom={2}
        maxZoom={18}
        scrollWheelZoom={true}
        zoomControl={false}
        attributionControl={false}
        className="w-full h-full"
      >
        <MapControlsManager />

        {}
        <LayersControl position="bottomright">
          <LayersControl.BaseLayer checked name="OpenStreetMap Standard">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="ESRI Satellite Imagery">
            <TileLayer
              attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="Tactical Night Mode">
            <TileLayer
              className="tactical-dark-tiles"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        <TacticalMapController
          center={center}
          spillGeoJson={spillGeoJson}
          culpritGeoJson={culpritGeoJson}
        />

        {showSpill && spillGeoJson && (
          <GeoJSON
            key={`spill-${JSON.stringify(spillGeoJson)}`}
            data={spillGeoJson}
            style={spillStyle}
            onEachFeature={onEachSpillFeature}
          />
        )}

        {showVessel && culpritGeoJson && (
          <GeoJSON
            key={`culprit-${JSON.stringify(culpritGeoJson)}`}
            data={culpritGeoJson}
            style={vesselStyle}
            onEachFeature={onEachVesselFeature}
            pointToLayer={pointToLayerVessel}
          />
        )}
      </MapContainer>

      {}
      <style jsx global>{`
        .leaflet-container {
          background-color: #0b0f19 !important;
          font-family: inherit !important;
        }
        .tactical-dark-tiles {
          filter: brightness(0.6) invert(1) contrast(3) hue-rotate(200deg) saturate(0.3) brightness(0.7);
        }
        .tactical-tooltip {
          background: rgba(15, 23, 42, 0.92) !important;
          border: 1px solid rgba(51, 65, 85, 0.8) !important;
          border-radius: 8px !important;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
          backdrop-filter: blur(8px) !important;
          padding: 4px 8px !important;
          color: #f8fafc !important;
        }
        .leaflet-control-layers {
          background: rgba(15, 23, 42, 0.92) !important;
          border: 1px solid rgba(51, 65, 85, 0.8) !important;
          color: #e2e8f0 !important;
          border-radius: 8px !important;
          font-size: 11px !important;
          backdrop-filter: blur(8px) !important;
        }
        .leaflet-control-scale-line {
          background: rgba(15, 23, 42, 0.75) !important;
          border-color: #64748b !important;
          color: #cbd5e1 !important;
          font-size: 9px !important;
          border-radius: 4px !important;
          backdrop-filter: blur(4px) !important;
        }
      `}</style>
    </div>
  );
}