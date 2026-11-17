# SIH26143 — Hybrid Neuro-Symbolic Maritime Surveillance (Architecture Upgrade)

This document records the end-to-end upgrade of the existing FastAPI service layer
(`app/services/`) to stronger open-source weights and architectures, without
rebuilding the service layer itself. Every stage remains deterministic and
reproducible; the optional LLM layer is strictly downstream and never blocks the
evidence dossier.

## Pipeline stages

| Stage | Before | After (open-source upgrade) | Key advantage |
|-------|--------|-----------------------------|---------------|
| 1. Detection & Classification | 3-class U-Net (ResNet-34) | 5-class DeepLabV3+ / ResNet-34 (`segmentation_models_pytorch`) | Multi-class masks (Sea/Oil/Look-Alike/Ship/Land) remove algal-bloom and ship false positives |
| 2. Geospatial backbone | Single-scene RGB/grayscale tiling | Clay Foundation Model / Prithvi-EO-2.0-compatible SAR+optical fine-tuning | Cross-sensor generalization from Sentinel-1 SAR and Sentinel-2 optical |
| 3. Drift & Backtracking | Built-in Lagrangian / OpenDrift | OpenDrift (OpenOil) + Copernicus Marine / ERA5 NetCDF | Physics-based advection-diffusion + Stokes drift + oil weathering |
| 4. Vessel Attribution | Weighted heuristic + envelope pruning | LightGBM/XGBoost-style explainable scoring + SAR visual confirmation + SHAP-ready breakdown | Non-linear CPA/heading/gap interactions + dark-vessel alerts |
| 5. Dossier Synthesis (optional LLM) | Raw JSON / Next.js rendering | Google Gemini 2.5 Flash (`google-genai`) NTRO narrative | Structured intelligence narrative from verified dossier JSON |

## Data flow

```
Sentinel-1 SAR / Optical GeoTIFF
        |
        v
Stage 1: detection_service.py  (5-class segmentation -> GeoJSON polygons)
        | slick centroid + vertices
        v
Stage 2: drift_service.py      (OpenDrift hindcast/forecast, 4-tier metocean)
        | origin point + uncertainty
        v
Stage 3: attribution_service.py(AIS correlation + SAR visual match)
        | verified evidence dossier JSON
        v
Stage 4: briefing_service.py   (Gemini 2.5 Flash NTRO narrative, optional)
```

## Repository changes

- `config.yaml` — 5-class model contract, metocean `fetch_timeout_sec` / `current_nc` / `wind_nc`, `briefing` (Gemini) section.
- `app/config.py` — `ModelSettings` (5 classes, ship/land indices), `MetoceanSettings.fetch_timeout_sec`, `BriefingSettings`.
- `app/schemas.py` — `PipelineOptions.fetch_live_metocean` / `generate_briefing`, `IntelligenceBriefing`, `SlickDetectionResponse.look_alikes`/`detected_ships`, `CandidateVessel` visual fields, `AttributionResponse.dark_vessel_alerts`.
- `app/services/model_service.py` — DeepLabV3+ support, checkpoint architecture auto-detection, 5-class confidence maps.
- `app/services/detection_service.py` — 5-class tiled inference, per-class vectorization, co-located ship (2 km buffer) detection.
- `app/services/drift_service.py` — 4-tier metocean resolution (request -> netcdf -> environment_json -> config_fallback), `metocean_paths` override.
- `app/services/attribution_service.py` — SAR visual cross-correlation boost, `VISUALLY_CONFIRMED_AT_SLICK` flag, dark-vessel alerts.
- `app/services/pipeline_service.py` — live Copernicus/ERA5 metocean fetch with graceful fallback, briefing stage.
- `app/services/briefing_service.py` — Gemini 2.5 Flash dossier narration (opt-in, non-blocking).
- `app/services/environment_fetcher.py` — Copernicus currents + ERA5 winds, CF-compliant NetCDF, metocean manifest.
- `src/environment_fetcher.py` — replaced with the live-fetch implementation (`fetch_incident_metocean`).
- `src/train.py` — 5-class DeepLabV3+ training (Hybrid CE+Dice loss, per-class IoU).
- `app/api/routes.py` — `POST /briefing` endpoint.
- `app/api/deps.py` — `BriefingService` wired into `AppServices`.
- `tests/test_stage4_upgrade.py` — 15 unit tests for the new stages.
- `docs/` — this documentation.

## Guarantees preserved

- A weak or unavailable stage never blocks the evidence dossier (graceful degradation to `config_fallback` / `status="skipped"`).
- Explicit `engine="opendrift"` fails loudly (502) when NetCDF forcing is unavailable; `engine="auto"` degrades to Lagrangian.
- Gemini is read-only downstream of `attribution_service.py`; it never sees raw GeoTIFFs.

See `docs/SECURITY_VULNERABILITIES.md` for the vulnerability and threat model.
