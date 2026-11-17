# API Reference — new / changed endpoints and fields

## POST /api/v1/pipeline (and /pipeline/upload)
Body options (`PipelineOptions`) gained two optional booleans:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `fetch_live_metocean` | bool | `false` | Download live Copernicus currents + ERA5 winds before drift; degrades gracefully on failure. |
| `generate_briefing` | bool | `false` | Ask Gemini to synthesize an NTRO intelligence narrative from the dossier. |

`PipelineResponse` gained an optional `briefing` field (`IntelligenceBriefing`) populated
only when `generate_briefing=true`. The `stages` array includes `"briefing"` only when
requested (default dossier keeps the original four stages).

Detection response (`SlickDetectionResponse`) now includes:
- `look_alikes` — Class-2 polygons (warnings only, no drift).
- `detected_ships` — Class-3 SAR ship detections.
- Each `SlickGeometry` has `visual_ship_detected` and `colocated_ships`.

Attribution response (`AttributionResponse`) now includes:
- `dark_vessel_alerts` — SAR ships near slicks with no AIS match.
- Each `CandidateVessel` has `visually_confirmed`, `evidence_flags`, `sar_visual_match`.

Drift response (`DriftResponse`) reports `metocean_source` and `engine_used`.

## POST /api/v1/briefing (new)
Body: `{ "dossier": { ... } }` where `dossier` is the verified detection/drift/attribution JSON.
Returns `IntelligenceBriefing` (schema-enforced). When Gemini is unavailable the response
reports `status="skipped"`; the endpoint never 5xxs on LLM failure.

## Example dossier excerpt (attribution)
```json
{
  "candidates": [{
    "mmsi": 419001234,
    "visually_confirmed": true,
    "evidence_flags": ["VISUALLY_CONFIRMED_AT_SLICK", "DIRECT_ORIGIN_OVERPASS"],
    "score_breakdown": {
      "visual_sar_boost_pts": 23.35
    },
    "sar_visual_match": {
      "ship_detection_id": "sar_ship_1",
      "sar_to_ais_offset_km": 0.142,
      "sar_ship_confidence": 0.934
    }
  }],
  "dark_vessel_alerts": []
}
```

## Environment variables
| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini briefing (read via `settings.briefing.api_key_env`). |
| `OILSPILL_METOCEAN_JSON` | Override metocean manifest path. |
| `OILSPILL_MAX_UPLOAD_BYTES` | Upload size cap in bytes. |
| `OILSPILL_MAX_STEPS` / `OILSPILL_MAX_ENDPOINTS` | Drift integration/payload caps. |
| `OILSPILL_TEST_DOWNSCALE` / `OILSPILL_TEST_SCENES_PER_CLASS` | Test-suite speed controls. |

## Tests
`tests/test_stage4_upgrade.py` — 15 unit tests covering briefing skip, SAR co-location +
visual matching, per-class vectorization, metocean field tiers, and manifest assembly.
