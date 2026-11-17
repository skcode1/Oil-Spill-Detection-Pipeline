# Stage 3 — Vessel Attribution (attribution_service.py)

## What changed
The AIS correlation pipeline keeps its explainable weighted scoring and adds two
capabilities driven by the Stage-1 5-class SAR detections:

### 1. SAR visual confirmation boost
For every scored AIS vessel, its position at/near satellite acquisition time is compared
against the co-located SAR ship polygons (`detection.slicks[].colocated_ships`). When a
vessel falls within `sar_match_radius_km` (default 0.75 km) of a SAR ship and its ping is
within `SAR_ACQ_WINDOW_MIN` (30 min) of acquisition:

- `visual_sar_boost_pts` (default up to 25 pts × SAR confidence) is added to the score.
- Evidence flag `VISUALLY_CONFIRMED_AT_SLICK` is set.
- `sar_visual_match` records the matched SAR detection (offset, slick distance, confidence).

### 2. Dark-vessel alerting
Any SAR-detected ship near a slick (within `ship_buffer_km`) that is **not** matched by an
AIS transponder within the time window is emitted as a `dark_vessel_alerts` entry, giving
NTRO a hard visual cue for an AIS-off ("dark") vessel at the spill site.

## Output (`AttributionResponse`)
- `candidates[].visually_confirmed`, `evidence_flags`, `sar_visual_match`, `score_breakdown`
  (incl. `visual_sar_boost_pts`).
- `dark_vessel_alerts` — unmatched SAR ships near slicks.
- `funnel.unmatched_dark_vessels` count.

## Config (`config.yaml` / `settings.attribution`)
```
sar_match_radius_km: 0.75   # AIS-to-SAR visual match radius (km)
visual_boost_points: 25.0   # max score boost for visual confirmation
```

## Contract preserved
`AttributionService.attribute(AttributionRequest)` keeps its signature; `AttributionRequest`
gains an optional `detection_result` dict carrying `slicks[].colocated_ships`.
