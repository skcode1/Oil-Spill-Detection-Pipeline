# SIH26143 - Oil Spill Detection & Vessel Attribution

Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data
Correlations to Identify Responsible Vessels (NTRO).

Turns a SAR/optical satellite scene into an evidence dossier: where the slick
is, where the oil came from (backtracing), where it is heading (forecasting),
and which AIS vessels could plausibly have released it - with explainable
scores instead of a single opaque guess.

The service layer (`app/services/`) is intentionally not rebuilt between model
upgrades: Stage 1 segmentation, Stage 2 drift, Stage 3 attribution and Stage 4
briefing all consume the same deterministic dossier contracts, so stronger
open-source weights can be dropped in without touching the API or route code.

## Pipeline stages

| Stage | What it does | Key output |
| --- | --- | --- |
| `detection` | 5-class DeepLabV3+ / ResNet-34 segmentation (`segmentation_models_pytorch`), tiled inference, per-class vectorisation, co-located ship detection | slick polygons, look-alikes, detected ships, area, confidence |
| `drift` (hindcast) | Lagrangian backtracking from the slick (OpenDrift when readable metocean files exist, built-in engine otherwise) | probable origin + time of release |
| `forecast` | Same engine forward in time from the slick | predicted extent, drift bearing/speed |
| `attribution` | AIS window + distance pruning around the drift origin, envelope/delay modelling, weighted explainable scoring, SAR visual cross-correlation, dark-vessel alerting | ranked candidate vessels with a score breakdown and funnel counts |
| `briefing` (optional LLM) | Google Gemini 2.5 Flash synthesises an NTRO intelligence narrative from the verified dossier JSON (read-only downstream of attribution; never blocks the dossier) | executive summary, suspect rationale, risk level, next actions |

Each stage reports its own `status` (`completed` / `skipped` / `failed`) so a
weak stage never hides the rest of the dossier, and the hindcast origin - not
the slick centroid - is what attribution aims at (`origin_source` tells you
which one was used).

### Stage 1 — Detection (5-class SAR segmentation)

The taxonomy (Krestenitis et al. SAR dataset standard):

| Class | Meaning |
| --- | --- |
| 0 | Sea Surface |
| 1 | Oil Spill |
| 2 | Look-Alike |
| 3 | Ship |
| 4 | Land |

- SAR scenes are normalised to a 3-channel composite `[VV, VH, VV/VH ratio]` per tile.
- Sliding-window (512 px, 256 px stride) inference with overlap averaging removes tile seams.
- Class-1 (oil) polygons feed drift & attribution; Class-2 (look-alike) polygons are logged as dossier warnings only, never backtracked; Class-3 (ship) polygons within `ship_buffer_km` (default 2.0 km) of a slick are flagged as co-located visual candidates ("dark vessel" cue).
- `model.confidence_threshold` defaults to `0.90` so clean-water scenes stay free of false slicks.

**Readiness validation:** `detection_service` checks that the loaded checkpoint exposes all 5 classes. If it exposes fewer (e.g. a legacy 3-class U-Net checkpoint), ship/land detection is reported unavailable and the operator is directed to `src/train.py` to train a 5-class checkpoint on the Krestenitis SAR dataset before production use.

### Stage 4 — Briefing (optional LLM synthesis)

`POST /briefing` accepts a verified dossier JSON (detection + drift + attribution) and returns a Gemini-structured NTRO intelligence narrative. It is opt-in (`generate_briefing=true` in pipeline options) and never blocks the dossier: when `GEMINI_API_KEY` is unset or the call fails, the response reports `status="skipped"` and the rest of the pipeline is unaffected.

## Layout

| Path | Role |
| --- | --- |
| `app/main.py` | FastAPI entry point (`create_app` factory, request-id middleware, CORS, lifespan) |
| `app/api/routes.py` | HTTP layer: validated requests in, service calls out |
| `app/api/deps.py` | Service container (`AppServices`), dependency wiring, health report |
| `app/services/` | `detection_service`, `drift_service`, `attribution_service`, `pipeline_service`, `briefing_service`, geo/geometry helpers |
| `app/schemas.py`, `app/errors.py`, `app/config.py` | Pydantic contracts, the error envelope, `config.yaml` loading |
| `src/` | Research/training scripts (`train.py`, `run_pipeline.py`, `opendrift_engine.py`, `environment_fetcher.py`, ...) |
| `backend/main.py` | Earlier minimal API, superseded by `app/main.py` |
| `maritime-surveillance/` | Next.js dashboard that consumes this API |
| `docs/` | Per-stage docs, security & vulnerability model, API reference |

## Install

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt   # includes fastapi, uvicorn, python-multipart
```

Optional live metocean (Copernicus currents + ERA5 winds) and briefing (Gemini):

```bash
pip install copernicusmarine cdsapi xarray netCDF4   # live metocean fetch
pip install google-genai                              # Gemini briefing
copernicusmarine login                                # ~/.copernicusmarine credentials
# ~/.cdsapirc: url: https://copernicus.eu  / key: YOUR_PERSONAL_ACCESS_TOKEN
export GEMINI_API_KEY=...                             # briefing stage
```

`config.yaml` (project root) drives everything: scene/model/AIS paths, model
architecture and checkpoint, tiling, regions, metocean files, drift parameters,
AIS pruning tolerances and the `api:` and `briefing:` blocks (title/version/host/port/cors_origins,
Gemini model/temperature/timeout/API-key env name).

Environment overrides, when a deployment needs them: `OILSPILL_MODEL_PATH`,
`OILSPILL_OUTPUT_DIR`, `OILSPILL_AIS_CSV`, `OILSPILL_AUTHENTIC_TEST_DIR`,
`OILSPILL_DEVICE`, `OILSPILL_TILE_BATCH`, `OILSPILL_PORT`,
`OILSPILL_CORS_ORIGINS`, `OILSPILL_METOCEAN_JSON`, `OILSPILL_METOCEAN_FETCH_TIMEOUT_SEC`,
`OILSPILL_BRIEFING_TIMEOUT_SEC`, `OILSPILL_GEMINI_API_KEY`.

## Run the API

```bash
venv/bin/python -m uvicorn app.main:app --reload
```

* Swagger UI: `http://127.0.0.1:8000/docs`, OpenAPI: `/openapi.json`
* The segmentation checkpoint is warmed during startup (~1 s, device chosen by
  `app.config`); `/api/v1/health` reports `model.ready` and warns while it is
  not loaded.  Pass `load_model=False` to `create_app()` for lazy loading.

### Endpoints (all under `/api/v1`)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | service index + endpoint map (kept out of the OpenAPI schema) |
| `GET` | `/health` | model/device/checkpoint info, path readiness, warnings |
| `GET` | `/regions` | configured search regions and their defaults |
| `POST` | `/pipeline` | full dossier for a scene already on the server |
| `POST` | `/pipeline/upload` | same, with a multipart scene upload plus `options` JSON |
| `POST` | `/drift/hindcast` | backtrace a slick or point to a probable origin |
| `POST` | `/drift/forecast` | forward-drift a slick or point into the future |
| `POST` | `/briefing` | synthesise an NTRO intelligence narrative from a verified dossier JSON |

Pipeline options gained two flags: `fetch_live_metocean` (download live
Copernicus currents + ERA5 winds before drift; degrades gracefully on failure)
and `generate_briefing` (ask Gemini to synthesise an NTRO narrative from the
dossier).

### Examples

```bash
# Full dossier for a scene already on the server (stage summary out)
curl -s -X POST http://127.0.0.1:8000/api/v1/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"path": "data/raw/scene.tif",
       "drift_duration_hours": 24,
       "forecast_duration_hours": 12,
       "ais_csv_path": "data/raw/ais_data.csv",
       "fetch_live_metocean": true,
       "generate_briefing": true}' | jq '.stages, .briefing'

# Upload a scene and skip attribution (no AIS source needed)
curl -s -X POST http://127.0.0.1:8000/api/v1/pipeline/upload \
  -F 'file=@data/raw/scene.tif' \
  -F 'options={"run_attribution": false}' | jq '.detection.slicks | length'

# Backtrace a point to a probable discharge location
curl -s -X POST http://127.0.0.1:8000/api/v1/drift/hindcast \
  -H 'Content-Type: application/json' \
  -d '{"source": {"source_lon": 2.01, "source_lat": 50.01, "source_radius_m": 1500},
       "start_time_utc": "2026-05-01T00:00:00Z",
       "duration_hours": 24,
       "engine": "lagrangian"}' | jq '{origin_lon, origin_lat, drift_bearing_deg}'

# Generate an NTRO briefing from an existing dossier
curl -s -X POST http://127.0.0.1:8000/api/v1/briefing \
  -H 'Content-Type: application/json' \
  -d '{"dossier": {"detection": {...}, "drift": {...}, "attribution": {...}}}' | jq '.executive_summary'
```

`engine` accepts `auto` (OpenDrift when the NetCDF readers are readable, the
built-in Lagrangian engine otherwise), `lagrangian` (deterministic, no external
data) and `opendrift` (fails loudly instead of silently degrading).

Every drift run takes its current and wind vectors from the first source that is
usable, and reports which one it used in `metocean_source`:

1. the `metocean` samples sent in the request (`"request"`);
2. per-request `metocean_paths` override (`current_nc` / `wind_nc`, `"netcdf"`);
3. readable configured `current_nc`/`wind_nc` files, sampled by OpenDrift itself (`"netcdf"`);
4. the per-incident manifest at `metocean.environment_json`, by default
   `data/processed/metocean_environment.json` (`"environment_json"`; override the
   path with `OILSPILL_METOCEAN_JSON`, leave it empty to disable the manifest);
5. the `config.yaml -> metocean` constants (`"config_fallback"`).

Live metocean (`fetch_live_metocean=true`) downloads Copernicus currents and ERA5
winds centered on the primary slick via `src/environment_fetcher.py`, enforces CF
standard names for OpenDrift's `reader_netCDF_CF_generic`, writes
`results/api/{request_id}/metocean/metocean_environment.json`, and degrades to
the config fallback on timeout/failure so the dossier is never blocked.

A manifest is either one uniform vector (`u_current_mps`, `v_current_mps`,
`u_wind_mps`, `v_wind_mps`, optionally stamped `timestamp_utc`/`timestamp_sat`)
or a time series - `samples: [{timestamp_utc, u_current_mps, ...}, ...]` - which
the engine interpolates in time, so tidal or seasonal veering inside the
integration window is actually resolved.  An optional `wind_drift_factor` in the
manifest overrides the configured windage.  A manifest that cannot be read or
parsed never fails a request: the response records why and continues on the
config field, `/health` publishes the resolved sources under `metocean`, and
warnings name any window the samples do not cover.

## Response conventions

* Every response carries `X-Request-Id` (also stamped into the log lines for
  that request) and `X-Process-Time-Ms`.
* Failures always use one envelope, so clients parse errors the same way:
  `{"error": {"code": "...", "message": "...", "details": ..., "request_id": "..."}}`

| Status | `error.code` | Typical cause |
| --- | --- | --- |
| 400 | `invalid_input` / `invalid_image` / `invalid_ais_source` | malformed `options` form field, empty upload, unreadable scene, unparsable AIS file |
| 404 | `artifact_not_found` | scene, AIS or NetCDF path missing on the server |
| 422 | `validation_error` | body violates the schema (`duration_hours` <= 0, attribution asked for with no AIS source, ...) |
| 502 | `drift_engine_failure` | the selected drift engine could not run (explicit `opendrift` with unreadable NetCDF) |
| 503 | `model_unavailable` | segmentation checkpoint cannot be loaded |
| 500 | `internal_error` | unexpected failure, logged with the request id |

**No synthetic fallback:** when no real AIS pings overlap the spatio-temporal
envelope, attribution returns an empty candidate list and `dark_vessel_alerts`
from any co-located SAR ships - it never fabricates a vessel track or score.

## Model checkpoint and inference recipe

The shipped `models/oil_spill_segmentation_model.pth.bak` is the last sane
checkpoint (3-class U-Net); `config.yaml` points at it. `src/train.py` now
trains a 5-class DeepLabV3+ / ResNet-34 on the Krestenitis SAR dataset taxonomy
(Sea / Oil Spill / Look-Alike / Ship / Land) with a hybrid Cross-Entropy +
Dice loss and per-class IoU tracking, selecting the best checkpoint on Oil
Spill IoU.

Fine-tuning recipe (warm start):

```bash
venv/bin/python src/train.py \
  --data-root data/raw/SAR_training \
  --epochs 40 --batch-size 8 --learning-rate 3e-4 \
  --save-to models/sar_5class_deeplabv3p_resnet34.pt
```

**Threshold calibration:** the sweep over the authentic corpus put the
false-slick-free operating point for clean-water scenes at `0.90` (no-oil
scenes: 0 false pixels at 0.90 vs ~85k at 0.50), so `model.confidence_threshold`
and the dashboard slider default are both `0.90`.

**Known model gaps:** the lookalike class is still not fully separated -
lookalike scenes can produce confidently-labelled oil pixels at some thresholds.
`tests/test_scene_pipeline_integration.py` encodes this as an `xfail` so the
gap stays visible. A supervised fix needs better lookalike annotations (or
contrastive/negative mining), not just more epochs.

## Dashboard (`maritime-surveillance/`)

The Next.js dashboard consumes this API. Its map HUD region label and default
center are now driven by incident metadata (scene filename / slick centroid) and
fall back to neutral values ("Open Ocean", `[20.0, 0.0]`) instead of a
hardcoded Persian Gulf assumption, so it works for arbitrary maritime incident
locations worldwide.

## Verifying a checkout

The pytest suite drives the real FastAPI stack through `TestClient` and asserts
on outcomes:

```bash
venv/bin/python -m pytest tests/ -q -k "not native"
```

The API harness stubs only the pipeline service (so no GPU or 100 MB weights
are needed) while still calling the real drift engine, error handlers, multipart
parsing and OpenAPI generation. The pipeline/drift harnesses use synthetic
scenes plus a temporary artifact directory, and `tests/test_stage4_upgrade.py`
covers briefing skip behavior, SAR co-location + visual matching,
per-class vectorisation, metocean field tiers, and manifest assembly without
network access.

The server itself can be spot-checked with:

```bash
curl -s http://127.0.0.1:8000/api/v1/health | jq '.status, .model.ready'
```

## Notes

* `results/` receives detection outputs and `results/api/` the request-scoped
  artifacts (uploaded scenes, seed GeoJSON); `/api/v1/health` verifies both are
  writable.
* Attribution aims at the hindcast origin when the drift stage succeeded, and
  falls back to the slick centroid otherwise - `origin_source` in the response
  says which one was used, so a fallback is never silent.
* OpenDrift stays optional: `engine="auto"` degrades to the built-in engine and
  records a warning, while an explicit `engine="opendrift"` request returns 502
  rather than presenting fallback results as OpenDrift results.
* `metocean.weathering_model: adios` needs the `simple-adios` package.  When it is
  missing, `DriftService._weathering_model` runs OpenOil with weathering disabled
  and says so in the response warnings and in `/health`
  (`metocean.weathering_model_effective`), so a transport-only result is never
  mistaken for full oil-fate modelling.
* The metocean manifest is data, not configuration: a malformed or absent
  `metocean_environment.json` produces a warning and the config field, and never
  rejects drift traffic.  Replace it with real sampled Copernicus/ERA5 vectors
  (`metocean.datasets` plus `src/environment_fetcher.py`) per incident.
