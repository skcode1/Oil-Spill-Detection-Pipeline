# SIH26143 - Oil Spill Detection & Vessel Attribution

Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data
Correlations to Identify Responsible Vessels (NTRO).

Turns a SAR/optical satellite scene into an evidence dossier: where the slick
is, where the oil came from (backtracing), where it is heading (forecasting),
and which AIS vessels could plausibly have released it - with explainable
scores instead of a single opaque guess.

## Pipeline stages

| Stage | What it does | Key output |
| --- | --- | --- |
| `detection` | U-Net (ResNet-34 encoder) segmentation, tiled inference, vectorised slick polygons, quality checks | slick polygons, area, confidence |
| `drift` (hindcast) | Lagrangian backtracking from the slick (OpenDrift when readable metocean files exist, built-in engine otherwise) | probable origin + time of release |
| `forecast` | Same engine forward in time from the slick | predicted extent, drift bearing/speed |
| `attribution` | AIS window + distance pruning around the drift origin, envelope/delay modelling, weighted explainable scoring | ranked candidate vessels with a score breakdown and funnel counts |

Each stage reports its own `status` (`completed` / `skipped` / `failed`) so a
weak stage never hides the rest of the dossier, and the hindcast origin - not
the slick centroid - is what attribution aims at (`origin_source` tells you
which one was used).

## Layout

| Path | Role |
| --- | --- |
| `app/main.py` | FastAPI entry point (`create_app` factory, request-id middleware, CORS, lifespan) |
| `app/api/routes.py` | HTTP layer: validated requests in, service calls out |
| `app/api/deps.py` | Service container (`AppServices`), dependency wiring, health report |
| `app/services/` | `detection_service`, `drift_service`, `attribution_service`, `pipeline_service`, geo/geometry helpers |
| `app/schemas.py`, `app/errors.py`, `app/config.py` | Pydantic contracts, the error envelope, `config.yaml` loading |
| `src/` | Research/training scripts (`train.py`, `run_pipeline.py`, `opendrift_engine.py`, ...) |
| `backend/main.py` | Earlier minimal API, superseded by `app/main.py` |
| `maritime-surveillance/` | Next.js dashboard that consumes this API |

## Install

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt   # includes fastapi, uvicorn, python-multipart
```

`config.yaml` (project root) drives everything: scene/model/AIS paths, model
architecture and checkpoint, tiling, regions, metocean files, drift parameters,
AIS pruning tolerances and the `api:` block (title/version/host/port/cors_origins).

Environment overrides, when a deployment needs them: `OILSPILL_MODEL_PATH`,
`OILSPILL_OUTPUT_DIR`, `OILSPILL_AIS_CSV`, `OILSPILL_AUTHENTIC_TEST_DIR`,
`OILSPILL_DEVICE`, `OILSPILL_TILE_BATCH`, `OILSPILL_PORT`,
`OILSPILL_CORS_ORIGINS`.

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

### Examples

```bash
# Full dossier for a scene already on the server (stage summary out)
curl -s -X POST http://127.0.0.1:8000/api/v1/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"path": "data/raw/scene.tif",
       "drift_duration_hours": 24,
       "forecast_duration_hours": 12,
       "ais_csv_path": "data/raw/ais_data.csv"}' | jq '.stages'

# Upload a scene and skip the attribution stage (no AIS source needed)
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
```

`engine` accepts `auto` (OpenDrift when the NetCDF readers are readable, the
built-in Lagrangian engine otherwise), `lagrangian` (deterministic, no external
data) and `opendrift` (fails loudly instead of silently degrading).

Every drift run takes its current and wind vectors from the first source that is
usable, and reports which one it used in `metocean_source`:

1. the `metocean` samples sent in the request (`"request"`);
2. readable `current_nc`/`wind_nc` files, sampled by OpenDrift itself (`"netcdf"`);
3. the per-incident manifest at `metocean.environment_json`, by default
   `data/processed/metocean_environment.json` (`"environment_json"`; override the
   path with `OILSPILL_METOCEAN_JSON`, leave it empty to disable the manifest);
4. the `config.yaml -> metocean` constants (`"config_fallback"`).

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
| 502 | `drift_engine_failure` | the selected drift engine could not run |
| 503 | `model_unavailable` | segmentation checkpoint cannot be loaded |
| 500 | `internal_error` | unexpected failure, logged with the request id |

## Model checkpoint and inference recipe

Three stacked defects made the dashboard report **Clear** for every scene:

1. **Degenerate checkpoint.** The shipped
   `models/oil_spill_segmentation_model.pth` produced input-independent,
   exploding logits (softmax outputs differed by ~2e-5 between a real scene
   and random noise), so after morphological cleanup no slick survived.
   `models/oil_spill_segmentation_model.pth.bak` is the last sane checkpoint
   and `config.yaml` now points at it; `src/train.py` refuses to save a
   candidate whose output is input-independent or whose logits explode
   (`_checkpoint_sanity`).
2. **Frontend/backend contract drift.** The dashboard read legacy fields
   (`has_spill`, `geo_stats`) that `PipelineResponse` never contained, so the
   verdict was always falsy. It now derives the verdict from
   `detection.slicks.length` and maps `detection.spill_geojson` /
   `detection.statistics` directly.
3. **Georeferencing crash.** `rasterio.transform.xy` received the raw
   `ds.transform` sequence instead of an `Affine`, aborting every georeferenced
   scene. `as_affine()` in `app/services/detection_service.py` coerces the
   sequence.

Inference must reproduce the **per-tile min-max normalisation** the training
chips used (`src/dataset.py`): whole-scene normalisation shifted the input
distribution and cost roughly half the oil IoU. Both `ModelService` and
`src/detect_and_vectorize.py` now normalise inside the sliding window, and
`config.yaml` runs the window at `tile_size: 512 / stride: 256` to match the
512-px chip footprint.

**Threshold calibration:** the sweep over the authentic corpus put the
false-slick-free operating point for clean-water scenes at `0.90` (no-oil
scenes: 0 false pixels at 0.90 vs ~85k at 0.50), so `model.confidence_threshold`
and the dashboard slider default are both `0.90`.

**Known model gaps:** the lookalike class is still not separated - lookalike
scenes keep producing confidently-labelled oil pixels at every threshold tested
(0.50-0.95). `tests/test_scene_pipeline_integration.py` encodes this as an
`xfail` so the gap stays visible. The training chips mark whole lookalike scenes
as class 2 without pixel-level labels, so a supervised fix needs better
lookalike annotations (or contrastive/negative mining), not just more epochs.

**Fine-tuning recipe** (warm start, ~20 min/epoch on 4 CPU cores at
256×256 / batch 16):

```bash
venv/bin/python src/train.py \
  --init-from models/oil_spill_segmentation_model.pth.bak \
  --save-to models/oil_spill_segmentation_model_v2.pth \
  --epochs 3 --learning-rate 1e-4
```

## Verifying a checkout

There is no pytest suite; these harnesses boot the real code - the API one
drives the full FastAPI stack through `TestClient` - and assert on outcomes:

```bash
venv/bin/python scratch_api_smoke.py             # API SMOKE OK - 7 scenarios
venv/bin/python scratch_pipeline_smoke.py        # PIPELINE SMOKE OK - 9 scenarios
venv/bin/python scratch_drift_opendrift_smoke.py # OPENDRIFT DRIFT SMOKE OK - 5 scenarios
venv/bin/python scratch_metocean_smoke.py        # manifest -> field -> engine precedence
venv/bin/python scratch_attr_smoke.py            # AIS pruning + explainable scoring
```

The API harness keeps itself checkpoint-free by stubbing only the pipeline
service (so no GPU or 100 MB weights are needed) while still calling the real
drift engine, error handlers, multipart parsing and OpenAPI generation.  The
pipeline and drift harnesses use synthetic scenes plus a temporary artifact
directory, and the server itself can be spot-checked with:

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

