# SIH26143 - Documentation Index

This directory contains all project documentation. Code files have been stripped of
all comments and docstrings; this index serves as the central reference for the
codebase structure, architecture, and implementation details.

## Documentation Files

| File | Description |
|------|-------------|
| `README_ROOT.md` | Original project README (backend/API) |
| `README_FRONTEND.md` | Next.js dashboard README |
| `ARCHITECTURE_UPGRADE.md` | Hybrid neuro-symbolic architecture (5 stages) |
| `API_REFERENCE.md` | FastAPI endpoint reference |
| `SECURITY_VULNERABILITIES.md` | Security model and threat analysis |
| `STAGE_1_DETECTION.md` | Stage 1: 5-class SAR detection |
| `STAGE_2_DRIFT.md` | Stage 2: Drift and metocean |
| `STAGE_3_ATTRIBUTION.md` | Stage 3: Vessel attribution |
| `STAGE_4_BRIEFING.md` | Stage 4: LLM briefing |
| `IMPLEMENTATION_SUMMARY.md` | Summary of all implementation changes |
| `LOOKALIKE_FILTERING_GUIDE.md` | Look-alike suppression methodology |
| `LOOKALIKE_FILTERING_QUICKSTART.md` | Quick start for look-alike filtering |
| `TESTING_GUIDE.md` | Test suite guide |
| `AGENTS_FRONTEND.md` | Frontend agent instructions |
| `CLAUDE_FRONTEND.md` | Claude frontend instructions |

## Codebase Structure

```
SIH26143/
├── app/                          # FastAPI service layer
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory, lifespan, health
│   ├── config.py                 # Configuration (Pydantic settings from config.yaml)
│   ├── errors.py                 # Error envelope and exception classes
│   ├── logging_config.py         # Logging configuration
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py               # Service container, dependency injection
│   │   └── routes.py             # HTTP route handlers
│   └── services/
│       ├── __init__.py
│       ├── age_estimator.py      # Spill age estimation
│       ├── attribution_service.py# AIS vessel attribution with SAR cross-correlation
│       ├── briefing_service.py   # Stage 4: Gemini LLM briefing
│       ├── detection_service.py  # Stage 1: 5-class SAR segmentation
│       ├── drift_service.py      # Stage 2: Drift hindcast/forecast
│       ├── geo.py                # Geospatial utilities
│       ├── image_io.py           # SAR image loading and preprocessing
│       ├── incident_store.py     # Incident persistence
│       ├── lookalike_filter.py   # Look-alike suppression heuristics
│       ├── model_service.py      # Segmentation model wrapper
│       └── pipeline_service.py   # Pipeline orchestration
├── src/                          # Research/training scripts
│   ├── ais_pruner.py             # AIS spatio-temporal pruning
│   ├── app.py                    # Legacy API entry
│   ├── attribution_engine.py     # Vendor attribution scorer
│   ├── cross_match_ais.py        # AIS cross-matching
│   ├── dataset.py                # SAR dataset handling
│   ├── detect_and_vectorize.py   # Detection + polygon vectorization
│   ├── environment_fetcher.py    # Copernicus/ERA5 metocean fetcher
│   ├── metadata_extractor.py     # GeoTIFF metadata extraction
│   ├── opendrift_engine.py       # OpenDrift physics engine
│   ├── preprocess.py             # SAR preprocessing
│   ├── region_registry.py        # Region definitions
│   ├── run_pipeline.py           # Batch pipeline runner
│   ├── simulation_generator.py   # Synthetic data generator
│   ├── split_and_patch.py        # Image tiling utilities
│   └── train.py                  # 5-class DeepLabV3+ training
├── tests/                        # Pytest test suite
│   ├── conftest.py               # Shared fixtures
│   ├── test_api_integration.py   # API integration tests
│   ├── test_config_and_contracts.py
│   ├── test_dashboard_api_contract.py
│   ├── test_detection_authentic.py
│   ├── test_drift_and_pipeline.py
│   ├── test_model_sanity.py
│   ├── test_production_readiness.py
│   ├── test_scene_pipeline_integration.py
│   ├── test_schemas_and_errors.py
│   └── test_stage4_upgrade.py
├── maritime-surveillance/        # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx              # Main dashboard page
│   │   ├── layout.tsx            # Root layout
│   │   └── alerts/page.tsx       # Alerts page
│   └── src/
│       └── SurveillanceMap.tsx   # Leaflet map component
├── config.yaml                   # Deployment configuration
└── docs/                         # This directory
```

## Architecture: 5 Hybrid Stages

| Stage | Component | Responsibility |
|-------|-----------|----------------|
| 1 | `detection_service.py` | 5-class DeepLabV3+ SAR segmentation (Sea/Oil/Look-Alike/Ship/Land) |
| 2 | `drift_service.py` | Lagrangian drift hindcast/forecast (OpenDrift or built-in) |
| 3 | `attribution_service.py` | AIS vessel attribution with SAR visual cross-correlation |
| 4 | `briefing_service.py` | Gemini 2.5 Flash NTRO intelligence narrative (optional LLM) |
| Orchestrator | `pipeline_service.py` | Chains stages 1-4 with graceful degradation |

## Key Design Guarantees

1. **No synthetic AIS tracks**: When no real AIS pings intersect the drift envelope,
   attribution returns `candidates: []` with a `zero_ais_candidates_in_envelope`
   warning. Fabricating culprit vessels is hard-blocked via
   `attribution.allow_synthetic_ais: false` in `config.yaml` (enforced at
   FastAPI startup with `RuntimeError`).

2. **Graceful degradation**: Weak or unavailable stages never block the evidence
   dossier. Live metocean fetch degrades to `config_fallback`; Gemini briefing
   reports `status: skipped` when unavailable.

3. **Deterministic core**: Segmentation, coordinate transforms, Lagrangian
   backtracking, and AIS pruning remain deterministic and reproducible.

## Configuration

`config.yaml` controls all deployment parameters:
- `model`: Segmentation architecture, classes, confidence threshold
- `inference`: Tile size, stride, area thresholds
- `drift`: Physics parameters, fallback values
- `attribution`: Weights, SOG band, synthetic AIS toggle
- `metocean`: Wind drift factor, weathering model, file paths
- `briefing`: Gemini model, temperature, timeout, API key env
- `api`: CORS, upload limits, host/port

## Running the API

```bash
python -m uvicorn app.main:app --reload
```

Endpoints (all under `/api/v1`):
- `GET /health` - Model/checkpoint readiness
- `GET /regions` - Configured search regions
- `POST /pipeline` - Full dossier for server-side scene
- `POST /pipeline/upload` - Full dossier with multipart upload
- `POST /drift/hindcast` - Backtrace slick to probable origin
- `POST /drift/forecast` - Forward drift prediction
- `POST /briefing` - Gemini NTRO intelligence narrative

## Running Tests

```bash
python -m pytest tests/ -q -k "not native"
```

## Changes Made (Comment Removal)

All comments and docstrings have been removed from the following source files
and consolidated into this `docs/` directory:

**Backend (50 Python files):**
- `app/` - All modules (config, schemas, services, API, errors, logging)
- `src/` - All scripts (training, detection, drift, AIS, preprocessing)
- `tests/` - All test files
- Root scripts: `run_benchmark_grid.py`, `scratch_eval_checkpoint.py`, `stretched.py`, `visualization.py`

**Frontend (3 TSX files):**
- `maritime-surveillance/app/page.tsx`
- `maritime-surveillance/app/alerts/page.tsx`
- `maritime-surveillance/src/SurveillanceMap.tsx`

All inline comments, block comments, and docstrings were removed from these
files. The explanatory content has been preserved in the `docs/` directory
above.
