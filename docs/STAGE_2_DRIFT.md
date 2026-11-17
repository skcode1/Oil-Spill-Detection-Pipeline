# Stage 2 — Drift & Metocean (drift_service.py / environment_fetcher.py)

## Metoacon resolution hierarchy (4 tiers)
Strict priority, most specific first:

1. `request` — inline metocean samples supplied in the API payload.
2. `netcdf` — per-request `metocean_paths` override (`current_nc` + `wind_nc`), else the
   configured `config.yaml` NetCDF paths (OpenDrift forcing).
3. `environment_json` — per-incident JSON manifest (`metocean_paths.environment_json`,
   `OILSPILL_METOCEAN_JSON`, or `config.yaml -> metocean.environment_json`).
4. `config_fallback` — static `config.yaml` current/wind constants (guaranteed).

The active source is reported as `DriftResponse.metocean_source` (`request` | `netcdf` |
`environment_json` | `config_fallback`).

## Live fetch (`src/environment_fetcher.py`, re-exported via `app/services/environment_fetcher.py`)
- `fetch_incident_metocean(lon, lat, detection_time_utc, ...)` downloads:
  - Copernicus Marine surface currents (`uo`, `vo`) via `copernicusmarine.subset`.
  - ERA5 10 m winds (`u10`, `v10`) via `cdsapi`.
- CF standard names are injected (`eastward_sea_water_velocity`, `northward_sea_water_velocity`,
  `x_wind`, `y_wind`) so OpenDrift's `reader_netCDF_CF_generic` accepts the files.
- `build_environment_manifest` samples both grids at the slick centroid and writes
  `data/processed/metocean_environment.json` (time series of u/v current and wind).

## Pipeline wiring (`app/services/pipeline_service.py`)
When `fetch_live_metocean=true`, `resolve_incident_metocean_async` runs the fetch in a thread
with a `fetch_timeout_sec` wall-clock cap. Any timeout/failure degrades gracefully to
`config_fallback` and records a warning — the dossier is never blocked.

## OpenDrift vs Lagrangian
- `engine="opendrift"` requires readable NetCDF; otherwise HTTP 502 (fails loudly).
- `engine="auto"` uses OpenDrift when NetCDF is available, otherwise falls back to the
  built-in Lagrangian engine using the resolved metocean field.

## Credentials (one-time setup)
```
pip install copernicusmarine cdsapi xarray netCDF4
copernicusmarine login          # ~/.copernicusmarine/.copernicusmarine-credentials
# ~/.cdsapirc:
url: https://copernicus.eu
key: YOUR_PERSONAL_ACCESS_TOKEN
```
`copernicusmarine`/`cdsapi` are imported lazily, so the module imports cleanly when they are
absent; live fetch then degrades to the config fallback.
