# Security, Vulnerabilities & Threat Model

## Design principles
1. **Deterministic core** — segmentation, coordinate transforms, Lagrangian backtracking
   and AIS pruning remain reproducible; no LLM weights influence evidence generation.
2. **LLM isolation** — Gemini receives only sanitized dossier JSON; no raw imagery,
   coordinates math, or secrets are sent to the model.
3. **Never block** — external services (Copernicus, ERA5, Gemini) are behind timeouts
   and degrade gracefully; the dossier is always produced.

## Identified risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Live metocean fetch (Copernicus/ERA5) hangs or leaks credentials | High | `fetch_timeout_sec` wall-clock cap; credentials in `~/.copernicusmarine` / `~/.cdsapirc` (never in code); lazy imports so missing packages degrade to `config_fallback`. |
| Gemini API key exposure | High | Read from `GEMINI_API_KEY` env only; never logged or returned; key never embedded in prompts/responses. |
| Prompt injection via dossier | Medium | Dossier is deterministic JSON produced by the pipeline; model instructed not to invent coordinates/scores. Output schema-enforced via `response_schema`. |
| Malformed metocean manifest | Medium | Manifest parse failures are caught and degrade to `config_fallback` with a warning; never 5xx the request. |
| Explicit `engine=opendrift` with missing NetCDF | Medium | Fails loudly with HTTP 502 instead of silently falling back (configurable via `engine=auto` for graceful degradation). |
| Large SAR uploads | Medium | `max_upload_mb` cap (config `api.max_upload_mb` / `OILSPILL_MAX_UPLOAD_BYTES`); oversized uploads rejected before inference. |
| Model checkpoint mismatch | Medium | `model_service` auto-detects checkpoint architecture (U-Net vs DeepLabV3+) and infers class count from the segmentation head, so a 3-class U-Net checkpoint still loads correctly. |
| In-memory upload georeferencing lost | Low | `load_scene_from_bytes` now preserves embedded GeoTIFF transform/CRS via `rasterio.MemoryFile` (best-effort); falls back to pixel space. |

## Operational notes
- Run `copernicusmarine login` and configure `~/.cdsapirc` before enabling live metocean fetch.
- Set `GEMINI_API_KEY` to enable the briefing stage; omit it to run fully deterministic.
- `OILSPILL_METOCEAN_JSON` overrides the manifest path; set empty to disable manifest use.
