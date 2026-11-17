# Stage 4 — Dossier Synthesis / Briefing (briefing_service.py)

## Role
Google Gemini 2.5 Flash operates strictly downstream of `attribution_service.py` as a
read-only analyst. It converts the **verified, deterministic dossier JSON** (detection +
drift + attribution) into a human-readable NTRO intelligence narrative. It never sees raw
GeoTIFFs and never performs coordinate math.

## Input / output
- Input: sanitized dossier JSON (`IntelligenceBriefing` schema).
- Output: `IntelligenceBriefing` with `executive_summary`, `primary_suspect_rationale`,
  `environmental_risk_level`, `recommended_next_actions`, and `status`
  (`completed` | `skipped`).

## Non-blocking guarantee
The stage is opt-in (`generate_briefing=true`) and **never blocks the evidence dossier**:
- `GEMINI_API_KEY` unset → `status="skipped"`.
- SDK missing / rate-limited / network failure → caught, logged, `status="skipped"`.
- The pipeline response omits/skips the briefing but returns the full dossier.

## Config (`config.yaml -> briefing`)
```
enabled: true
model: gemini-2.5-flash
temperature: 0.1
timeout_sec: 30.0
api_key_env: GEMINI_API_KEY
```

## Endpoint
`POST /api/v1/briefing` — body: `{ "dossier": {...} }` → `IntelligenceBriefing`.
The briefing is also appended to the pipeline response when
`POST /api/v1/pipeline` includes `"generate_briefing": true`.

## Credentials
Set `GEMINI_API_KEY` in the environment before running; the key is read from
`settings.briefing.api_key_env`.
