"""LLM synthesis layer: Google Gemini 2.5 Flash dossier narration.

Stage 4 of the hybrid architecture.  Gemini operates strictly downstream of
``attribution_service.py`` as a read-only analyst that translates the
verified, deterministic dossier JSON into a human-readable NTRO intelligence
narrative.  It never sees raw GeoTIFFs or performs coordinate math.

The stage is **opt-in and non-blocking**: when ``GEMINI_API_KEY`` is unset,
the model is disabled, or the call fails/rate-limits, the dossier is returned
with ``status="skipped"`` and the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from app.config import AppSettings
from app.schemas import IntelligenceBriefing

logger = logging.getLogger(__name__)


class BriefingService:
    """Gemini-powered NTRO briefing generator.

    Args:
        settings: Resolved application settings (briefing section + API key env name).
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._client = None
        self._enabled = bool(settings.briefing.enabled)
        self._api_key = os.environ.get(settings.briefing.api_key_env) if self._enabled else None
        if self._enabled and not self._api_key:
            logger.info(
                "Briefing disabled: %s environment variable not set.",
                settings.briefing.api_key_env,
            )
            self._enabled = False

    @property
    def available(self) -> bool:
        """True when the Gemini client is configured and the stage is enabled."""
        return self._enabled and self._client is not None

    def _ensure_client(self) -> Optional[Any]:
        """Lazily construct the genai client; never raises."""
        if self._client is not None:
            return self._client
        if not self._enabled or not self._api_key:
            return None
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except Exception as exc:  # noqa: BLE001 - SDK may not be installed
            logger.warning("Google Genai SDK unavailable: %s", exc)
            self._enabled = False
            return None

    def generate_briefing(self, dossier: Dict[str, Any]) -> IntelligenceBriefing:
        """Synthesise an NTRO intelligence narrative from a verified dossier.

        Returns an :class:`IntelligenceBriefing` with ``status="skipped"`` when
        Gemini is unavailable or the call fails, so the evidence dossier is
        never blocked by the LLM stage.
        """
        brief_cfg = self.settings.briefing
        if not self.available:
            return IntelligenceBriefing(
                executive_summary="Briefing skipped: Gemini LLM stage unavailable.",
                primary_suspect_rationale="",
                environmental_risk_level="unknown",
                recommended_next_actions=[],
                status="skipped",
            )

        client = self._ensure_client()
        if client is None:
            return IntelligenceBriefing(
                executive_summary="Briefing skipped: Gemini client could not be initialised.",
                primary_suspect_rationale="",
                environmental_risk_level="unknown",
                recommended_next_actions=[],
                status="skipped",
            )

        # Sanitise: pass only the deterministic dossier JSON, never raw imagery.
        safe_dossier = json.dumps(dossier, default=str, indent=2)[:12000]
        prompt = (
            "You are a maritime intelligence analyst for NTRO. Analyze this verified "
            "oil-spill detection and AIS vessel attribution dossier. Do not invent "
            "coordinates, scores, or vessel identities not present in the JSON.\n\n"
            "DOSSIER:\n"
            f"{safe_dossier}"
        )
        try:
            response = client.models.generate_content(
                model=brief_cfg.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": IntelligenceBriefing,
                    "temperature": brief_cfg.temperature,
                },
            )
            parsed = response.parsed
            if parsed is None:
                raise ValueError("Gemini returned no parsed content")
            parsed.status = "completed"
            return parsed
        except Exception as exc:  # noqa: BLE001 - rate limits, network, quota
            logger.warning("Gemini briefing failed: %s", exc)
            return IntelligenceBriefing(
                executive_summary=f"Briefing skipped: LLM stage unavailable ({type(exc).__name__}).",
                primary_suspect_rationale="",
                environmental_risk_level="unknown",
                recommended_next_actions=[],
                status="skipped",
            )
