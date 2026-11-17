"""AIS traffic pruning and multi-factor vessel attribution service.

The heavy lifting is delegated to the mature, unit-tested ``src/`` modules:
- ``src/ais_pruner.py::prune_ais_stream``: Three-stage chunked pruning.
- ``src/attribution_engine.py::MultiFactorAttributionScorer``: Multi-factor scoring.

This service handles incident geometry resolution, uncertainty envelope building,
funnel summaries, artifact export, and resilient candidate track alignment
so suspect vessels and trajectories render reliably even with disconnected test scenes.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from shapely.geometry import mapping

from app.config import AppSettings
from app.errors import AisSourceError, AttributionError, InvalidInputError, SceneNotFoundError
from app.schemas import (
    AttributionRequest,
    AttributionResponse,
    CandidateVessel,
    PruneRequest,
    PruneResponse,
    TrafficFilterSummary,
    UncertaintyEllipse,
    utc_iso,
)
from app.services.drift_service import parse_time
from app.services.geo import (
    ellipse_polygon_lonlat,
    feature_collection,
    normalize_ellipse,
    polygon_centroid_lonlat,
    polygon_from_geojson,
)

logger = logging.getLogger(__name__)

#: Ranked pings echoed back in a :class:`PruneResponse`.
MAX_PRUNED_RECORDS = 5_000

#: Score bands on the 0-100 attribution scale.
HIGH_CONFIDENCE_SCORE = 60.0
MEDIUM_CONFIDENCE_SCORE = 35.0

#: Factor names understood by the scorer.
KNOWN_FACTORS = (
    "proximity",
    "temporal",
    "trajectory",
    "kinematics",
    "ais_gaps",
    "vessel_type",
)

#: Board column carrying each factor score.
FACTOR_COLUMNS = {
    "proximity": "proximity_score",
    "temporal": "temporal_score",
    "trajectory": "trajectory_score",
    "kinematics": "kinematic_score",
    "ais_gaps": "ais_gap_score",
    "vessel_type": "type_risk_score",
}

#: Canonical factor -> human-readable phrase.
FACTOR_PHRASES = {
    "proximity": "passed close to the inferred release point",
    "temporal": "was on station at the inferred discharge time",
    "trajectory": "track aligns with the slick drift axis",
    "kinematics": "showed discharge-consistent manoeuvring",
    "ais_gaps": "has a transponder blackout across the event",
    "vessel_type": "is of a high-risk vessel class",
}


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between two WGS84 coordinates."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


#: Radius (km) within which an AIS position is considered to match a SAR ship.
SAR_MATCH_RADIUS_KM = 0.75
#: Maximum minutes between an AIS ping and SAR acquisition for a visual match.
SAR_ACQ_WINDOW_MIN = 30.0
#: Score boost (0-100 points) for a visually confirmed vessel.
VISUAL_BOOST_POINTS = 25.0


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _collect_colocated_sar_ships(detection_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten unique co-located ship detections across all slicks in the scene."""
    if not detection_result:
        return []
    seen_ids = set()
    colocated: List[Dict[str, Any]] = []
    for slick in detection_result.get("slicks", []) or []:
        for ship in slick.get("colocated_ships", []) or []:
            sid = ship.get("ship_detection_id")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                colocated.append({**ship, "associated_slick_id": slick.get("slick_id")})
    return colocated


def load_pruner() -> Tuple[Any, ...]:
    """Import the vendored AIS pruner."""
    try:
        from src.ais_pruner import (
            AISPrunerConfig,
            AISPrunerError,
            prune_ais_stream,
            resolve_ais_schema,
            sniff_delimiter,
        )
    except Exception as exc:
        raise AttributionError(f"AIS pruner is unavailable: {exc}") from exc
    return (
        prune_ais_stream,
        AISPrunerConfig,
        AISPrunerError,
        resolve_ais_schema,
        sniff_delimiter,
    )


def load_scorer() -> Tuple[Any, ...]:
    """Import the vendored attribution scorer."""
    try:
        from src.attribution_engine import (
            AttributionConfig,
            MultiFactorAttributionScorer,
        )
    except Exception as exc:
        raise AttributionError(f"Attribution engine is unavailable: {exc}") from exc
    return AttributionConfig, MultiFactorAttributionScorer


@dataclass
class IncidentGeometry:
    """Resolved attribution target: where, when and with what uncertainty."""

    origin_lon: float
    origin_lat: float
    t0_utc: datetime
    semi_major_m: float
    semi_minor_m: float
    orientation_deg: float
    confidence_level: float
    window_hours: float
    warnings: List[str] = field(default_factory=list)

    @property
    def window_start(self) -> datetime:
        return self.t0_utc - timedelta(hours=self.window_hours)

    @property
    def window_end(self) -> datetime:
        return self.t0_utc + timedelta(hours=self.window_hours)

    def ellipse_model(self) -> UncertaintyEllipse:
        return UncertaintyEllipse(**normalize_ellipse(
            self.semi_major_m,
            self.semi_minor_m,
            self.orientation_deg,
            self.confidence_level,
            self.origin_lon,
            self.origin_lat,
        ))


def _read_header(path: Path, sniff) -> Tuple[List[str], str]:
    try:
        delimiter = sniff(str(path))
    except Exception:
        delimiter = ","
    return (
        list(
            pd.read_csv(
                path,
                nrows=0,
                sep=delimiter,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="skip",
            ).columns
        ),
        delimiter,
    )


def traffic_funnel(
    path: Path,
    geometry: IncidentGeometry,
    *,
    chunksize: int,
    resolve_schema,
    sniff,
    source_label: str,
) -> TrafficFilterSummary:
    scanned = 0
    in_window = 0
    vessels: set = set()
    try:
        header, delimiter = _read_header(path, sniff)
        schema = resolve_schema(header)
        reader = pd.read_csv(
            path,
            sep=delimiter,
            usecols=list(schema.values()),
            chunksize=chunksize,
            dtype=str,
            on_bad_lines="skip",
        )
        rename = {column: canonical for canonical, column in schema.items()}
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            scanned += int(len(chunk))
            stamps = pd.to_datetime(chunk.get("timestamp"), utc=True, errors="coerce")
            keep = stamps.between(
                pd.Timestamp(geometry.window_start), pd.Timestamp(geometry.window_end)
            )
            in_window += int(keep.sum())
            if bool(keep.any()) and "mmsi" in chunk.columns:
                vessels.update(
                    chunk.loc[keep, "mmsi"].dropna().astype(str).tolist()
                )
    except Exception as exc:
        logger.warning("Traffic funnel scan failed for %s: %s", path, exc)
        return TrafficFilterSummary(
            source=source_label,
            temporal_window_hours=round(geometry.window_hours, 4),
            window_start_utc=utc_iso(geometry.window_start),
            window_end_utc=utc_iso(geometry.window_end),
            reconstruction_available=False,
        )

    return TrafficFilterSummary(
        source=source_label,
        temporal_window_hours=round(geometry.window_hours, 4),
        window_start_utc=utc_iso(geometry.window_start),
        window_end_utc=utc_iso(geometry.window_end),
        total_records_scanned=scanned,
        records_in_time_window=in_window,
        unique_vessels_in_time_window=len(vessels),
    )


@dataclass
class TrafficBundle:
    """Pruned AIS traffic together with its geometry and funnel summary."""

    geometry: IncidentGeometry
    source_label: str
    frame: pd.DataFrame
    summary: Optional[TrafficFilterSummary] = None


class AttributionService:
    """Prune AIS traffic against the drift envelope and rank suspect vessels."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def prune(self, request: PruneRequest) -> PruneResponse:
        geometry = self._geometry(
            origin_lon=request.origin_lon,
            origin_lat=request.origin_lat,
            t0_utc=request.t0_utc,
            slick_geojson=None,
            uncertainty=request.uncertainty,
            temporal_window_hours=request.temporal_window_hours,
        )
        bundle = self._prune_stage(
            geometry,
            request.ais_csv_path,
            request.ais_records,
            request.chunksize,
            include_funnel=request.include_traffic_reconstruction,
        )
        frame = bundle.frame
        mmsi = [] if frame.empty else sorted({int(m) for m in frame["mmsi"]})
        records = [] if frame.empty else _jsonify(frame.head(MAX_PRUNED_RECORDS).to_dict("records"))
        return PruneResponse(
            origin_lon=geometry.origin_lon,
            origin_lat=geometry.origin_lat,
            t0_utc=utc_iso(geometry.t0_utc),
            ellipse=geometry.ellipse_model(),
            traffic_filter=bundle.summary,
            pruned_ping_count=int(len(frame)),
            candidate_vessel_count=len(mmsi),
            candidate_mmsi=[int(m) for m in mmsi],
            pruned_records=records,
            artifact_paths={},
        )

    def attribute(self, request: AttributionRequest) -> AttributionResponse:
        """Prune the traffic, score every vessel and return the leaderboard.

        Adds SAR visual cross-correlation (visual confirmation boost) and dark
        vessel alerting on top of the multi-factor scored candidates.
        """
        geometry = self._geometry(
            origin_lon=request.origin_lon,
            origin_lat=request.origin_lat,
            t0_utc=request.t0_utc,
            slick_geojson=request.slick_geojson,
            uncertainty=request.uncertainty,
            temporal_window_hours=request.temporal_window_hours,
        )
        for note in geometry.warnings:
            logger.warning("Attribution geometry: %s", note)

        colocated_ships = _collect_colocated_sar_ships(request.detection_result)
        acq_dt = _parse_ts(request.t0_utc) or geometry.t0_utc

        bundle = self._prune_stage(
            geometry,
            request.ais_csv_path,
            request.ais_records,
            request.chunksize,
            include_funnel=request.include_traffic_reconstruction,
        )
        # Empty-result contract: never fabricate culprit vessels.  When no real
        # AIS pings intersect the spatio-temporal envelope, return candidates []
        # with an explicit warning so the dossier is auditable.
        if bundle.frame is None or bundle.frame.empty:
            geometry.warnings.append(
                "zero_ais_candidates_in_envelope: no real AIS pings intersect the "
                "drift-origin spatio-temporal envelope; attribution returns no "
                "candidates (dark-vessel alerts may still apply from SAR)."
            )

        artifact_dir = self._artifact_dir(geometry)
        weights, board = self._rank(
            bundle, request.weights, request.drift_bearing_deg, artifact_dir
        )
        candidates = self._candidates(
            board, weights, bundle.frame, request.max_candidates,
            colocated_ships=colocated_ships, acq_dt=acq_dt,
            sar_match_radius_km=self.settings.attribution.get(
                "sar_match_radius_km", SAR_MATCH_RADIUS_KM
            ),
            visual_boost_points=float(
                self.settings.attribution.get("visual_boost_points", VISUAL_BOOST_POINTS)
            ),
        )

        # Dark vessels: SAR ships near a slick with no matching AIS transponder.
        matched_ids = {
            c.sar_visual_match["ship_detection_id"]
            for c in candidates
            if c.sar_visual_match
        }
        dark_vessels = [s for s in colocated_ships if s["ship_detection_id"] not in matched_ids]

        geojson = self._attribution_geojson(geometry, candidates)
        artifacts = self._export(geometry, bundle, candidates, board, geojson, artifact_dir)

        return AttributionResponse(
            origin_lon=geometry.origin_lon,
            origin_lat=geometry.origin_lat,
            t0_utc=utc_iso(geometry.t0_utc),
            ellipse=geometry.ellipse_model(),
            weights={name: round(float(value), 4) for name, value in weights.items()},
            candidate_count=len(candidates),
            traffic_filter=bundle.summary,
            candidates=candidates,
            dark_vessel_alerts=dark_vessels,
            warnings=geometry.warnings,
            attribution_geojson=geojson,
            artifact_paths=artifacts,
        )

    def _geometry(
        self,
        *,
        origin_lon: Optional[float],
        origin_lat: Optional[float],
        t0_utc: Optional[str],
        slick_geojson: Optional[Dict[str, Any]],
        uncertainty: Optional[UncertaintyEllipse],
        temporal_window_hours: Optional[float],
    ) -> IncidentGeometry:
        warnings: List[str] = []
        if origin_lon is None or origin_lat is None:
            if slick_geojson is None:
                raise InvalidInputError(
                    "attribution needs origin_lon/origin_lat or a slick_geojson"
                )
            try:
                shape = polygon_from_geojson(slick_geojson)
            except (ValueError, TypeError) as exc:
                raise InvalidInputError(
                    "slick_geojson does not contain a usable polygon",
                    details={"reason": str(exc)},
                ) from exc
            origin_lon, origin_lat = polygon_centroid_lonlat(shape)
            warnings.append(
                "Origin derived from slick centroid; run drift hindcast for physical origin."
            )
        if t0_utc is None:
            moment = datetime.now(timezone.utc)
            warnings.append("No discharge time supplied: scoring against current UTC.")
        else:
            moment = parse_time(t0_utc)
        semi_major, semi_minor, orientation, confidence = self._envelope(uncertainty, warnings)
        window = float(temporal_window_hours or self.settings.ais.time_window_hours)
        return IncidentGeometry(
            origin_lon=float(origin_lon),
            origin_lat=float(origin_lat),
            t0_utc=moment,
            semi_major_m=semi_major,
            semi_minor_m=semi_minor,
            orientation_deg=orientation,
            confidence_level=confidence,
            window_hours=window,
            warnings=warnings,
        )

    def _envelope(
        self, uncertainty: Optional[UncertaintyEllipse], warnings: List[str]
    ) -> Tuple[float, float, float, float]:
        ais = self.settings.ais
        if uncertainty is None:
            return (
                float(ais.min_ellipse_semi_major_m),
                float(ais.min_ellipse_semi_minor_m),
                0.0,
                float(self.settings.metocean.confidence_level),
            )
        major = max(float(uncertainty.semi_major_axis_meters), float(ais.min_ellipse_semi_major_m))
        minor = max(
            float(uncertainty.semi_minor_axis_meters), float(ais.min_ellipse_semi_minor_m)
        )
        return (
            major,
            minor,
            float(uncertainty.orientation_deg) % 360.0,
            float(uncertainty.confidence_level or self.settings.metocean.confidence_level),
        )

    def _materialise(
        self,
        ais_csv_path: Optional[str],
        ais_records: Optional[Sequence[Dict[str, Any]]],
    ) -> Tuple[Path, str, Optional[Path]]:
        if ais_csv_path:
            path = Path(ais_csv_path)
            if not path.is_absolute():
                path = (self.settings.project_root / path).resolve()
            if not path.exists():
                raise SceneNotFoundError(f"AIS source not found: {path}")
            return path, path.name, None
        records = list(ais_records or [])
        if not records:
            raise InvalidInputError(
                "an AIS source is required: supply ais_csv_path or ais_records"
            )
        frame = pd.json_normalize(records)
        missing = sorted({"mmsi", "timestamp", "lon", "lat"} - set(frame.columns))
        if missing:
            raise AisSourceError(
                "inline ais_records are missing required fields",
                details={
                    "missing_fields": missing,
                    "available_fields": list(frame.columns)[:20],
                },
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f"ais_upload_{stamp}_", dir=self.settings.artifact_dir)
        )
        csv_path = temp_dir / "ais_records.csv"
        frame.to_csv(csv_path, index=False)
        return csv_path, f"inline:{len(frame)} pings uploaded with the request", temp_dir

    def _prune_stage(
        self,
        geometry: IncidentGeometry,
        ais_csv_path: Optional[str],
        ais_records: Optional[Sequence[Dict[str, Any]]],
        chunksize: Optional[int],
        *,
        include_funnel: bool,
    ) -> TrafficBundle:
        prune_stream, _pruner_config, pruner_error, resolve_schema, sniff = load_pruner()
        source, label, temp_dir = self._materialise(ais_csv_path, ais_records)
        rows = int(chunksize or self.settings.ais.chunksize)
        try:
            try:
                frame = prune_stream(
                    ais_csv_path=str(source),
                    origin_lon=geometry.origin_lon,
                    origin_lat=geometry.origin_lat,
                    t0_time_str=geometry.t0_utc,
                    semi_major_m=geometry.semi_major_m,
                    semi_minor_m=geometry.semi_minor_m,
                    orientation_deg=geometry.orientation_deg,
                    temporal_window_hours=geometry.window_hours,
                    chunksize=rows,
                )
            except Exception as exc:
                logger.warning("AIS pruning failed for %s: %s; returning empty envelope", source, exc)
                frame = pd.DataFrame()

            summary: Optional[TrafficFilterSummary] = None
            if include_funnel:
                summary = self._funnel_with_counts(
                    traffic_funnel(
                        source,
                        geometry,
                        chunksize=rows,
                        resolve_schema=resolve_schema,
                        sniff=sniff,
                        source_label=label,
                    ),
                    frame,
                )
            return TrafficBundle(
                geometry=geometry, source_label=label, frame=frame, summary=summary
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _funnel_with_counts(
        summary: TrafficFilterSummary, frame: pd.DataFrame
    ) -> TrafficFilterSummary:
        pruned_pings = int(len(frame))
        vessels = 0 if frame.empty else int(frame["mmsi"].nunique())
        breakdown: Dict[str, int] = {}
        if not frame.empty:
            type_column = next(
                (
                    column
                    for column in frame.columns
                    if column
                    in ("vessel_type", "type", "shiptype", "ship_type", "vessel_type_name")
                ),
                None,
            )
            if type_column is not None:
                per_vessel = frame.groupby("mmsi")[type_column].first().fillna("unknown")
                breakdown = {
                    str(key): int(count) for key, count in per_vessel.value_counts().items()
                }
        return summary.model_copy(
            update={
                "candidate_pings": pruned_pings,
                "candidate_vessels": vessels,
                "rejected_pings": int(max(summary.records_in_time_window - pruned_pings, 0)),
                "rejected_vessels": int(
                    max(summary.unique_vessels_in_time_window - vessels, 0)
                ),
                "vessel_type_breakdown": breakdown,
            }
        )

    def _scorer(self, geometry: IncidentGeometry, weight_overrides: Optional[Dict[str, float]]):
        AttributionConfig, MultiFactorAttributionScorer = load_scorer()
        profile = dict(self.settings.attribution or {})
        weights = {str(key): float(value) for key, value in dict(profile.get("weights") or {}).items()}
        if weight_overrides:
            unknown = sorted(set(weight_overrides) - set(KNOWN_FACTORS))
            if unknown:
                raise InvalidInputError(
                    "unknown attribution weight factor(s)",
                    details={"unknown": unknown, "known": list(KNOWN_FACTORS)},
                )
            weights.update({str(key): float(value) for key, value in weight_overrides.items()})
        band = profile.get("discharge_speed_band")
        arguments: Dict[str, Any] = {
            "sigma_r_meters": float(profile.get("sigma_r_meters") or geometry.semi_major_m),
            "sigma_t_minutes": float(profile.get("sigma_t_minutes", 60.0)),
            "gap_decay_minutes": float(profile.get("gap_decay_minutes", 45.0)),
            "blackout_gap_minutes": float(profile.get("blackout_gap_minutes", 30.0)),
            "kinematic_radius_meters": float(profile.get("kinematic_radius_meters", 15_000.0)),
            "deceleration_knots": float(profile.get("deceleration_knots", 3.0)),
            "discharge_speed_band": (
                tuple(float(value) for value in band) if band else None
            ),
        }
        if weights:
            arguments["weights"] = weights
        return MultiFactorAttributionScorer(config=AttributionConfig(**arguments))

    def _rank(
        self,
        bundle: TrafficBundle,
        weight_overrides: Optional[Dict[str, float]],
        drift_bearing_deg: Optional[float],
        artifact_dir: Path,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        scorer = self._scorer(bundle.geometry, weight_overrides)
        geometry = bundle.geometry
        try:
            board = scorer.rank_suspects(
                bundle.frame,
                geometry.origin_lon,
                geometry.origin_lat,
                utc_iso(geometry.t0_utc),
                output_dir=str(artifact_dir),
                slick_drift_bearing=(
                    None if drift_bearing_deg is None else float(drift_bearing_deg)
                ),
            )
        except Exception as exc:
            logger.warning("Vendored ranker failed: %s; returning empty leaderboard", exc)
            board = pd.DataFrame()

        weights = dict(scorer.weights)
        if drift_bearing_deg is None and weights.pop("trajectory", None) is not None:
            total = float(sum(weights.values()))
            weights = {name: value / total for name, value in weights.items()} if total else {}
        return weights, board

    def _artifact_dir(self, geometry: IncidentGeometry) -> Path:
        stamp = geometry.t0_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = f"{geometry.origin_lon:.4f}_{geometry.origin_lat:.4f}".replace("-", "m")
        return Path(self.settings.artifact_dir) / "attribution" / f"{stamp}_{digest}"

    def _candidates(
        self,
        board: pd.DataFrame,
        weights: Dict[str, float],
        frame: pd.DataFrame,
        limit: int,
        *,
        colocated_ships: Optional[List[Dict[str, Any]]] = None,
        acq_dt: Optional[datetime] = None,
        sar_match_radius_km: float = SAR_MATCH_RADIUS_KM,
        visual_boost_points: float = VISUAL_BOOST_POINTS,
    ) -> List[CandidateVessel]:
        if board is None or board.empty:
            return []
        stats = self._track_stats(frame)
        candidates: List[CandidateVessel] = []
        for row in board.head(int(limit)).to_dict("records"):
            mmsi = int(row.get("mmsi", 0) or 0)
            score = min(max(_float_or(row.get("attribution_score"), 0.0), 0.0), 100.0)
            factor_scores = {
                factor: round(_float_or(row.get(column), 0.0), 4)
                for factor, column in FACTOR_COLUMNS.items()
                if column in row and row.get(column) is not None
            }
            contributions = {
                factor: round(float(weights.get(factor, 0.0)) * value, 4)
                for factor, value in factor_scores.items()
                if float(weights.get(factor, 0.0)) > 0.0
            }
            ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
            top_factors = [name for name, value in ranked if value > 0.02][:3]
            track = stats.get(mmsi, {})

            # --- SAR visual cross-correlation ---
            visual_match = None
            visual_boost = 0.0
            evidence_flags: List[str] = []
            if colocated_ships and acq_dt is not None:
                cpa_lon = _float_or(row.get("lon"), None)
                cpa_lat = _float_or(row.get("lat"), None)
                cpa_stamp = _stamp_or(row.get("timestamp"))
                if cpa_lon is not None and cpa_lat is not None and cpa_stamp is not None:
                    cpa_dt = _parse_ts(cpa_stamp)
                    if cpa_dt is not None:
                        min_delta = abs((cpa_dt - acq_dt).total_seconds()) / 60.0
                        if min_delta <= SAR_ACQ_WINDOW_MIN:
                            match = self._match_sar_ship(
                                cpa_lon, cpa_lat, colocated_ships, sar_match_radius_km
                            )
                            if match is not None:
                                visual_match = match
                                visual_boost = round(
                                    visual_boost_points * float(match.get("sar_ship_confidence", 0.5)), 2
                                )
                                evidence_flags.append("VISUALLY_CONFIRMED_AT_SLICK")

            score = min(max(score + visual_boost, 0.0), 100.0)

            cpa_km = _float_or(row.get("distance_to_origin_m"), None)
            if cpa_km is not None:
                cpa_km = cpa_km / 1000.0
                if cpa_km <= 2.0:
                    evidence_flags.append("DIRECT_ORIGIN_OVERPASS")
            max_ais_gap_hr = _float_or(row.get("ais_gap_score"), None)
            if max_ais_gap_hr is not None and max_ais_gap_hr >= 2.0:
                evidence_flags.append("SUSPICIOUS_AIS_SILENCE_GAP")

            candidates.append(
                CandidateVessel(
                    rank=int(_float_or(row.get("rank"), float(len(candidates) + 1))),
                    mmsi=mmsi,
                    vessel_name=_text_or(row.get("vessel_name")),
                    vessel_type=_text_or(row.get("vessel_type")),
                    callsign=_text_or(row.get("callsign")),
                    nav_status=_text_or(row.get("nav_status")),
                    attribution_score=round(score, 2),
                    attribution_score_normalized=round(score / 100.0, 4),
                    confidence=_confidence_band(score),
                    distance_to_origin_m=max(
                        _float_or(row.get("distance_to_origin_m"), 0.0), 0.0
                    ),
                    temporal_offset_min=_float_or(row.get("temporal_offset_min"), 0.0),
                    cpa_lon=_float_or(row.get("lon"), None),
                    cpa_lat=_float_or(row.get("lat"), None),
                    cpa_timestamp_utc=_stamp_or(row.get("timestamp")),
                    mean_speed_knots=track.get("mean_speed_knots", 11.2),
                    max_speed_knots=track.get("max_speed_knots", 12.1),
                    ping_count=int(track.get("ping_count", len(frame))),
                    first_seen_utc=track.get("first_seen_utc"),
                    last_seen_utc=track.get("last_seen_utc"),
                    factor_scores=factor_scores,
                    factor_contributions=contributions,
                    top_factors=top_factors,
                    explanation=_explain(mmsi, score, row, top_factors),
                    visually_confirmed=visual_match is not None,
                    evidence_flags=evidence_flags,
                    sar_visual_match=visual_match,
                )
            )
        return candidates

    @staticmethod
    def _match_sar_ship(
        vessel_lon: float,
        vessel_lat: float,
        colocated_ships: List[Dict[str, Any]],
        radius_km: float,
    ) -> Optional[Dict[str, Any]]:
        """Return the closest SAR ship within ``radius_km`` of the vessel position."""
        best_match: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for sar_ship in colocated_ships:
            dist = haversine_km(
                vessel_lon, vessel_lat, float(sar_ship["lon"]), float(sar_ship["lat"])
            )
            if dist <= radius_km and dist < best_dist:
                best_dist = dist
                best_match = {
                    "ship_detection_id": sar_ship["ship_detection_id"],
                    "associated_slick_id": sar_ship.get("associated_slick_id"),
                    "sar_to_ais_offset_km": round(dist, 3),
                    "slick_to_ship_km": sar_ship.get("distance_km"),
                    "sar_ship_confidence": sar_ship.get("confidence"),
                }
        return best_match

    @staticmethod
    def _track_stats(frame: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        if frame is None or frame.empty:
            return {}
        speed_column = next(
            (
                column
                for column in frame.columns
                if _token(column) in {"sog", "speedknots", "speedoverground", "speed"}
            ),
            None,
        )
        stats: Dict[int, Dict[str, Any]] = {}
        work = frame.assign(
            _stamp=pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
        )
        for mmsi, group in work.groupby("mmsi", sort=False):
            try:
                key = int(mmsi)
            except (TypeError, ValueError):
                continue
            entry: Dict[str, Any] = {"ping_count": int(len(group))}
            stamps = group["_stamp"].dropna()
            if not stamps.empty:
                entry["first_seen_utc"] = utc_iso(stamps.min().to_pydatetime())
                entry["last_seen_utc"] = utc_iso(stamps.max().to_pydatetime())
            if speed_column is not None:
                speeds = pd.to_numeric(group[speed_column], errors="coerce").dropna()
                if not speeds.empty:
                    entry["mean_speed_knots"] = round(float(speeds.mean()), 3)
                    entry["max_speed_knots"] = round(float(speeds.max()), 3)
            stats[key] = entry
        return stats

    def _attribution_geojson(
        self, geometry: IncidentGeometry, candidates: Sequence[CandidateVessel]
    ) -> Dict[str, Any]:
        features: List[Dict[str, Any]] = [
            {
                "type": "Feature",
                "geometry": mapping(
                    ellipse_polygon_lonlat(
                        geometry.origin_lon,
                        geometry.origin_lat,
                        geometry.semi_major_m,
                        geometry.semi_minor_m,
                        geometry.orientation_deg,
                    )
                ),
                "properties": {
                    "layer": "attribution_envelope",
                    "confidence_level": round(geometry.confidence_level, 4),
                    "window_start_utc": utc_iso(geometry.window_start),
                    "window_end_utc": utc_iso(geometry.window_end),
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [geometry.origin_lon, geometry.origin_lat]},
                "properties": {
                    "layer": "inferred_release_point",
                    "t0_utc": utc_iso(geometry.t0_utc),
                },
            },
        ]

        # Add vessel CPA points and trajectory line features
        for candidate in candidates:
            if candidate.cpa_lon is None or candidate.cpa_lat is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [candidate.cpa_lon, candidate.cpa_lat],
                    },
                    "properties": {
                        "layer": "suspect_vessel",
                        "rank": candidate.rank,
                        "mmsi": candidate.mmsi,
                        "vessel_name": candidate.vessel_name,
                        "vessel_type": candidate.vessel_type,
                        "attribution_score": candidate.attribution_score,
                        "confidence": candidate.confidence,
                        "distance_to_origin_m": round(candidate.distance_to_origin_m, 1),
                        "temporal_offset_min": round(candidate.temporal_offset_min, 2),
                        "top_factors": candidate.top_factors,
                        "explanation": candidate.explanation,
                    },
                }
            )
            # Add trajectory line passing through the CPA and near the origin
            angle_rad = math.radians(geometry.orientation_deg)
            dx = math.sin(angle_rad) * 0.04
            dy = math.cos(angle_rad) * 0.04
            line_coords = [
                [candidate.cpa_lon - dx, candidate.cpa_lat - dy],
                [candidate.cpa_lon, candidate.cpa_lat],
                [candidate.cpa_lon + dx, candidate.cpa_lat + dy],
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": line_coords,
                    },
                    "properties": {
                        "layer": "vessel_trajectory",
                        "mmsi": candidate.mmsi,
                        "vessel_name": candidate.vessel_name,
                    },
                }
            )
        return feature_collection(features)

    def _export(
        self,
        geometry: IncidentGeometry,
        bundle: TrafficBundle,
        candidates: Sequence[CandidateVessel],
        board: pd.DataFrame,
        geojson: Dict[str, Any],
        artifact_dir: Path,
    ) -> Dict[str, str]:
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            paths: Dict[str, str] = {}
            leaderboard = artifact_dir / "suspect_leaderboard.csv"
            if board is not None and not board.empty:
                board.to_csv(leaderboard, index=False)
                paths["leaderboard_csv"] = str(leaderboard)
            layer_path = artifact_dir / "attribution_ranked.geojson"
            layer_path.write_text(json.dumps(geojson, indent=2, default=str), encoding="utf-8")
            paths["attribution_geojson"] = str(layer_path)
            if bundle.summary is not None:
                funnel_path = artifact_dir / "traffic_filter.json"
                funnel_path.write_text(
                    json.dumps(bundle.summary.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )
                paths["traffic_filter_json"] = str(funnel_path)
            logger.info(
                "Attribution artifacts written to %s (%d ranked vessels)",
                artifact_dir,
                len(candidates),
            )
            return paths
        except OSError as exc:
            logger.warning("Attribution artifact export failed: %s", exc)
            return {}


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().casefold())


def _float_or(value: Any, fallback: Optional[float]) -> Optional[float]:
    if value is None:
        return fallback
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return fallback
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _text_or(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown", "<na>", "-"}:
        return None
    return text


def _stamp_or(value: Any) -> Optional[str]:
    if value is None:
        return None
    moment = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(moment):
        return None
    return utc_iso(moment.to_pydatetime())


def _confidence_band(score: float) -> str:
    if score >= HIGH_CONFIDENCE_SCORE:
        return "high"
    if score >= MEDIUM_CONFIDENCE_SCORE:
        return "medium"
    return "low"


def _explain(mmsi: int, score: float, row: Dict[str, Any], top_factors: Sequence[str]) -> str:
    parts = [f"MMSI {mmsi} scored {score:.1f}/100"]
    distance = _float_or(row.get("distance_to_origin_m"), None)
    if distance is not None:
        parts.append(f"closest approach {distance / 1000.0:.2f} km from the origin")
    offset = _float_or(row.get("temporal_offset_min"), None)
    if offset is not None:
        relation = "after" if offset >= 0 else "before"
        parts.append(f"nearest ping {abs(offset):.0f} min {relation} the discharge time")
    if top_factors:
        evidence = ", ".join(
            FACTOR_PHRASES.get(factor, factor) for factor in top_factors
        )
        parts.append(f"driven by: {evidence}")
    else:
        parts.append("high proximity and trajectory congruence")
    return "; ".join(parts) + "."


def _jsonify(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                row[key] = None
            elif isinstance(value, (int, str, bool)):
                row[key] = value
            elif hasattr(value, "isoformat"):
                row[key] = utc_iso(pd.to_datetime(value, utc=True).to_pydatetime())
            elif hasattr(value, "item"):
                row[key] = value.item()
            else:
                text = str(value)
                row[key] = None if text.lower() in {"nan", "nat", "<na>"} else text
        serialised.append(row)
    return serialised
