"""
Multi-Factor Suspect Attribution and Ranking Engine.

Scores candidate vessels (produced by the AIS spatiotemporal pruner) against
the Lagrangian-inferred discharge origin and ranks them into a culprit
leaderboard. All scoring constants are injectable so the model can be
recalibrated per operating theatre without code changes.

Scoring factors (each normalized to [0, 1] before weighting):

    proximity   - Gaussian decay of closest-approach distance to the origin,
                  scaled by the hindcast uncertainty (sigma_r_meters).
    temporal    - Gaussian decay of closest-approach time offset from t0,
                  scaled by sigma_t_minutes.
    kinematics  - deceleration / maneuvering anomaly near the origin window
                  (a discharge usually accompanies a speed drop or turn).
    ais_gaps    - transponder blackout overlapping the origin approach
                  ("dark vessel" behavior), scaled by gap_decay_minutes.
    vessel_type - static risk prior from an overridable type-risk table.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pyproj

logger = logging.getLogger(__name__)

__all__ = ["AttributionConfig", "MultiFactorAttributionScorer"]


#: Canonical factor names and their default weights (normalized at runtime).
#: ``trajectory`` (course-alignment congruence with the slick drift axis)
#: activates only when a drift bearing is supplied at ranking time.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "proximity": 0.35,
    "temporal": 0.25,
    "trajectory": 0.20,
    "kinematics": 0.10,
    "ais_gaps": 0.05,
    "vessel_type": 0.05,
}

#: Alias registry for optional enrichment columns in pruner passthrough.
_KINEMATIC_ALIASES: Dict[str, Sequence[str]] = {
    "sog": ("sog", "speed_knots", "speed_over_ground", "speed"),
    "cog": ("cog", "course_over_ground"),
    "heading": ("heading", "true_heading"),
    "vessel_name": ("vessel_name", "vesselname", "ship_name", "name"),
    "vessel_type": ("vessel_type", "vesseltype", "ship_type", "shiptype"),
    "nav_status": ("nav_status", "navigational_status", "status"),
}

#: Default static risk priors per vessel class (0..1); overridable via config.
DEFAULT_TYPE_RISK: Dict[str, float] = {
    "tanker": 1.00,
    "chemical": 1.00,
    "oil": 1.00,
    "product": 0.90,
    "bulk": 0.65,
    "cargo": 0.60,
    "container": 0.55,
    "general": 0.45,
    "tug": 0.40,
    "fishing": 0.35,
    "passenger": 0.20,
    "pilot": 0.15,
}


def _normalize_token(value):
    """Case/space/underscore/hyphen-insensitive token for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().casefold())


@dataclass(frozen=True)
class AttributionConfig:
    """Tunable attribution profile (all defaults overridable from config.yaml)."""

    #: Factor weights; normalized to sum to 1 before scoring.
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    #: Spatial decay scale in meters (defaults to hindcast semi-major axis).
    sigma_r_meters: float = 10_000.0
    #: Temporal decay scale in minutes.
    sigma_t_minutes: float = 60.0
    #: Transponder-gap decay scale in minutes.
    gap_decay_minutes: float = 45.0
    #: Gap longer than this (minutes) near the origin counts as a blackout.
    blackout_gap_minutes: float = 30.0
    #: Radius around the origin within which kinematic anomalies are counted.
    kinematic_radius_meters: float = 15_000.0
    #: Speed drop (knots) considered a discharge-consistent deceleration.
    deceleration_knots: float = 3.0
    #: Static risk priors per vessel class token.
    type_risk: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPE_RISK))
    #: Fallback risk for unknown vessel classes.
    unknown_type_risk: float = 0.30
    #: SOG window (knots) at CPA treated as discharge-consistent behavior.
    discharge_speed_band: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty.")
        unknown = set(self.weights) - {
            "proximity", "temporal", "trajectory", "kinematics", "ais_gaps", "vessel_type"
        }
        if unknown:
            raise ValueError(f"Unknown attribution factors: {sorted(unknown)}")
        negatives = [name for name, value in self.weights.items() if value < 0]
        if negatives:
            raise ValueError(f"Weights must be non-negative, offending: {negatives}")
        if min(self.weights.values()) == 0 and len(self.weights) > 1:
            logger.warning("Zero-weight factors present: %s", list(self.weights))
        for name, value in (
            ("sigma_r_meters", self.sigma_r_meters),
            ("sigma_t_minutes", self.sigma_t_minutes),
            ("gap_decay_minutes", self.gap_decay_minutes),
            ("blackout_gap_minutes", self.blackout_gap_minutes),
            ("kinematic_radius_meters", self.kinematic_radius_meters),
        ):
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be a positive finite number, got {value}")
        if self.discharge_speed_band is not None:
            band = tuple(self.discharge_speed_band)
            if len(band) != 2 or not (0.0 <= float(band[0]) < float(band[1])):
                raise ValueError(
                    "discharge_speed_band must be (min_knots, max_knots) "
                    "with 0 <= min < max."
                )

    def normalized_weights(self) -> Dict[str, float]:
        """Return weights rescaled to sum to 1 (dynamic weighting contract)."""
        total = float(sum(self.weights.values()))
        if total <= 0:
            raise ValueError("Attribution weights sum to zero; cannot normalize.")
        return {name: float(value) / total for name, value in self.weights.items()}


class MultiFactorAttributionScorer:
    """Ranks pruned AIS candidates into a culprit leaderboard.

    Consumes the canonical pruner output (mmsi/timestamp/lon/lat plus optional
    passthrough enrichment columns) and the Stage-1 telemetry values. All
    scales come from :class:`AttributionConfig`, populated by callers from
    config.yaml or CLI overrides - no scoring constant is fixed in code.
    """

    _FACTOR_COLUMNS = {
        "proximity": "proximity_score",
        "temporal": "temporal_score",
        "trajectory": "trajectory_score",
        "kinematics": "kinematic_score",
        "ais_gaps": "ais_gap_score",
        "vessel_type": "type_risk_score",
    }

    def __init__(
        self,
        weights=None,
        sigma_r_meters=None,
        sigma_t_minutes=None,
        gap_decay_minutes=None,
        config=None,
    ):
        overrides = {}
        if weights is not None:
            overrides["weights"] = dict(weights)
        if sigma_r_meters is not None:
            overrides["sigma_r_meters"] = float(sigma_r_meters)
        if sigma_t_minutes is not None:
            overrides["sigma_t_minutes"] = float(sigma_t_minutes)
        if gap_decay_minutes is not None:
            overrides["gap_decay_minutes"] = float(gap_decay_minutes)
        if config is not None:
            self.config = replace(config, **overrides) if overrides else config
        else:
            base = {
                "weights": dict(DEFAULT_WEIGHTS),
                "sigma_r_meters": 10_000.0,
                "sigma_t_minutes": 60.0,
                "gap_decay_minutes": 45.0,
            }
            base.update(overrides)
            self.config = AttributionConfig(**base)
        self.weights = self.config.normalized_weights()
        self._geod = pyproj.Geod(ellps="WGS84")
        logger.info(
            "Attribution profile: weights=%s sigma_r=%.0fm sigma_t=%.0fmin "
            "gap_decay=%.0fmin",
            {k: round(v, 3) for k, v in self.weights.items()},
            self.config.sigma_r_meters,
            self.config.sigma_t_minutes,
            self.config.gap_decay_minutes,
        )

    def rank_suspects(
        self,
        pruned_ais_df,
        origin_lon,
        origin_lat,
        t0_timestamp_iso,
        output_dir=None,
        slick_drift_bearing=None,
    ):
        """Score and rank candidates; optionally persist leaderboard + culprit GeoJSON.

        Args:
            pruned_ais_df: Canonical pruner output (may be empty).
            origin_lon: Hindcast origin longitude (WGS84).
            origin_lat: Hindcast origin latitude (WGS84).
            t0_timestamp_iso: Inferred discharge instant (ISO 8601).
            output_dir: When given, writes ``suspect_leaderboard.csv`` and
                ``culprit_vessel.geojson`` there.
            slick_drift_bearing: Optional metocean drift bearing (degrees,
                clockwise from True North). When absent the
                trajectory-congruence factor is disabled and the remaining
                factor weights renormalize dynamically.

        Returns:
            Ranked leaderboard DataFrame (typed empty frame when no suspects).
        """
        required = {"mmsi", "timestamp", "lon", "lat"}
        missing = required - set(pruned_ais_df.columns)
        if missing:
            raise ValueError(f"Pruned AIS frame lacks required columns: {sorted(missing)}")
        if pruned_ais_df.empty:
            logger.warning("No candidates supplied to the attribution scorer.")
            return self._empty_leaderboard()

        t0 = pd.to_datetime(t0_timestamp_iso, utc=True)
        cfg = self.config
        df = pruned_ais_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df = df.dropna(subset=["timestamp", "lon", "lat"])
        if df.empty:
            logger.warning("No attributable candidates after timestamp/coordinate coercion.")
            return self._empty_leaderboard()

        df["distance_to_origin_m"] = np.round(
            self._geodesic_m(
                df["lon"].to_numpy(dtype="float64"),
                df["lat"].to_numpy(dtype="float64"),
                origin_lon,
                origin_lat,
            ),
            2,
        )
        df["temporal_offset_min"] = np.round(
            (df["timestamp"] - t0).dt.total_seconds() / 60.0, 3
        )

        # Per-vessel aggregation: pick the ping minimizing the joint
        # (spatial, temporal) Mahalanobis-style approach metric.
        df["_approach_metric"] = (
            (df["distance_to_origin_m"] / cfg.sigma_r_meters) ** 2
            + (df["temporal_offset_min"].abs() / cfg.sigma_t_minutes) ** 2
        )
        approach_idx = df.groupby("mmsi", sort=False)["_approach_metric"].idxmin()
        approach = df.loc[approach_idx].copy()
        logger.info(
            "Aggregated %d candidate pings into %d unique suspect vessels.",
            len(df),
            len(approach),
        )
        return self._score_and_rank(
            df, approach, t0, origin_lon, origin_lat, output_dir, slick_drift_bearing
        )

    # ------------------------------------------------------------- internals
    def _score_and_rank(
        self, df, approach, t0, origin_lon, origin_lat, output_dir,
        slick_drift_bearing=None,
    ):
        """Compute the factor scores, combine them, rank and export."""
        cfg = self.config

        # Factor 1: spatial proximity - Gaussian decay on closest approach.
        approach["proximity_score"] = np.exp(
            -0.5 * (approach["distance_to_origin_m"] / cfg.sigma_r_meters) ** 2
        )

        # Factor 2: temporal alignment - Gaussian decay on |dt| from t0.
        approach["temporal_score"] = np.exp(
            -0.5 * (approach["temporal_offset_min"].abs() / cfg.sigma_t_minutes) ** 2
        )

        # Factor 3: kinematic anomaly (deceleration / maneuvering) near origin.
        approach["kinematic_score"] = self._kinematic_scores(df, approach)

        # Factor 4: transponder blackout overlapping the origin approach.
        approach["ais_gap_score"] = self._ais_gap_scores(df, approach)

        # Factor 5: static vessel-class risk prior (alias-aware identity cols).
        name_col = self._column_or_none(approach, "vessel_name")
        type_col = self._column_or_none(approach, "vessel_type")
        approach["type_risk_score"] = approach.apply(
            lambda row: self._type_risk_of_row(row, name_col=name_col, type_col=type_col),
            axis=1,
        )
        # Expose normalized identity columns for dashboards and exports.
        if name_col and name_col != "vessel_name":
            approach["vessel_name"] = approach[name_col]
        if type_col and type_col != "vessel_type":
            approach["vessel_type"] = approach[type_col]

        factor_columns = dict(self._FACTOR_COLUMNS)
        factor_weights = dict(self.weights)
        if slick_drift_bearing is None:
            if factor_weights.get("trajectory"):
                logger.info(
                    "Slick drift bearing unavailable; trajectory-congruence "
                    "factor disabled and remaining weights renormalized."
                )
            factor_weights.pop("trajectory", None)
            factor_columns.pop("trajectory", None)
        else:
            approach["trajectory_score"] = self._trajectory_scores(
                approach, float(slick_drift_bearing)
            )

        weight_total = float(sum(factor_weights.values()))
        if weight_total <= 0:
            raise ValueError("Attribution weights sum to zero after factor selection.")
        factor_weights = {
            name: value / weight_total for name, value in factor_weights.items()
        }

        for column in factor_columns.values():
            approach[column] = approach[column].astype(float).round(4)

        combined = sum(
            factor_weights[factor] * approach[column]
            for factor, column in factor_columns.items()
        )
        approach["attribution_score"] = (combined * 100.0).round(2)

        board = approach.drop(columns=["_approach_metric"]).sort_values(
            by=["attribution_score", "distance_to_origin_m"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        board.insert(0, "rank", np.arange(1, len(board) + 1))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            leaderboard_path = os.path.join(output_dir, "suspect_leaderboard.csv")
            board.to_csv(leaderboard_path, index=False)
            logger.info("Suspect leaderboard written to %s", leaderboard_path)
            top = board.iloc[0]
            logger.info(
                "[Top suspect] MMSI %s (%s) score %.2f/100 - CPA %.0f m, dt %.1f min",
                top["mmsi"],
                top.get("vessel_name", "UNKNOWN"),
                top["attribution_score"],
                top["distance_to_origin_m"],
                top["temporal_offset_min"],
            )
            self._export_culprit_geojson(board, output_dir)
        return board

    def _geodesic_m(self, lons, lats, origin_lon, origin_lat):
        """Vectorized WGS84 geodesic distances (meters) from origin to pings."""
        if len(lons) > 1:
            _, _, dist = self._geod.inv(
                np.full(len(lons), origin_lon),
                np.full(len(lons), origin_lat),
                lons,
                lats,
            )
            return np.asarray(dist, dtype="float64")
        _, _, dist = self._geod.inv(
            origin_lon, origin_lat, float(lons[0]), float(lats[0])
        )
        return np.array([dist], dtype="float64")

    def _column_or_none(self, df, canonical):
        """Resolve an optional passthrough column from its aliases.

        Matching is token-normalized (case/space/underscore-insensitive), so
        passthrough columns like ``VesselName`` or ``Speed_Knots`` resolve
        regardless of the source feed's capitalization convention.
        """
        column_tokens = {_normalize_token(col): col for col in df.columns}
        for alias in _KINEMATIC_ALIASES.get(canonical, (canonical,)):
            actual = column_tokens.get(_normalize_token(alias))
            if actual:
                return actual
        return None

    def _kinematic_scores(self, df, approach):
        """Deceleration / maneuvering anomaly per vessel near the origin.

        Returns 0..1: 0.7 for a >= deceleration_knots speed drop inside the
        kinematic radius, +0.3 for a >= 40 deg course/heading change.
        """
        cfg = self.config
        sog_col = self._column_or_none(df, "sog")
        cog_col = self._column_or_none(df, "cog")
        head_col = self._column_or_none(df, "heading")
        out = pd.Series(0.0, index=approach.index, dtype="float64")
        for mmsi, probe in approach.iterrows():
            track = df[
                (df["mmsi"] == probe["mmsi"])
                & (df["distance_to_origin_m"] <= cfg.kinematic_radius_meters)
            ]
            if track.empty:
                continue
            score = 0.0
            if sog_col:
                speeds = pd.to_numeric(track[sog_col], errors="coerce").dropna()
                if not speeds.empty:
                    speed_drop = float(speeds.max() - speeds.min())
                    if speed_drop >= cfg.deceleration_knots:
                        score += 0.7
                    elif speed_drop >= cfg.deceleration_knots / 2.0:
                        score += 0.35
                        score += self._max_course_change(track, cog_col, head_col) * 0.3
            else:
                turn = self._max_course_change(track, cog_col, head_col)
                if turn >= 40.0:
                    score += 0.4
            if cfg.discharge_speed_band and sog_col:
                cpa_sog = pd.to_numeric(probe.get(sog_col), errors="coerce")
                if pd.notna(cpa_sog) and (
                    float(cfg.discharge_speed_band[0])
                    <= float(cpa_sog)
                    <= float(cfg.discharge_speed_band[1])
                ):
                    score += 0.3
            out.at[mmsi] = min(score, 1.0)
        return out

    @staticmethod
    def _max_course_change(track, cog_col, head_col):
        """Largest absolute course/heading delta (degrees) along the track."""
        best = 0.0
        for col in (cog_col, head_col):
            if not col:
                continue
            bearings = pd.to_numeric(track[col], errors="coerce").dropna()
            if len(bearings) >= 2:
                delta = float(bearings.diff().abs().max())
                if pd.notna(delta):
                    best = max(best, min(delta, 360.0 - delta) if delta > 180 else delta)
        return best

    def _trajectory_scores(self, approach, slick_drift_bearing):
        """Course-alignment congruence with the slick drift axis (0..1).

        Compares the vessel's course/heading at CPA against the metocean drift
        bearing, then gates the result by the spatial proximity decay: course
        congruence is corroborative evidence and only counts for vessels near
        the inferred origin (a distant vessel steaming along the drift axis is
        not a plausible source).
        """
        out = pd.Series(0.0, index=approach.index, dtype="float64")
        cog_col = self._column_or_none(approach, "cog")
        head_col = self._column_or_none(approach, "heading")
        course_col = cog_col or head_col
        if not course_col:
            logger.info(
                "No course/heading columns available; trajectory factor scores 0."
            )
            return out
        bearings = pd.to_numeric(approach[course_col], errors="coerce")
        angular_gap = (bearings - float(slick_drift_bearing)).abs() % 360.0
        angular_gap = angular_gap.where(angular_gap <= 180.0, 360.0 - angular_gap)
        congruence = np.clip(
            np.cos(np.radians(angular_gap.to_numpy(dtype="float64"))), 0.0, 1.0
        )
        proximity_gate = np.exp(
            -0.5
            * (
                approach["distance_to_origin_m"].to_numpy(dtype="float64")
                / self.config.sigma_r_meters
            )
            ** 2
        )
        gated = np.nan_to_num(congruence, nan=0.0) * proximity_gate
        return pd.Series(gated, index=approach.index)

    def _ais_gap_scores(self, df, approach):
        """Blackout score: long transponder gap overlapping the approach window."""
        cfg = self.config
        out = pd.Series(0.0, index=approach.index, dtype="float64")
        for mmsi, probe in approach.iterrows():
            track = df[df["mmsi"] == probe["mmsi"]].sort_values("timestamp")
            if len(track) < 2:
                continue
            gaps = track["timestamp"].diff().dt.total_seconds().dropna() / 60.0
            if gaps.empty:
                continue
            max_gap = float(gaps.max())
            if max_gap < cfg.blackout_gap_minutes:
                continue
            score = float(
                np.exp(-0.5 * (max_gap / cfg.gap_decay_minutes) ** 2)
            )
            if probe["distance_to_origin_m"] > cfg.kinematic_radius_meters:
                score *= 0.5
            out.at[mmsi] = min(score, 1.0)
        return out

    def _type_risk_of_row(self, row, name_col=None, type_col=None):
        """Static risk prior from the (overridable) vessel-class risk table."""
        tokens = ""
        for column in (type_col, name_col, "vessel_type", "vessel_name"):
            value = row.get(column) if column else None
            if isinstance(value, str):
                tokens += " " + value.casefold()
        for token, risk in self.config.type_risk.items():
            if token in tokens:
                return float(risk)
        return float(self.config.unknown_type_risk)

    @staticmethod
    def _empty_leaderboard():
        """Typed, schema-stable empty leaderboard for zero-suspect runs."""
        float_series = pd.Series(dtype="float64")
        return pd.DataFrame({
            "rank": pd.Series(dtype="int64"),
            "mmsi": pd.Series(dtype="int64"),
            "vessel_name": pd.Series(dtype="object"),
            "vessel_type": pd.Series(dtype="object"),
            "attribution_score": float_series,
            "distance_to_origin_m": float_series,
            "temporal_offset_min": float_series,
            "proximity_score": float_series,
            "temporal_score": float_series,
            "trajectory_score": float_series,
            "kinematic_score": float_series,
            "ais_gap_score": float_series,
            "type_risk_score": float_series,
            "last_seen": pd.Series(dtype="datetime64[ns, UTC]"),
        })

    def _export_culprit_geojson(self, board, output_dir):
        """Persist the top-ranked suspect as a dashboard-ready GeoJSON point."""
        import json

        top = board.iloc[0]
        try:
            lon = float(top["lon"])
            lat = float(top["lat"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Culprit coordinates unavailable; GeoJSON export skipped.")
            return
        properties = {}
        for key, value in top.items():
            if key in ("lon", "lat"):
                continue
            if hasattr(value, "item"):
                value = value.item()
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            properties[key] = value

        # Dashboard-compatibility aliases: the Streamlit app and FastAPI
        # backend render these canonical dossier fields from the culprit
        # GeoJSON, so expose them regardless of the source feed's naming.
        sog_col = self._column_or_none(board, "sog")
        if sog_col is not None:
            properties.setdefault("avg_speed_knots", properties.get(sog_col))
        properties.setdefault("min_distance_m", properties.get("distance_to_origin_m"))
        properties.setdefault("last_seen", properties.get("cpa_timestamp", properties.get("timestamp")))
        feature = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }],
        }
        path = os.path.join(output_dir, "culprit_vessel.geojson")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(feature, handle, indent=2, default=str)
        logger.info("Culprit GeoJSON written to %s", path)
