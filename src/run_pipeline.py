#!/usr/bin/env python3
"""
NTRO Automated Spill Hindcasting and Multi-Factor Suspect Attribution Pipeline.
Integrates Lagrangian back-tracking with high-resolution AIS vessel identification.
"""

import argparse
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd

# Configure Production Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineOrchestrator")


def _load_yaml_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Loads pipeline defaults from config.yaml (tolerates absence/malformation)."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.info("No config.yaml found; falling back to built-in defaults.")
        return {}
    except Exception as exc:  # config is advisory, never fatal
        logger.warning(f"Could not parse config '{path}' ({exc}); using built-in defaults.")
        return {}


try:
    from src.opendrift_engine import OpenDriftHindcastEngine
except ImportError:
    OpenDriftHindcastEngine = None

from src.ais_pruner import prune_marinecadastre_ais

try:
    from src.attribution_engine import (
        AttributionConfig,
        MultiFactorAttributionScorer,
    )
except ImportError:  # attribution engine is optional at import time (like OpenDrift)
    AttributionConfig = None
    MultiFactorAttributionScorer = None


class NTROSpillAttributionPipeline:
    """
    End-to-end orchestrator for satellite slick attribution:
    Slick Extraction -> Lagrangian Hindcasting -> Trajectory Pruning -> Attribution.
    """

    def __init__(self, output_dir: str = "results", config: Optional[Dict[str, Any]] = None):
        # Operational defaults come from config.yaml when not injected explicitly.
        self.config = dict(config) if config is not None else _load_yaml_config()
        self.output_dir = os.path.abspath(output_dir)
        self.hindcast_dir = os.path.join(self.output_dir, "hindcast")
        self.pruning_dir = os.path.join(self.output_dir, "pruned_traffic")
        self.attribution_dir = os.path.join(self.output_dir, "attribution")

        for directory in [self.hindcast_dir, self.pruning_dir, self.attribution_dir]:
            os.makedirs(directory, exist_ok=True)

    def execute_stage1_hindcast(
        self,
        slick_geojson: str,
        sat_time_iso: str,
        current_nc: str,
        wind_nc: str,
        wave_nc: Optional[str] = None,
        hindcast_hours: Optional[float] = None,
        time_step_min: Optional[int] = None,
        particles: Optional[int] = None,
        oil_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs Lagrangian backward hindcast to determine probable discharge point.

        Simulation parameters default to config.yaml -> pipeline.* when not
        supplied explicitly (formerly hard-coded literals).
        """
        logger.info("=" * 70)
        logger.info(">>> STAGE 1: METOCEAN LAGRANGIAN BACKWARD HINDCASTING")
        logger.info("=" * 70)

        pipeline_cfg = self.config.get("pipeline") or {}
        hindcast_hours = float(
            hindcast_hours if hindcast_hours is not None
            else pipeline_cfg.get("hindcast_hours", 24.0)
        )
        time_step_min = int(
            time_step_min if time_step_min is not None
            else pipeline_cfg.get("time_step_minutes", 15)
        )
        particles = int(
            particles if particles is not None
            else pipeline_cfg.get("particles", 2000)
        )
        oil_type = oil_type or str(pipeline_cfg.get("oil_type", "GENERIC MEDIUM CRUDE"))
        logger.info(
            f"Stage 1 profile: hindcast={hindcast_hours}h, step={time_step_min}min, "
            f"particles={particles}, oil={oil_type}"
        )

        if OpenDriftHindcastEngine is None:
            raise RuntimeError(
                "OpenDrift is not installed. Activate the required conda/virtual "
                "environment or run with --demo."
            )

        t_sat = pd.to_datetime(sat_time_iso, utc=True).to_pydatetime()

        metocean_cfg = self.config.get("metocean") or {}
        engine = OpenDriftHindcastEngine(
            current_data_path=current_nc,
            wind_data_path=wind_nc,
            wave_data_path=wave_nc,
            oil_type=oil_type,
            horizontal_diffusivity=float(metocean_cfg.get("horizontal_diffusivity", 10.0)),
            wind_drift_factor=float(metocean_cfg.get("wind_drift_factor", 0.032)),
            weathering_model=str(metocean_cfg.get("weathering_model", "adios")),
            release_seed_buffer_deg=float(metocean_cfg.get("release_seed_buffer_deg", 0.01)),
            confidence_level=float(metocean_cfg.get("confidence_level", 0.95)),
            output_dir=self.hindcast_dir
        )

        telemetry = engine.run_backward_hindcast(
            slick_geojson_path=slick_geojson,
            satellite_time=t_sat,
            hindcast_hours=hindcast_hours,
            time_step_minutes=time_step_min,
            num_particles=particles
        )

        telemetry_path = os.path.join(self.hindcast_dir, "origin_telemetry.json")
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=4)

        logger.info(f"[✓] Stage 1 complete. Inferred origin: {telemetry_path}")
        return telemetry

    def execute_stage2_pruning(
        self,
        telemetry: Dict[str, Any],
        ais_csv_path: str,
        temporal_window_hours: Optional[float] = None,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """Prunes AIS dataset using the rotated uncertainty ellipse.

        Window/chunksize default to config.yaml -> ais.time_window_hours /
        ais.chunksize when not supplied explicitly; the AIS schema is
        auto-detected by the pruner, so any feed layout works.
        """
        logger.info("=" * 70)
        logger.info(">>> STAGE 2: ELLIPTICAL SPATIO-TEMPORAL TRAFFIC PRUNING")
        logger.info("=" * 70)

        ais_cfg = self.config.get("ais") or {}
        effective_window = (
            float(temporal_window_hours) if temporal_window_hours is not None
            else float(ais_cfg.get("time_window_hours", 3.0))
        )
        effective_chunksize = (
            int(chunksize) if chunksize is not None
            else int(ais_cfg.get("chunksize", 250000))
        )
        logger.info(
            f"Stage 2 profile: +/-{effective_window}h window, chunksize={effective_chunksize}"
        )

        uncertainty = telemetry.get("uncertainty", {})
        semi_major = float(uncertainty.get("semi_major_axis_meters", 10000.0))
        semi_minor = float(uncertainty.get("semi_minor_axis_meters", 6000.0))
        orientation = float(uncertainty.get("orientation_deg", 0.0))

        pruned_df = prune_marinecadastre_ais(
            ais_csv_path=ais_csv_path,
            origin_lon=float(telemetry["origin_lon"]),
            origin_lat=float(telemetry["origin_lat"]),
            t0_time_str=str(telemetry["inferred_t0"]),
            semi_major_m=semi_major,
            semi_minor_m=semi_minor,
            orientation_deg=orientation,
            temporal_window_hours=effective_window,
            chunksize=effective_chunksize
        )

        pruned_csv_path = os.path.join(self.pruning_dir, "pruned_candidate_trajectories.csv")
        pruned_df.to_csv(pruned_csv_path, index=False)
        logger.info(f"[✓] Stage 2 complete. {len(pruned_df)} suspect pings written to {pruned_csv_path}")

        return pruned_df

    def execute_stage3_attribution(
        self,
        pruned_df: pd.DataFrame,
        telemetry: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Computes multi-factor attribution scores for candidate vessels."""
        logger.info("=" * 70)
        logger.info(">>> STAGE 3: MULTI-FACTOR CULPRIT ATTRIBUTION")
        logger.info("=" * 70)

        if MultiFactorAttributionScorer is None or AttributionConfig is None:
            raise RuntimeError(
                "src.attribution_engine is unavailable. Ensure src/attribution_engine.py "
                "is present next to the other pipeline modules."
            )
        if pruned_df.empty:
            logger.warning("[!] Candidate DataFrame is empty. Attribution scoring omitted.")
            return pd.DataFrame()

        # Every key under config.yaml -> attribution flows into the scoring
        # profile dynamically; unknown keys are ignored defensively.
        attribution_cfg = dict(self.config.get("attribution") or {})
        weights = weights or attribution_cfg.pop("weights", None)
        default_sigma_r = float(
            telemetry.get("uncertainty", {}).get("semi_major_axis_meters", 10000.0)
        )
        attribution_cfg.setdefault("sigma_r_meters", default_sigma_r)
        attribution_cfg.setdefault("sigma_t_minutes", 60.0)
        attribution_cfg.setdefault("gap_decay_minutes", 45.0)
        known_fields = {field_info.name for field_info in fields(AttributionConfig)}
        scorer_config = AttributionConfig(
            **{
                key: value
                for key, value in attribution_cfg.items()
                if key in known_fields
            }
        )
        scorer = MultiFactorAttributionScorer(config=scorer_config, weights=weights)

        leaderboard = scorer.rank_suspects(
            pruned_ais_df=pruned_df,
            origin_lon=telemetry["origin_lon"],
            origin_lat=telemetry["origin_lat"],
            t0_timestamp_iso=telemetry["inferred_t0"],
            output_dir=self.attribution_dir,
            slick_drift_bearing=telemetry.get("drift_bearing_deg"),
        )

        final_csv_path = os.path.join(self.output_dir, "culprit_attribution_leaderboard.csv")
        leaderboard.to_csv(final_csv_path, index=False)
        logger.info(f"[✓] Stage 3 complete. Ranked leaderboard written to: {final_csv_path}")

        return leaderboard

    def run(
        self,
        slick_geojson: str,
        sat_time_iso: str,
        ais_csv_path: str,
        current_nc: str,
        wind_nc: str,
        wave_nc: Optional[str] = None,
        hindcast_hours: Optional[float] = None,
        time_step_min: Optional[int] = None,
        particles: Optional[int] = None,
        oil_type: Optional[str] = None,
        temporal_window_hours: Optional[float] = None
    ) -> pd.DataFrame:
        """Executes the full pipeline sequentially (defaults from config.yaml)."""
        start_time = datetime.now()
        logger.info("Launching Automated NTRO Attribution Pipeline...")

        telemetry = self.execute_stage1_hindcast(
            slick_geojson=slick_geojson,
            sat_time_iso=sat_time_iso,
            current_nc=current_nc,
            wind_nc=wind_nc,
            wave_nc=wave_nc,
            hindcast_hours=hindcast_hours,
            time_step_min=time_step_min,
            particles=particles,
            oil_type=oil_type
        )

        pruned_df = self.execute_stage2_pruning(
            telemetry=telemetry,
            ais_csv_path=ais_csv_path,
            temporal_window_hours=temporal_window_hours
        )

        leaderboard = self.execute_stage3_attribution(
            pruned_df=pruned_df,
            telemetry=telemetry
        )

        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline executed successfully in {runtime:.2f}s.")
        return leaderboard


# =========================================================================
# SELF-CONTAINED VERIFICATION HARNESS
# =========================================================================

def _run_mock_demo_suite(output_dir: str, ais_override_path: Optional[str] = None):
    """Executes a dry run against an elliptical profile to verify geometric math."""
    logger.info("Initializing synthetic verification harness...")
    demo_dir = os.path.join(output_dir, "demo_harness")
    os.makedirs(demo_dir, exist_ok=True)

    t_sat = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    t0_dt = t_sat - timedelta(hours=6)

    # 1. OpenDrift Hindcast Output Manifest Mock
    telemetry = {
        "origin_lon": 4.00,
        "origin_lat": 60.00,
        "inferred_t0": t0_dt.isoformat(),
        "satellite_timestamp": t_sat.isoformat(),
        "uncertainty": {
            "semi_major_axis_meters": 12000.0,
            "semi_minor_axis_meters": 5000.0,
            "orientation_deg": 45.0  # 45 deg clockwise from North (NE-SW corridor)
        }
    }

    # 2. Determine target AIS dataset
    if ais_override_path and os.path.isfile(ais_override_path):
        target_ais_path = ais_override_path
        logger.info(f"Using designated AIS path: {target_ais_path}")
    else:
        target_ais_path = os.path.join(demo_dir, "mock_ais_stream.csv")
        logger.info(f"Synthesizing test fixtures to: {target_ais_path}")
        records = [
            # Culprit 1: Directly inside the oriented ellipse along the NE-SW corridor
            {
                "MMSI": 999000111,
                "BaseDateTime": (t0_dt - timedelta(minutes=10)).isoformat(),
                "LAT": 60.05,
                "LON": 4.07,
                "SOG": 4.2,
                "COG": 45.0,
                "VesselType": 80,
                "VesselName": "TANKER_ALPHA"
            },
            # False Positive Candidate: Within circular buffer (11 km NW), but OUTSIDE rotated ellipse
            {
                "MMSI": 999000222,
                "BaseDateTime": t0_dt.isoformat(),
                "LAT": 60.07,
                "LON": 3.86,
                "SOG": 14.0,
                "COG": 315.0,
                "VesselType": 70,
                "VesselName": "TRANSIT_BETA"
            },
            # Noise: Temporal Outlier (24 hours prior)
            {
                "MMSI": 999000333,
                "BaseDateTime": (t0_dt - timedelta(days=1)).isoformat(),
                "LAT": 60.00,
                "LON": 4.00,
                "SOG": 12.0,
                "COG": 90.0,
                "VesselType": 80,
                "VesselName": "HISTORIC_GAMMA"
            }
        ]
        pd.DataFrame(records).to_csv(target_ais_path, index=False)

    pipeline = NTROSpillAttributionPipeline(output_dir=output_dir)

    pruned = pipeline.execute_stage2_pruning(
        telemetry=telemetry,
        ais_csv_path=target_ais_path,
        temporal_window_hours=3.0
    )

    leaderboard = pipeline.execute_stage3_attribution(
        pruned_df=pruned,
        telemetry=telemetry
    )

    print("\n" + "=" * 80)
    print("                 PIPELINE EXECUTION RESULTS")
    print("=" * 80)
    if not leaderboard.empty:
        print(leaderboard.to_string(index=False))
    else:
        print("Pipeline finished: No suspect vessels intersected the elliptical envelope.")
    print("=" * 80 + "\n")


def _merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursive dict merge used for per-incident config overrides."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_incident_manifest(
    manifest_path: str,
    default_config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Executes a batch of independent spill incidents (multi-region scaling).

    Each entry in ``manifest.yaml -> incidents`` is a fully independent
    incident (own region, own AIS feed, own output directory, own config
    overrides), so simultaneous incidents across different geographic regions
    are processed in isolation - a failure in one never aborts the others.

    Args:
        manifest_path: YAML manifest with an ``incidents`` list.
        default_config_path: Base configuration merged with each incident's
            ``config_overrides``.

    Returns:
        ``{"succeeded": [names], "failed": [{"name", "error"}]}``
    """
    import yaml as _yaml

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = _yaml.safe_load(handle) or {}
    incidents = manifest.get("incidents") or []
    if not incidents:
        logger.warning("Incident manifest contains no entries: %s", manifest_path)
        return {"succeeded": [], "failed": []}

    succeeded, failed = [], []
    for index, spec in enumerate(incidents):
        name = str(spec.get("name", f"incident_{index:02d}"))
        try:
            cfg = _merge_config(_load_yaml_config(default_config_path), spec.get("config_overrides"))
            pipeline = NTROSpillAttributionPipeline(
                output_dir=spec.get("output_dir") or os.path.join("results", name),
                config=cfg,
            )
            pipeline.run(
                slick_geojson=spec["slick"],
                sat_time_iso=spec["sat_time"],
                ais_csv_path=spec.get("ais_csv") or cfg.get("paths", {}).get("ais_csv", "data/raw/ais_data.csv"),
                current_nc=spec["current_nc"],
                wind_nc=spec["wind_nc"],
                wave_nc=spec.get("wave_nc"),
                hindcast_hours=spec.get("hindcast_hours"),
                time_step_min=spec.get("time_step_minutes"),
                particles=spec.get("particles"),
                oil_type=spec.get("oil_type"),
                temporal_window_hours=spec.get("window_hours"),
            )
            succeeded.append(name)
        except Exception as exc:  # per-incident fault isolation
            logger.error("Incident %s failed: %s", name, exc, exc_info=True)
            failed.append({"name": name, "error": str(exc)})

    logger.info(
        "Batch execution complete: %d succeeded, %d failed.", len(succeeded), len(failed)
    )
    return {"succeeded": succeeded, "failed": failed}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NTRO Automated Spill Attribution Production Pipeline"
    )
    parser.add_argument("--slick", type=str, help="Path to detected oil spill polygon (.geojson)")
    parser.add_argument("--sat-time", type=str, help="Observation timestamp (ISO 8601 format)")
    parser.add_argument("--ais-csv", type=str, help="Path to AIS CSV dataset (any schema; auto-detected)")
    parser.add_argument("--current-nc", type=str, help="Path to Copernicus ocean current NetCDF")
    parser.add_argument("--wind-nc", type=str, help="Path to ERA5 10m wind NetCDF")
    parser.add_argument("--wave-nc", type=str, default=None, help="Path to WaveWatch III NetCDF")
    parser.add_argument("--hindcast-hours", type=float, default=None, help="Lagrangian hindcast window (hrs; default: config.yaml pipeline.hindcast_hours)")
    parser.add_argument("--time-step", type=int, default=None, help="Hindcast integration step in minutes (default: config.yaml pipeline.time_step_minutes)")
    parser.add_argument("--particles", type=int, default=None, help="Lagrangian particle count (default: config.yaml pipeline.particles)")
    parser.add_argument("--oil-type", type=str, default=None, help="OpenDrift oil classification (default: config.yaml pipeline.oil_type)")
    parser.add_argument("--window-hours", type=float, default=None, help="Temporal window tolerance (± hrs; default: config.yaml ais.time_window_hours)")
    parser.add_argument("--output-dir", type=str, default="results", help="Target output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to pipeline configuration YAML")
    parser.add_argument("--incidents", type=str, default=None, help="YAML manifest with multiple independent incidents (multi-region batch mode)")
    parser.add_argument("--demo", action="store_true", help="Run local validation with synthetic harness")
    return parser


if __name__ == "__main__":
    cli = build_cli()
    args = cli.parse_args()

    if args.demo:
        _run_mock_demo_suite(
            output_dir=args.output_dir,
            ais_override_path=args.ais_csv
        )
        sys.exit(0)

    if args.incidents:
        run_incident_manifest(args.incidents, default_config_path=args.config)
        sys.exit(0)

    required_args = ["slick", "sat_time", "ais_csv", "current_nc", "wind_nc"]
    unmet_args = [arg for arg in required_args if getattr(args, arg) is None]
    if unmet_args:
        cli.error(f"Missing mandatory parameters: {unmet_args}. Run with --demo to dry run.")

    orchestrator = NTROSpillAttributionPipeline(
        output_dir=args.output_dir,
        config=_load_yaml_config(args.config)
    )
    orchestrator.run(
        slick_geojson=args.slick,
        sat_time_iso=args.sat_time,
        ais_csv_path=args.ais_csv,
        current_nc=args.current_nc,
        wind_nc=args.wind_nc,
        wave_nc=args.wave_nc,
        hindcast_hours=args.hindcast_hours,
        time_step_min=args.time_step,
        particles=args.particles,
        oil_type=args.oil_type,
        temporal_window_hours=args.window_hours
    )