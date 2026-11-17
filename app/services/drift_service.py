"""Lagrangian spill-drift modelling: backward hindcast and forward forecast.

Two interchangeable engines
---------------------------
``opendrift``
    ``src/opendrift_engine.py`` (OpenOil + NetCDF readers): full oil-weathering
    physics.  Selected when ``current_nc``/``wind_nc`` point at readable files.
``lagrangian``
    The built-in engine in this module.  It integrates the surface velocity

        v = u_current(t) + wind_drift_factor * u_wind(t)

    with an explicit random-walk turbulent-diffusion term
    ``sqrt(2 * D * dt) * N(0, 1)`` per horizontal axis, stepping particles along
    WGS84 geodesics.  ``D`` and the leeway factor come from
    ``config.yaml -> metocean`` (``horizontal_diffusivity``,
    ``wind_drift_factor``) so both engines share one parameter set.  It needs no
    external data files, which keeps the API usable - and the tests hermetic -
    without a metocean archive.

Backward (hindcast) integration negates the deterministic velocity and walks the
metocean time axis in reverse, the standard time-reversal of a negligible-inertia
drift problem.  The diffusion term stays symmetric, so the recovered origin
carries a physically meaningful uncertainty ellipse.

Both engines return the same :class:`~app.schemas.DriftResponse`, and the ellipse
is always fitted with the shared :func:`app.services.geo.compute_error_ellipse`
helper, which mirrors ``src/opendrift_engine.py::_compute_error_ellipse``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry

from app.config import AppSettings
from app.errors import DriftEngineError, InvalidInputError
from app.schemas import (
    DriftPathPoint,
    DriftRequest,
    DriftResponse,
    MetoceanSample,
    UncertaintyEllipse,
    utc_iso,
)
from app.services.geo import (
    GEOD,
    bearing_deg,
    compute_error_ellipse,
    convex_hull_geojson,
    geodesic_distance_m,
    meters_per_degree,
    normalize_ellipse,
    offset_point,
    polygon_centroid_lonlat,
    polygon_from_geojson,
    sample_points_in_geometry,
    to_geographic,
)

logger = logging.getLogger(__name__)

#: Hard cap on integration steps (guards against pathological durations).
#: Backwards-compatibility alias — live runs use
#: ``settings.metocean.max_steps`` (``metocean.max_steps`` / ``OILSPILL_MAX_STEPS``).
#: Tunable via ``config.yaml -> metocean.max_steps`` or ``OILSPILL_MAX_STEPS``;
#: ``DEFAULT_MAX_STEPS`` is the safety default used when neither is provided.
DEFAULT_MAX_STEPS = 20_000
MAX_STEPS = DEFAULT_MAX_STEPS

#: Maximum number of particle end-points echoed back to the client.
#: Backwards-compatibility alias — live runs use
#: ``settings.metocean.max_endpoint_samples``
#: (``metocean.max_endpoint_samples`` / ``OILSPILL_MAX_ENDPOINTS``).
#: Tunable via ``config.yaml -> metocean.max_endpoint_samples`` or
#: ``OILSPILL_MAX_ENDPOINTS``; ``DEFAULT_MAX_ENDPOINT_SAMPLES`` keeps payloads
#: bounded during tests and low-bandwidth operations.
DEFAULT_MAX_ENDPOINT_SAMPLES = 200
MAX_ENDPOINT_SAMPLES = DEFAULT_MAX_ENDPOINT_SAMPLES

#: Seed-region dilation (degrees) used when the source degenerates to a point.
#: Kept as a module-level alias; the live value comes from
#: ``settings.metocean.point_seed_buffer_deg`` (see ``LagrangianDriftEngine``).
POINT_SEED_BUFFER_DEG = 0.01

#: Keys of a metocean sample inside a JSON manifest (see
#: :meth:`MetoceanField.from_environment_json`); anything else in the file
#: (hindcast windows, origin estimates, provenance notes) is ignored.
SAMPLE_KEYS = frozenset(
    {"timestamp_utc", "u_current_mps", "v_current_mps", "u_wind_mps", "v_wind_mps"}
)

#: Timestamp aliases accepted in a metocean manifest.
TIMESTAMP_ALIASES = ("timestamp_utc", "timestamp_sat", "time", "timestamp")


def as_utc(moment: datetime) -> datetime:
    """Return ``moment`` as a timezone-aware UTC datetime."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass
class MetoceanField:
    """Uniform, time-interpolated metocean field shared by all particles.

    The API consumes spatially uniform vectors (incident-averaged currents and
    winds, or one uniform sample of a gridded product).  Samples are linearly
    interpolated in time and clamped outside the supplied range.
    """

    times: List[datetime]
    u_current: np.ndarray
    v_current: np.ndarray
    u_wind: np.ndarray
    v_wind: np.ndarray
    source: str = "config_fallback"
    #: Wind drift factor carried by the source (``None`` = config/request value).
    wind_drift_factor: Optional[float] = None

    @classmethod
    def uniform(cls, sample: Dict[str, float], source: str = "config_fallback") -> "MetoceanField":
        """Time-invariant field built from a single u/v sample (config defaults)."""
        return cls(
            times=[datetime(1970, 1, 1, tzinfo=timezone.utc)],
            u_current=np.array([float(sample["u_current_mps"])]),
            v_current=np.array([float(sample["v_current_mps"])]),
            u_wind=np.array([float(sample["u_wind_mps"])]),
            v_wind=np.array([float(sample["v_wind_mps"])]),
            source=source,
        )

    @classmethod
    def from_samples(
        cls, samples: Sequence[MetoceanSample], default_time: datetime, source: str = "request"
    ) -> "MetoceanField":
        """Time-interpolated field from request samples (missing stamps -> default)."""
        if not samples:
            raise InvalidInputError("metocean sample list is empty")
        times: List[datetime] = []
        for sample in samples:
            if sample.timestamp_utc:
                moment = datetime.fromisoformat(sample.timestamp_utc.replace("Z", "+00:00"))
                times.append(as_utc(moment))
            else:
                times.append(default_time)
        order = np.argsort(np.array([moment.timestamp() for moment in times]), kind="stable")
        return cls(
            times=[times[int(index)] for index in order],
            u_current=np.array([samples[int(i)].u_current_mps for i in order], dtype=float),
            v_current=np.array([samples[int(i)].v_current_mps for i in order], dtype=float),
            u_wind=np.array([samples[int(i)].u_wind_mps for i in order], dtype=float),
            v_wind=np.array([samples[int(i)].v_wind_mps for i in order], dtype=float),
            source=source,
        )

    @classmethod
    def from_environment_json(
        cls,
        payload: Mapping[str, Any],
        *,
        default_time: datetime,
        source: str = "environment_json",
    ) -> "MetoceanField":
        """Time-interpolated field built from a per-incident JSON manifest.

        A metocean manifest (``config.yaml -> metocean.environment_json``, e.g.
        ``data/processed/metocean_environment.json``) is accepted in two shapes:

        * one uniform vector - ``u_current_mps``/``v_current_mps``/``u_wind_mps``/
          ``v_wind_mps`` at the top level, optionally stamped with
          ``timestamp_utc`` (``timestamp_sat``/``time``/``timestamp`` are aliases);
        * a time series - ``samples`` (or ``time_series``/``metocean``) holding a
          list of such vectors, or a mapping keyed by ISO timestamp.  Temporal
          variability is preserved and interpolated exactly like the samples a
          client sends in the request body.

        Optional ``wind_drift_factor`` is carried through so the manifest can
        tune the windage term.  Unknown keys (hindcast windows, cached origin
        estimates, provenance notes) are ignored.

        Raises:
            InvalidInputError: If the manifest carries no usable vectors.
        """
        raw: Any = None
        for key in ("samples", "time_series", "metocean"):
            if payload.get(key) is not None:
                raw = payload[key]
                break
        if isinstance(raw, Mapping):
            raw = [dict(value or {}, timestamp_utc=key) for key, value in raw.items()]
        if isinstance(raw, (list, tuple)):
            entries: List[Any] = list(raw)
        elif "u_current_mps" in payload or "v_current_mps" in payload:
            entries = [payload]
        else:
            raise InvalidInputError(
                "Metocean manifest carries no current vectors: expected "
                "'u_current_mps'/'v_current_mps' at the top level or a 'samples' list."
            )

        samples: List[MetoceanSample] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise InvalidInputError(
                    f"Metocean manifest sample {index} is not a JSON object: {entry!r}"
                )
            clean: Dict[str, Any] = {key: entry[key] for key in SAMPLE_KEYS if key in entry}
            if "timestamp_utc" not in clean:
                for alias in TIMESTAMP_ALIASES[1:]:
                    if entry.get(alias):
                        clean["timestamp_utc"] = entry[alias]
                        break
            missing = [key for key in ("u_current_mps", "v_current_mps") if key not in clean]
            if missing:
                raise InvalidInputError(
                    f"Metocean manifest sample {index} needs 'u_current_mps' and"
                    f" 'v_current_mps'; missing {', '.join(missing)} "
                    f"from keys {sorted(entry)}"
                )
            samples.append(MetoceanSample.model_validate(clean))

        factor = payload.get("wind_drift_factor")
        wind_drift_factor = None if factor is None else float(factor)
        field = cls.from_samples(samples, default_time=default_time, source=source)
        field.wind_drift_factor = wind_drift_factor
        return field

    @property
    def span_seconds(self) -> float:
        """Time covered by the samples (0.0 for a spatially uniform field)."""
        if len(self.times) < 2:
            return 0.0
        return (self.times[-1] - self.times[0]).total_seconds()

    def at(self, moment: datetime) -> Tuple[float, float, float, float]:
        """Interpolated ``(u_current, v_current, u_wind, v_wind)`` at ``moment``."""
        if len(self.times) == 1:
            return (
                float(self.u_current[0]),
                float(self.v_current[0]),
                float(self.u_wind[0]),
                float(self.v_wind[0]),
            )
        stamps = np.array([instant.timestamp() for instant in self.times])
        target = as_utc(moment).timestamp()
        if target <= stamps[0]:
            index, weight = 0, 0.0
        elif target >= stamps[-1]:
            index, weight = len(stamps) - 2, 1.0
        else:
            index = int(np.searchsorted(stamps, target, side="right") - 1)
            span = stamps[index + 1] - stamps[index]
            weight = 0.0 if span <= 0 else (target - stamps[index]) / span

        def blend(values: np.ndarray) -> float:
            return float(values[index] + weight * (values[index + 1] - values[index]))

        return (
            blend(self.u_current),
            blend(self.v_current),
            blend(self.u_wind),
            blend(self.v_wind),
        )


def _circle_polygon(center_lon: float, center_lat: float, radius_m: float, vertices: int = 48):
    """Approximate a metric-radius disc around a WGS84 point as a shapely polygon."""
    coords: List[Tuple[float, float]] = []
    for index in range(vertices):
        angle = 2.0 * math.pi * index / vertices
        east = radius_m * math.cos(angle)
        north = radius_m * math.sin(angle)
        lon, lat = offset_point(center_lon, center_lat, east, north)
        coords.append((lon, lat))
    coords.append(coords[0])
    return Polygon(coords)


def _advance(
    lons: np.ndarray, lats: np.ndarray, east_m: np.ndarray, north_m: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised geodesic step by metric east/north offsets, lon/lat wrapped."""
    distance = np.hypot(east_m, north_m)
    heading = np.degrees(np.arctan2(east_m, north_m)) % 360.0
    out_lon, out_lat, _ = GEOD.fwd(lons, lats, heading, distance)
    wrapped_lon = np.empty_like(out_lon)
    wrapped_lat = np.empty_like(out_lat)
    for index, (lon, lat) in enumerate(zip(out_lon.tolist(), out_lat.tolist())):
        wrapped_lon[index], wrapped_lat[index] = to_geographic(lon, lat)
    return wrapped_lon, wrapped_lat


@dataclass
class LagrangianResult:
    """Terminal state of a built-in Lagrangian ensemble integration."""

    direction: str
    start_time: datetime
    end_time: datetime
    lons: np.ndarray
    lats: np.ndarray
    seed_lon: float
    seed_lat: float
    centroid_path: List[Tuple[datetime, float, float]]
    mean_speed_mps: float
    max_speed_mps: float
    steps: int
    requested_particles: int
    dropped_particles: int


class LagrangianDriftEngine:
    """Dependency-free ensemble drift integrator (see module docstring)."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def max_steps(self) -> int:
        """Configured integration-step cap (never below 1)."""
        return max(1, int(getattr(self.settings.metocean, "max_steps", DEFAULT_MAX_STEPS)))

    @property
    def max_endpoints(self) -> int:
        """Configured response payload cap (never below 1)."""
        return max(
            1,
            int(getattr(
                self.settings.metocean, "max_endpoint_samples", DEFAULT_MAX_ENDPOINT_SAMPLES
            )),
        )

    @property
    def seed_buffer_deg(self) -> float:
        """Configured point-seed dilation (degrees)."""
        try:
            return float(getattr(self.settings.metocean, "point_seed_buffer_deg", 0.01))
        except (TypeError, ValueError):
            return 0.01

    # ---------------------------------------------------------------- seeding
    def seed_positions(
        self, request: DriftRequest, particles: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Return ``(lons, lats, seed_centroid_lon, seed_centroid_lat)``.

        Points are drawn uniformly inside the slick polygon when one is supplied,
        otherwise inside a disc of ``source.source_radius_m`` around the point.
        """
        source = request.source
        if source.slick_geojson is not None:
            try:
                geometry = polygon_from_geojson(source.slick_geojson)
            except (ValueError, TypeError) as exc:
                raise InvalidInputError(
                    "source.slick_geojson does not contain a usable geometry",
                    details={"reason": str(exc)},
                ) from exc
            pairs = sample_points_in_geometry(
                geometry, particles, rng, buffer_deg=self.seed_buffer_deg
            )
            centroid_lon, centroid_lat = polygon_centroid_lonlat(geometry)
        else:
            centroid_lon = float(source.source_lon)  # type: ignore[arg-type]
            centroid_lat = float(source.source_lat)  # type: ignore[arg-type]
            disc = _circle_polygon(centroid_lon, centroid_lat, float(source.source_radius_m))
            pairs = sample_points_in_geometry(disc, particles, rng)
        lons = np.array([pair[0] for pair in pairs], dtype="float64")
        lats = np.array([pair[1] for pair in pairs], dtype="float64")
        return lons, lats, centroid_lon, centroid_lat

    # ------------------------------------------------------------ integration
    def integrate(
        self,
        *,
        lons: np.ndarray,
        lats: np.ndarray,
        seed_lon: float,
        seed_lat: float,
        start_time: datetime,
        duration_hours: float,
        time_step_minutes: int,
        direction: str,
        field: MetoceanField,
        wind_drift_factor: float,
        diffusivity: float,
        seed: int,
    ) -> LagrangianResult:
        """Euler-integrate the particle ensemble forward or backward in time.

        Args:
            lons: Particle longitudes at ``start_time`` (degrees).
            lats: Particle latitudes at ``start_time`` (degrees).
            seed_lon: Seed-region centroid longitude (bearing reference).
            seed_lat: Seed-region centroid latitude (bearing reference).
            start_time: Satellite acquisition time (backward) or release time.
            duration_hours: Integration window length (positive).
            time_step_minutes: Output/integration step length (positive).
            direction: ``"backward"`` (hindcast) or ``"forward"`` (forecast).
            field: Time-interpolated metocean field.
            wind_drift_factor: Surface-leeway fraction of the wind vector.
            diffusivity: Horizontal turbulent diffusivity ``D`` (m^2/s).
            seed: PRNG seed; identical inputs give identical output.

        Returns:
            A :class:`LagrangianResult` holding terminal positions, the
            ensemble-centroid path and realised drift speeds.

        Raises:
            DriftEngineError: If the requested window needs more than the
                configured ``max_steps``, or if every particle is lost.
        """
        dt = float(time_step_minutes) * 60.0
        steps = int(max(1, math.ceil(duration_hours * 3600.0 / dt)))
        max_steps = self.max_steps
        if steps > max_steps:
            raise DriftEngineError(
                "Drift integration too long for the requested time step",
                details={"steps_required": steps, "max_steps": max_steps},
            )
        sign = -1.0 if direction == "backward" else 1.0
        rng = np.random.default_rng(int(seed))
        sigma = math.sqrt(2.0 * max(float(diffusivity), 0.0) * dt)

        live = np.isfinite(lons) & np.isfinite(lats)
        dropped = int(lons.size - live.sum())
        m_per_deg_lon, m_per_deg_lat = meters_per_degree(seed_lon, seed_lat)
        velocities: List[float] = []
        realised_speed = 0.0
        path: List[Tuple[datetime, float, float]] = [
            (start_time, _median(lons[live]), _median(lats[live]))
        ]

        for step in range(1, steps + 1):
            moment = start_time + timedelta(seconds=sign * step * dt)
            u_current, v_current, u_wind, v_wind = field.at(moment)
            east = u_current + wind_drift_factor * u_wind
            north = v_current + wind_drift_factor * v_wind
            velocities.append(float(math.hypot(east, north)))

            count = int(lons.size)
            east_step = np.full(count, sign * east * dt)
            north_step = np.full(count, sign * north * dt)
            if sigma > 0.0:
                east_step += rng.normal(0.0, sigma, size=count)
                north_step += rng.normal(0.0, sigma, size=count)

            previous_lon, previous_lat = lons.copy(), lats.copy()
            lons, lats = _advance(lons, lats, east_step, north_step)
            finite = np.isfinite(lons) & np.isfinite(lats)
            if not bool(finite.all()):
                dropped += int(finite.size - finite.sum())
                lons, lats = np.where(finite, lons, previous_lon), np.where(
                    finite, lats, previous_lat
                )
            step_speed = np.hypot(
                (lons - previous_lon) * m_per_deg_lon, (lats - previous_lat) * m_per_deg_lat
            )
            if step_speed.size and finite.any():
                realised_speed = max(realised_speed, float(np.max(step_speed)) / dt)
            live = finite
            path.append((moment, _median(lons[live]), _median(lats[live])))

        if not bool(live.any()):
            raise DriftEngineError("All particles were lost during the drift integration")

        end_time = start_time + timedelta(seconds=sign * steps * dt)
        mean_speed = float(sum(velocities) / len(velocities)) if velocities else 0.0
        logger.debug(
            "Lagrangian %s run: %d steps, %d/%d particles live, mean speed %.3f m/s",
            direction,
            steps,
            int(live.sum()),
            lons.size,
            mean_speed,
        )
        return LagrangianResult(
            direction=direction,
            start_time=start_time,
            end_time=end_time,
            lons=lons[live],
            lats=lats[live],
            seed_lon=seed_lon,
            seed_lat=seed_lat,
            centroid_path=path,
            mean_speed_mps=mean_speed,
            max_speed_mps=max(realised_speed, max(velocities) if velocities else 0.0),
            steps=steps,
            requested_particles=int(lons.size),
            dropped_particles=dropped,
        )


def _median(values: np.ndarray) -> float:
    """Median of ``values`` with a safe ``0.0`` fallback for empty arrays."""
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def parse_time(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime."""
    try:
        moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidInputError(f"Unparseable ISO-8601 timestamp: {raw!r}") from exc
    return as_utc(moment)


class DriftService:
    """Route drift requests to OpenDrift or the built-in Lagrangian engine.

    Metocean forcing is resolved in strict priority order:
      1. ``request``   - inline samples supplied in the API payload
      2. ``netcdf``    - per-request override (``metocean_paths``) or the
                         config ``current_nc``/``wind_nc`` (OpenDrift only)
      3. ``environment_json`` - per-incident JSON manifest
      4. ``config_fallback`` - static config.yaml constants
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.lagrangian = LagrangianDriftEngine(settings)

    # ----------------------------------------------------------- public entry
    def run_hindcast(
        self,
        request: DriftRequest,
        *,
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> DriftResponse:
        """Trace a slick backwards from sensing time to its likely origin."""
        return self._run(request, direction="backward", metocean_paths=metocean_paths)

    def run_forecast(
        self,
        request: DriftRequest,
        *,
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> DriftResponse:
        """Propagate a release forward to predict the future slick extent."""
        return self._run(request, direction="forward", metocean_paths=metocean_paths)

    def _run(
        self,
        request: DriftRequest,
        *,
        direction: str,
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> DriftResponse:
        warnings: List[str] = []
        start = parse_time(request.start_time_utc)
        wind_drift_factor, diffusivity, confidence = self._physics(request)
        field = self._resolve_field(request, start, warnings, metocean_paths=metocean_paths)
        if field.wind_drift_factor is not None and request.wind_drift_factor is None:
            wind_drift_factor = field.wind_drift_factor
            warnings.append(
                f"Wind drift factor {wind_drift_factor:.4f} came from the metocean "
                "field instead of the configured value."
            )
        engine = self._resolve_engine(request, direction, warnings, metocean_paths=metocean_paths)

        if engine == "opendrift":
            try:
                return self._run_opendrift(
                    request,
                    direction=direction,
                    start=start,
                    wind_drift_factor=wind_drift_factor,
                    diffusivity=diffusivity,
                    confidence=confidence,
                    warnings=warnings,
                    metocean_paths=metocean_paths,
                )
            except DriftEngineError:
                if request.engine == "opendrift":
                    raise
                warnings.append(
                    "OpenDrift run failed; results come from the built-in Lagrangian engine."
                )
                logger.exception("OpenDrift run failed, falling back to Lagrangian engine")
        return self._run_lagrangian(
            request,
            direction=direction,
            start=start,
            field=field,
            wind_drift_factor=wind_drift_factor,
            diffusivity=diffusivity,
            confidence=confidence,
            warnings=warnings,
        )

    # ------------------------------------------------------------- resolution
    def _physics(self, request: DriftRequest) -> Tuple[float, float, float]:
        """Resolve ``(wind_drift_factor, diffusivity, confidence)`` for a request."""
        defaults = self.settings.metocean
        diffusivity = (
            defaults.horizontal_diffusivity
            if request.horizontal_diffusivity_m2_s is None
            else float(request.horizontal_diffusivity_m2_s)
        )
        return (
            float(request.wind_drift_factor or defaults.wind_drift_factor),
            float(diffusivity),
            float(request.confidence_level or defaults.confidence_level),
        )

    def _resolve_field(
        self,
        request: DriftRequest,
        start: datetime,
        warnings: List[str],
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> MetoceanField:
        """Build the metocean field for the Lagrangian engine.

        Precedence: request samples -> per-incident JSON manifest (metocean_paths
        override, then configured manifest) -> config fallback constants.
        """
        if request.metocean:
            field = MetoceanField.from_samples(request.metocean, default_time=start)
            self._warn_coverage(field, request.duration_hours, warnings)
            return field
        field = self._environment_field(request, start, warnings, metocean_paths=metocean_paths)
        if field is not None:
            return field
        defaults = self.settings.metocean
        warnings.append(
            "No metocean samples supplied: using the config fallback field for "
            f"'{defaults.fallback_region}' (current=({defaults.fallback_current_u_mps}, "
            f"{defaults.fallback_current_v_mps}) m/s, wind=("
            f"{defaults.fallback_wind_u_mps}, {defaults.fallback_wind_v_mps}) m/s)."
        )
        return MetoceanField.uniform(defaults.fallback_sample())

    def _environment_field(
        self,
        request: DriftRequest,
        start: datetime,
        warnings: List[str],
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> Optional[MetoceanField]:
        """Field from the per-incident JSON manifest, or ``None`` when unusable.

        Resolution order: metocean_paths.environment_json (per-request override from
        the live fetch), OILSPILL_METOCEAN_JSON env var, config
        ``metocean.environment_json``.  A broken manifest never fails a request.
        """
        overrides = metocean_paths or {}
        manifest_str = overrides.get("environment_json")
        if manifest_str is None:
            manifest_str = os.getenv("OILSPILL_METOCEAN_JSON")
        if manifest_str is None:
            manifest = self.settings.metocean.environment_json
            manifest_str = str(manifest) if manifest else None
        if not manifest_str:
            return None
        path = self._resolve_data_path(manifest_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reason = " ".join(str(exc).split())
            logger.warning("Metocean manifest %s is unreadable: %s", path, reason)
            warnings.append(
                f"Metocean manifest {path} could not be read ({reason}); the config "
                "fallback field was used instead."
            )
            return None
        if not isinstance(payload, dict):
            warnings.append(
                f"Metocean manifest {path} does not hold a JSON object; the config "
                "fallback field was used instead."
            )
            return None
        try:
            field = MetoceanField.from_environment_json(payload, default_time=start)
        except (InvalidInputError, TypeError, ValueError) as exc:
            reason = " ".join(str(exc).split())
            logger.warning("Metocean manifest %s is invalid: %s", path, reason)
            warnings.append(
                f"Metocean manifest {path} is invalid ({reason}); the config fallback "
                "field was used instead."
            )
            return None
        warnings.append(
            f"Metocean field taken from manifest {path} ({len(field.times)} sample(s) "
            f"over {field.span_seconds / 3600.0:.2f} h)."
        )
        self._warn_coverage(field, request.duration_hours, warnings, start=start)
        return field

    @staticmethod
    def _warn_coverage(
        field: MetoceanField,
        duration_hours: float,
        warnings: List[str],
        start: Optional[datetime] = None,
    ) -> None:
        """Report every way the sampled window fails to cover the request."""
        if field.span_seconds + 1.0 < duration_hours * 3600.0:
            warnings.append(
                "Metocean samples cover "
                f"{field.span_seconds / 3600.0:.2f} h of the "
                f"{duration_hours:.2f} h window; values are clamped "
                "outside the sampled range."
            )
        if start is None:
            return
        # One typical integration step of slack, so a sample stamped exactly at
        # the acquisition time is never reported as stale.
        margin = 900.0
        window = (field.times[0].timestamp(), field.times[-1].timestamp())
        if not window[0] - margin <= start.timestamp() <= window[1] + margin:
            warnings.append(
                f"Requested start {utc_iso(start)} lies outside the sampled metocean "
                f"window {utc_iso(field.times[0])} to {utc_iso(field.times[-1])}; "
                "values are clamped to the nearest sample."
            )

    def _resolve_engine(
        self,
        request: DriftRequest,
        direction: str,
        warnings: List[str],
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """Pick ``'opendrift'`` or ``'lagrangian'`` for this request.

        NetCDF forcing comes from ``metocean_paths`` (per-request override from the
        live fetch) when supplied, otherwise from ``request.current_nc``/``wind_nc``,
        otherwise the configured config.yaml paths.
        """
        overrides = metocean_paths or {}
        cur_nc = overrides.get("current_nc") or request.current_nc
        wnd_nc = overrides.get("wind_nc") or request.wind_nc
        readable = all(
            path and self._resolve_data_path(path).is_file()
            for path in (cur_nc, wnd_nc)
        )
        if request.engine == "lagrangian":
            return "lagrangian"
        if request.engine == "opendrift":
            if not readable:
                raise DriftEngineError(
                    "OpenDrift requested but metocean files are not readable",
                    details={"current_nc": cur_nc, "wind_nc": wnd_nc},
                )
            if direction == "backward" and request.source.slick_geojson is None:
                raise DriftEngineError(
                    "OpenDrift hindcast needs source.slick_geojson (a polygon to seed)"
                )
            return "opendrift"
        if readable and direction == "forward":
            return "opendrift"
        if readable and direction == "backward" and request.source.slick_geojson is not None:
            return "opendrift"
        if any((cur_nc, wnd_nc)):
            warnings.append(
                "Configured NetCDF metocean files were not usable; the built-in "
                "Lagrangian engine was used instead."
            )
        return "lagrangian"

    def _resolve_data_path(self, raw: str) -> Path:
        """Resolve a server-side data path against the project root."""
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else self.settings.project_root / candidate

    def _weathering_model(self, warnings: List[str]) -> str:
        """Configured OpenOil weathering model, downgraded when unsupported.

        ``adios`` needs the ``simple-adios`` package.  Without it OpenOil fails
        while it is being built, so under ``engine=auto`` the request would
        silently drop to the built-in Lagrangian engine; stating the downgrade in
        the response keeps the honest label on the result instead.
        """
        model = str(self.settings.metocean.weathering_model or "none")
        if model.lower() in {"adios", "adios2"} and importlib.util.find_spec(
            "simple_adios"
        ) is None:
            logger.warning("simple_adios is unavailable: weathering_model -> 'none'")
            warnings.append(
                "The ADIOS weathering library is not installed, so OpenDrift runs "
                "with weathering disabled ('none'): transport is modelled but "
                "oil-fate processes (evaporation, emulsification, dispersion) are not."
            )
            return "none"
        return model

    # -------------------------------------------------------------- executors
    def _run_lagrangian(
        self,
        request: DriftRequest,
        *,
        direction: str,
        start: datetime,
        field: MetoceanField,
        wind_drift_factor: float,
        diffusivity: float,
        confidence: float,
        warnings: List[str],
    ) -> DriftResponse:
        """Integrate the built-in ensemble and assemble the API response."""
        rng = np.random.default_rng(int(request.random_seed))
        lons, lats, seed_lon, seed_lat = self.lagrangian.seed_positions(
            request, int(request.particles), rng
        )
        result = self.lagrangian.integrate(
            lons=lons,
            lats=lats,
            seed_lon=seed_lon,
            seed_lat=seed_lat,
            start_time=start,
            duration_hours=float(request.duration_hours),
            time_step_minutes=int(request.time_step_minutes),
            direction=direction,
            field=field,
            wind_drift_factor=wind_drift_factor,
            diffusivity=diffusivity,
            seed=int(request.random_seed),
        )
        if result.dropped_particles:
            warnings.append(
                f"{result.dropped_particles} particle(s) produced non-finite positions "
                "and were held at their last valid location."
            )
        if field.span_seconds and field.span_seconds + 1.0 < request.duration_hours * 3600.0:
            logger.debug("Integration ran beyond the sampled metocean window (%.1f h)",
                         field.span_seconds / 3600.0)

        terminal_lon, terminal_lat = _median(result.lons), _median(result.lats)
        fit = compute_error_ellipse(result.lons, result.lats, confidence)
        path_points = [
            DriftPathPoint(timestamp_utc=utc_iso(moment), lon=round(lon, 6), lat=round(lat, 6))
            for moment, lon, lat in _subsample(
                result.centroid_path, self.lagrangian.max_endpoints
            )
        ]
        endpoints = [
            [round(float(lon), 6), round(float(lat), 6)]
            for lon, lat in _subsample(
                list(zip(result.lons.tolist(), result.lats.tolist())),
                self.lagrangian.max_endpoints,
            )
        ]
        backward = direction == "backward"
        return DriftResponse(
            engine="lagrangian",
            direction=direction,  # type: ignore[arg-type]
            start_time_utc=utc_iso(result.start_time),
            end_time_utc=utc_iso(result.end_time),
            duration_hours=round(float(request.duration_hours), 4),
            time_step_minutes=int(request.time_step_minutes),
            particles=result.requested_particles,
            particles_in_domain=int(result.lons.size),
            wind_drift_factor=wind_drift_factor,
            horizontal_diffusivity_m2_s=diffusivity,
            metocean_source=field.source,
            drift_bearing_deg=round(bearing_deg(seed_lon, seed_lat, terminal_lon, terminal_lat), 3),
            mean_drift_speed_mps=round(result.mean_speed_mps, 5),
            max_drift_speed_mps=round(result.max_speed_mps, 5),
            origin_lon=round(terminal_lon, 6) if backward else None,
            origin_lat=round(terminal_lat, 6) if backward else None,
            endpoint_lon=None if backward else round(terminal_lon, 6),
            endpoint_lat=None if backward else round(terminal_lat, 6),
            inferred_discharge_time_utc=utc_iso(result.end_time) if backward else None,
            centroid_path=path_points,
            particle_endpoints=endpoints,
            envelope_geojson=convex_hull_geojson(
                result.lons, result.lats, layer="particle_envelope"
            ),
            uncertainty=_ellipse_from_fit(fit, confidence),
            warnings=warnings,
        )

    def _run_opendrift(
        self,
        request: DriftRequest,
        *,
        direction: str,
        start: datetime,
        wind_drift_factor: float,
        diffusivity: float,
        confidence: float,
        warnings: List[str],
        metocean_paths: Optional[Dict[str, str]] = None,
    ) -> DriftResponse:
        """Delegate to ``src/opendrift_engine.OpenDriftHindcastEngine``."""
        try:
            from src.opendrift_engine import OpenDriftHindcastEngine
        except Exception as exc:  # ImportError or a broken optional dependency
            raise DriftEngineError(
                f"OpenDrift engine is unavailable in this environment: {exc}"
            ) from exc

        overrides = metocean_paths or {}
        current_nc = overrides.get("current_nc") or request.current_nc
        wind_nc = overrides.get("wind_nc") or request.wind_nc

        seed_path, seed_lon, seed_lat = self._write_seed_geojson(request)
        output_dir = self.settings.artifact_dir / "opendrift"
        weathering_notes: List[str] = []
        try:
            engine = OpenDriftHindcastEngine(
                current_data_path=str(self._resolve_data_path(str(current_nc))),
                wind_data_path=str(self._resolve_data_path(str(wind_nc))),
                wave_data_path=(
                    str(self._resolve_data_path(request.wave_nc)) if request.wave_nc else None
                ),
                oil_type=request.oil_type or self.settings.metocean.oil_type,
                horizontal_diffusivity=diffusivity,
                wind_drift_factor=wind_drift_factor,
                weathering_model=self._weathering_model(weathering_notes),
                release_seed_buffer_deg=POINT_SEED_BUFFER_DEG,
                confidence_level=confidence,
                output_dir=str(output_dir),
            )
            if direction == "backward":
                summary = engine.run_backward_hindcast(
                    slick_geojson_path=str(seed_path),
                    satellite_time=start,
                    hindcast_hours=float(request.duration_hours),
                    time_step_minutes=int(request.time_step_minutes),
                    num_particles=int(request.particles),
                )
            else:
                summary = engine.run_forward_forecast(
                    release_point=None if seed_path else (seed_lon, seed_lat),
                    release_geojson_path=str(seed_path) if seed_path else None,
                    release_time=start,
                    forecast_hours=float(request.duration_hours),
                    time_step_minutes=int(request.time_step_minutes),
                    num_particles=int(request.particles),
                )
        except DriftEngineError:
            raise
        except Exception as exc:  # OpenDrift raises many generic exception types
            raise DriftEngineError(f"OpenDrift {direction} run failed: {exc}") from exc
        finally:
            if seed_path:
                Path(seed_path).unlink(missing_ok=True)

        # Only reported once the run really happened: a failed OpenDrift attempt
        # that falls back to the Lagrangian engine must not inherit them.
        warnings.extend(weathering_notes)
        backward = direction == "backward"
        fit = summary.get("uncertainty") or {}
        envelope_path = summary.get("origin_geojson") or summary.get("forecast_geojson")
        envelope: Dict[str, Any] = {}
        if envelope_path and Path(envelope_path).is_file():
            try:
                with open(envelope_path, encoding="utf-8") as handle:
                    envelope = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read OpenDrift envelope %s: %s", envelope_path, exc)
        # OpenOil puts no centre in the ellipse: a hindcast reports the recovered
        # origin, a forecast the median position of the surviving ensemble.  The
        # envelope geometry is the last resort before the seed itself.
        center_lon = fit.get("center_lon")
        center_lat = fit.get("center_lat")
        if center_lon is None or center_lat is None:
            key = "origin" if backward else "forecast"
            center_lon = summary.get(f"{key}_lon")
            center_lat = summary.get(f"{key}_lat")
        if center_lon is None or center_lat is None:
            center_lon, center_lat = _envelope_centroid(envelope) or (seed_lon, seed_lat)
        seconds = max(float(request.duration_hours) * 3600.0, 1.0)
        distance = geodesic_distance_m(seed_lon, seed_lat, center_lon, center_lat)
        realised_speed = distance / seconds
        warnings.append(
            "OpenDrift telemetry does not expose per-step velocities; mean and max "
            "drift speeds report the realised ensemble-centroid speed."
        )
        # A bearing comes with hindcast telemetry only, so a forecast takes the
        # geodesic direction of the ensemble-centroid displacement.
        bearing = summary.get("drift_bearing_deg")
        if bearing is None:
            bearing = bearing_deg(seed_lon, seed_lat, center_lon, center_lat)
            if not math.isfinite(bearing):  # coincident seed and predicted centre
                bearing = 0.0
            warnings.append(
                "OpenDrift forecast telemetry carries no bearing: the value is the "
                "geodesic direction from the release centroid to the predicted centre."
            )
        return DriftResponse(
            engine="opendrift",
            direction=direction,  # type: ignore[arg-type]
            start_time_utc=utc_iso(start),
            end_time_utc=str(
                summary.get("inferred_t0")
                or summary.get("forecast_end_time")
                or utc_iso(start)
            ),
            duration_hours=round(float(request.duration_hours), 4),
            time_step_minutes=int(request.time_step_minutes),
            particles=int(request.particles),
            particles_in_domain=int(summary.get("particles_tracked", request.particles)),
            wind_drift_factor=wind_drift_factor,
            horizontal_diffusivity_m2_s=diffusivity,
            metocean_source="netcdf",
            drift_bearing_deg=float(bearing) % 360.0,
            mean_drift_speed_mps=round(realised_speed, 5),
            max_drift_speed_mps=round(realised_speed, 5),
            origin_lon=float(summary["origin_lon"]) if backward else None,
            origin_lat=float(summary["origin_lat"]) if backward else None,
            endpoint_lon=None if backward else center_lon,
            endpoint_lat=None if backward else center_lat,
            inferred_discharge_time_utc=str(summary.get("inferred_t0")) if backward else None,
            centroid_path=[],
            particle_endpoints=[],
            envelope_geojson=envelope,
            uncertainty=_ellipse_from_fit(fit, confidence),
            warnings=warnings,
        )

    def _write_seed_geojson(self, request: DriftRequest) -> Tuple[Optional[Path], float, float]:
        """Materialise the seed polygon for OpenDrift as a temporary GeoJSON file.

        Returns:
            ``(path_or_None, seed_lon, seed_lat)``; ``path`` is ``None`` when the
            request seeds from a point - a forward run releases from that point,
            while a backward run buffers it first (see ``_run_opendrift``).
        """
        source = request.source
        if source.slick_geojson is None:
            return None, float(source.source_lon), float(source.source_lat)  # type: ignore[arg-type]
        geometry = polygon_from_geojson(source.slick_geojson)
        seed_lon, seed_lat = polygon_centroid_lonlat(geometry)
        return self._write_seed_geometry(geometry), seed_lon, seed_lat

    def _write_seed_geometry(self, geometry: BaseGeometry) -> Path:
        """Write ``geometry`` as a one-feature GeoJSON file and return its path."""
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"layer": "api_seed_polygon"},
                    "geometry": mapping(geometry),
                }
            ],
        }
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".geojson",
            delete=False,
            dir=str(self.settings.artifact_dir),
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle)
            return Path(handle.name)


# --------------------------------------------------------------------------- #
#  Module helpers
# --------------------------------------------------------------------------- #
def _envelope_centroid(envelope: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Centroid of an OpenDrift envelope, or ``None`` when it encloses no area."""
    try:
        return polygon_centroid_lonlat(polygon_from_geojson(envelope))
    except ValueError:
        return None


def _ellipse_from_fit(fit: Dict[str, Any], confidence: float) -> UncertaintyEllipse:
    """Convert a raw ellipse fit (or OpenDrift telemetry) into the API schema."""
    normalized = normalize_ellipse(
        float(fit.get("semi_major_axis_meters", 1.0)),
        float(fit.get("semi_minor_axis_meters", 1.0)),
        float(fit.get("orientation_deg", 0.0)),
        float(fit.get("confidence_level", confidence)),
        fit.get("center_lon"),
        fit.get("center_lat"),
    )
    return UncertaintyEllipse(
        semi_major_axis_meters=round(normalized["semi_major_axis_meters"], 3),
        semi_minor_axis_meters=round(normalized["semi_minor_axis_meters"], 3),
        orientation_deg=round(normalized["orientation_deg"], 3),
        confidence_level=normalized["confidence_level"],
        center_lon=normalized["center_lon"],
        center_lat=normalized["center_lat"],
    )


def _subsample(items: Sequence[Any], limit: int) -> List[Any]:
    """Evenly thin a sequence to at most ``limit`` items, always keeping the last."""
    sequence = list(items)
    if len(sequence) <= limit:
        return sequence
    step = math.ceil(len(sequence) / limit)
    picked = sequence[::step][:limit]
    if picked and picked[-1] != sequence[-1]:
        picked[-1] = sequence[-1]
    return picked








