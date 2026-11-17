"""Incident-specific metocean acquisition for OpenDrift / Lagrangian drift.

Fetches real ocean currents (Copernicus Marine) and surface winds (ERA5 via
ECMWF CDS), enforces CF-Convention metadata so OpenDrift's
``reader_netCDF_CF_generic`` accepts the files, and builds the
``data/processed/metocean_environment.json`` time-series manifest consumed by
the built-in Lagrangian fallback engine.

``copernicusmarine`` and ``cdsapi`` are imported lazily inside the fetch
routines so this module imports cleanly even when those optional packages are
not installed; a live-fetch failure degrades gracefully to the config fallback
field.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger("environment_fetcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

__all__ = ["fetch_incident_metocean", "fetch_wavewatch3_waves", "build_environment_manifest"]


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
#  Dataset acquisition (lazy optional imports)
# --------------------------------------------------------------------------- #
def fetch_copernicus_currents(
    bbox: Tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    output_path: Path,
    dataset_id: str = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
) -> Path:
    """Download surface ocean currents (uo, vo) in NetCDF via Copernicus Marine."""
    try:
        import copernicusmarine  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "copernicusmarine is required for live current downloads; "
            "install it or disable fetch_live_metocean."
        ) from exc

    min_lon, min_lat, max_lon, max_lat = bbox
    logger.info(
        "Fetching Copernicus currents [%s] for %s -> %s", dataset_id, start_dt, end_dt
    )
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["uo", "vo"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0.0,
        maximum_depth=1.5,
        output_directory=str(output_path.parent),
        output_filename=output_path.name,
        force_download=True,
    )
    _ensure_cf_standard_names(
        output_path,
        var_map={"uo": "eastward_sea_water_velocity", "vo": "northward_sea_water_velocity"},
    )
    return output_path


def fetch_era5_winds(
    bbox: Tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    output_path: Path,
) -> Path:
    """Download hourly ERA5 10m wind components (u10, v10) via CDS API."""
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cdsapi is required for live ERA5 wind downloads; install it or "
            "disable fetch_live_metocean."
        ) from exc

    min_lon, min_lat, max_lon, max_lat = bbox
    # CDS area format: [North, West, South, East]
    cds_area = [max_lat, min_lon, min_lat, max_lon]

    days_span = (end_dt.date() - start_dt.date()).days + 1
    dates = [start_dt.date() + timedelta(days=i) for i in range(days_span)]
    years = sorted({d.strftime("%Y") for d in dates})
    months = sorted({d.strftime("%m") for d in dates})
    days = sorted({d.strftime("%d") for d in dates})
    hours = [f"{h:02d}:00" for h in range(24)]

    logger.info("Fetching ERA5 10m winds for %s -> %s", start_dt, end_dt)
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": cds_area,
        },
        str(output_path),
    )
    _ensure_cf_standard_names(
        output_path, var_map={"u10": "x_wind", "v10": "y_wind"}
    )
    return output_path


def fetch_wavewatch3_waves(
    bbox: Tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    output_path: Path,
    dataset_id: str = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
) -> Path:
    """Download surface wave Stokes drift (VSDX, VSDY) via WaveWatch III / Copernicus Marine."""
    try:
        import copernicusmarine  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "copernicusmarine is required for live wave downloads; "
            "install it or disable fetch_live_metocean."
        ) from exc

    min_lon, min_lat, max_lon, max_lat = bbox
    logger.info("Fetching WaveWatch III waves [%s] for %s -> %s", dataset_id, start_dt, end_dt)
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["VSDX", "VSDY"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        output_directory=str(output_path.parent),
        output_filename=output_path.name,
        force_download=True,
    )
    _ensure_cf_standard_names(
        output_path,
        var_map={
            "VSDX": "sea_surface_wave_stokes_drift_x_velocity",
            "VSDY": "sea_surface_wave_stokes_drift_y_velocity",
            "u_stokes": "sea_surface_wave_stokes_drift_x_velocity",
            "v_stokes": "sea_surface_wave_stokes_drift_y_velocity",
        },
    )
    return output_path


def _ensure_cf_standard_names(nc_path: Path, var_map: Dict[str, str]) -> None:
    """Inject CF standard_name attributes required by OpenDrift's CF reader."""
    ds = xr.open_dataset(nc_path)
    modified = False
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        modified = True

    # Standard Stokes drift aliases for WaveWatch III
    stokes_aliases = {
        "VSDX": "sea_surface_wave_stokes_drift_x_velocity",
        "VSDY": "sea_surface_wave_stokes_drift_y_velocity",
        "u_stokes": "sea_surface_wave_stokes_drift_x_velocity",
        "v_stokes": "sea_surface_wave_stokes_drift_y_velocity",
    }
    full_var_map = {**stokes_aliases, **var_map}

    for var_name, standard_name in full_var_map.items():
        if var_name in ds.variables and ds[var_name].attrs.get("standard_name") != standard_name:
            ds[var_name].attrs["standard_name"] = standard_name
            modified = True
    if modified:
        tmp_path = nc_path.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp_path)
        ds.close()
        tmp_path.replace(nc_path)
    else:
        ds.close()


# --------------------------------------------------------------------------- #
#  Manifest assembly
# --------------------------------------------------------------------------- #
def build_environment_manifest(
    current_nc: Path,
    wind_nc: Path,
    center_lon: float,
    center_lat: float,
    manifest_path: Path,
    wind_drift_factor: float = 0.03,
    wave_nc: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sample NetCDF grids at the slick centroid to build the Lagrangian manifest.

    When wave_nc with Stokes drift variables (VSDX/VSDY or u_stokes/v_stokes)
    is provided, samples are enriched with Stokes drift velocities; otherwise,
    cleanly indicates fallback to the standard leeway formulation.
    """
    ds_cur = xr.open_dataset(current_nc)
    ds_wnd = xr.open_dataset(wind_nc)
    ds_wav = xr.open_dataset(wave_nc) if (wave_nc and Path(wave_nc).exists()) else None

    cur_pt = ds_cur.sel(longitude=center_lon, latitude=center_lat, method="nearest")
    if "depth" in cur_pt.dims:
        cur_pt = cur_pt.isel(depth=0)

    wnd_pt = ds_wnd.sel(longitude=center_lon, latitude=center_lat, method="nearest")
    cur_interp = cur_pt.interp(
        time=wnd_pt["time"], method="linear", kwargs={"fill_value": "extrapolate"}
    )

    wav_interp = None
    if ds_wav is not None:
        try:
            wav_pt = ds_wav.sel(longitude=center_lon, latitude=center_lat, method="nearest")
            wav_interp = wav_pt.interp(
                time=wnd_pt["time"], method="linear", kwargs={"fill_value": "extrapolate"}
            )
        except Exception as exc:
            logger.warning("Could not interpolate WaveWatch III grid at (%f, %f): %s", center_lon, center_lat, exc)
            wav_interp = None

    samples: list[dict[str, Any]] = []
    has_stokes = False
    for idx, t_val in enumerate(wnd_pt["time"].values):
        ts_iso = np.datetime_as_string(t_val, unit="s") + "Z"
        u_c = float(np.nan_to_num(cur_interp["uo"].isel(time=idx).values, nan=0.0))
        v_c = float(np.nan_to_num(cur_interp["vo"].isel(time=idx).values, nan=0.0))
        u_w = float(np.nan_to_num(wnd_pt["u10"].isel(time=idx).values, nan=0.0))
        v_w = float(np.nan_to_num(wnd_pt["v10"].isel(time=idx).values, nan=0.0))

        u_s, v_s = 0.0, 0.0
        if wav_interp is not None:
            for u_name in ("VSDX", "u_stokes", "sea_surface_wave_stokes_drift_x_velocity"):
                if u_name in wav_interp.variables:
                    u_s = float(np.nan_to_num(wav_interp[u_name].isel(time=idx).values, nan=0.0))
                    has_stokes = True
                    break
            for v_name in ("VSDY", "v_stokes", "sea_surface_wave_stokes_drift_y_velocity"):
                if v_name in wav_interp.variables:
                    v_s = float(np.nan_to_num(wav_interp[v_name].isel(time=idx).values, nan=0.0))
                    has_stokes = True
                    break

        sample_dict = {
            "timestamp_utc": ts_iso,
            "u_current_mps": round(u_c, 4),
            "v_current_mps": round(v_c, 4),
            "u_wind_mps": round(u_w, 4),
            "v_wind_mps": round(v_w, 4),
            "u_stokes_mps": round(u_s, 4),
            "v_stokes_mps": round(v_s, 4),
        }
        samples.append(sample_dict)

    ds_cur.close()
    ds_wnd.close()
    if ds_wav is not None:
        ds_wav.close()

    wave_prov = (
        "wavewatch3_stokes_drift"
        if has_stokes
        else "standard_leeway_fallback (u_current + 0.032 * u_wind)"
    )

    manifest: Dict[str, Any] = {
        "source": "copernicus_cmems_era5",
        "center_lon": center_lon,
        "center_lat": center_lat,
        "wind_drift_factor": wind_drift_factor,
        "wave_provenance": wave_prov,
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d time-series samples to %s", len(samples), manifest_path)
    return manifest


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def fetch_incident_metocean(
    lon: float,
    lat: float,
    detection_time_utc: str,
    hindcast_hours: int = 24,
    forecast_hours: int = 12,
    buffer_deg: float = 1.5,
    out_dir: str = "data/processed",
    fetch_waves: bool = False,
) -> Dict[str, str]:
    """Fetch Copernicus currents + ERA5 winds (+ optional WaveWatch III waves) and write manifest."""
    det_dt = parse_utc(detection_time_utc)
    start_dt = det_dt - timedelta(hours=hindcast_hours + 3)
    end_dt = det_dt + timedelta(hours=forecast_hours + 3)

    bbox = (
        round(lon - buffer_deg, 3),
        round(lat - buffer_deg, 3),
        round(lon + buffer_deg, 3),
        round(lat + buffer_deg, 3),
    )

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    current_nc = base / "current.nc"
    wind_nc = base / "wind.nc"
    wave_nc = base / "wave.nc"
    manifest_json = base / "metocean_environment.json"

    fetch_copernicus_currents(bbox, start_dt, end_dt, current_nc)
    fetch_era5_winds(bbox, start_dt, end_dt, wind_nc)
    if fetch_waves:
        try:
            fetch_wavewatch3_waves(bbox, start_dt, end_dt, wave_nc)
        except Exception as exc:
            logger.warning("WaveWatch III fetch failed (%s); proceeding with leeway fallback: %s", wave_nc, exc)

    build_environment_manifest(
        current_nc,
        wind_nc,
        lon,
        lat,
        manifest_json,
        wave_nc=wave_nc if wave_nc.exists() else None,
    )

    result = {
        "current_nc": str(current_nc),
        "wind_nc": str(wind_nc),
        "environment_json": str(manifest_json),
    }
    if wave_nc.exists():
        result["wave_nc"] = str(wave_nc)
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Copernicus currents & ERA5 winds for OpenDrift.")
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--time", type=str, required=True, help="Satellite acquisition time (ISO UTC)")
    parser.add_argument("--hindcast-hours", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()
    paths = fetch_incident_metocean(
        lon=args.lon,
        lat=args.lat,
        detection_time_utc=args.time,
        hindcast_hours=args.hindcast_hours,
        forecast_hours=args.forecast_hours,
        out_dir=args.out_dir,
    )
    print(json.dumps(paths, indent=2))
