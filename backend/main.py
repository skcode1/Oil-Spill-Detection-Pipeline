"""
╔══════════════════════════════════════════════════════════════════╗
║  DEPRECATED — Legacy monolithic backend                         ║
║                                                                  ║
║  This module is superseded by the production API at              ║
║  ``app/main.py`` (run via ``uvicorn app.main:app``).             ║
║                                                                  ║
║  All new development MUST target the ``app/`` package.           ║
║  This file is retained only for backward-compatibility           ║
║  reference and will be removed in a future release.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings as _warnings
_warnings.warn(
    "backend/main.py is deprecated. Use 'uvicorn app.main:app' instead.",
    DeprecationWarning,
    stacklevel=1,
)

import os
import sys
import glob
import json
import shutil
import subprocess
import cv2
import numpy as np
import geopandas as gpd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Maritime SAR & AIS Surveillance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "config.yaml")


def _load_config(path: str) -> dict:
    """Loads the pipeline config; absence/malformation degrades to defaults."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            import yaml

            return yaml.safe_load(handle) or {}
    return {}


config = _load_config(CONFIG_PATH)
paths_cfg = config.get("paths") or {}
backend_cfg = config.get("backend") or {}
validation_cfg = config.get("validation") or {}
map_cfg = config.get("map_dashboard") or {}
_regions = config.get("regions") or []

RAW_SAR_DIR = paths_cfg.get("test_sar_scene", "data/raw/SAR_training/sentinel/image")
OUTPUT_DIR = paths_cfg.get("output_dir", "results")
MODEL_PATH = paths_cfg.get("model_checkpoint", "models/oil_spill_segmentation_model.pth")
SPILL_PATH = os.path.join(OUTPUT_DIR, "spill_polygon.geojson")
CULPRIT_PATH = os.path.join(OUTPUT_DIR, "culprit_vessel.geojson")
LATEST_MASK_PATH = os.path.join(OUTPUT_DIR, "latest_mask.png")
BASE_URL = str(backend_cfg.get("base_url", "http://localhost:8000")).rstrip("/")
PYTHON_EXE = str(backend_cfg.get("python_executable") or sys.executable)
DEFAULT_LOCATION_NAME = str(
    backend_cfg.get("default_location_name")
    or (_regions[0].get("name") if _regions else "Maritime Surveillance Zone")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_SAR_DIR, exist_ok=True)

# Mount results folder to serve mask and uploaded images directly
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")
app.mount("/raw_sar", StaticFiles(directory=RAW_SAR_DIR), name="raw_sar")


def is_supported_sar_extension(filename: str):
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_reference_sar_stats():
    ref_dir = RAW_SAR_DIR
    fallback_stats = dict(validation_cfg.get("reference_stats") or {})
    fallback_stats.setdefault("mean", 85.0)
    fallback_stats.setdefault("std", 28.0)
    fallback_stats.setdefault("min_dim", 64)
    fallback_stats.setdefault("max_dim", 512)
    if not os.path.isdir(ref_dir):
        return fallback_stats

    values = []
    for file in sorted(glob.glob(os.path.join(ref_dir, "*.*")))[:100]:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            if img.shape[2] >= 3:
                b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
                diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))
                if diff_rg > 5.0 or diff_gb > 5.0:
                    continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        if gray.shape[0] < 32 or gray.shape[1] < 32:
            continue

        values.append({"mean": float(np.mean(gray)), "std": float(np.std(gray))})

    if not values:
        return fallback_stats

    mean_vals = np.array([v["mean"] for v in values], dtype=float)
    std_vals = np.array([v["std"] for v in values], dtype=float)
    return {
        "mean": float(np.mean(mean_vals)),
        "std": float(np.mean(std_vals)),
        "min_dim": int(fallback_stats["min_dim"]),
        "max_dim": int(fallback_stats["max_dim"]),
        "mean_low": float(np.min(mean_vals)),
        "mean_high": float(np.max(mean_vals)),
        "std_low": float(np.min(std_vals)),
        "std_high": float(np.max(std_vals)),
    }


def validate_sar_image(image_path: str):
    if not is_supported_sar_extension(os.path.basename(image_path)):
        return False, "Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted."

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "Unreadable image format."

    if img.ndim == 2:
        gray = img.astype(np.float32)
        rgb_diff = 0.0
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rgb_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float))) + np.mean(np.abs(g.astype(float) - b.astype(float))) + np.mean(np.abs(r.astype(float) - b.astype(float)))) / 3.0
        if rgb_diff > 5.0:
            return False, "Invalid SAR image: optical/RGB image detected. Only authentic SAR scene data matching the dataset is accepted."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        return False, "Invalid SAR image: unsupported image structure. Only SAR scenes matching the dataset are accepted."

    min_dim = int(validation_cfg.get("min_dim", fallback_stats["min_dim"]))
    max_dim = int(validation_cfg.get("max_dim", fallback_stats["max_dim"]))
    h, w = gray.shape[:2]
    if h < min_dim or w < min_dim or h > max_dim or w > max_dim:
        return False, "Invalid SAR image: dimensions do not match valid SAR training data."

    ref = get_reference_sar_stats()
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    mean_lo, mean_hi = (float(v) for v in validation_cfg.get("mean_range", (10.0, 180.0)))
    std_lo, std_hi = (float(v) for v in validation_cfg.get("std_range", (5.0, 90.0)))
    mean_tolerance = float(validation_cfg.get("mean_tolerance", 70.0))
    std_tolerance = float(validation_cfg.get("std_tolerance", 45.0))

    if gray_mean < mean_lo or gray_mean > mean_hi:
        return False, "Invalid SAR image: brightness profile does not match SAR training scenes."
    if gray_std < std_lo or gray_std > std_hi:
        return False, "Invalid SAR image: SAR texture signature is outside the valid dataset range."
    if abs(gray_mean - ref["mean"]) > mean_tolerance:
        return False, "Invalid SAR image: mean intensity does not match authentic SAR data."
    if abs(gray_std - ref["std"]) > std_tolerance:
        return False, "Invalid SAR image: SAR backscatter variance does not match the dataset."

    return True, "Valid SAR backscatter scene."


@app.get("/api/scenes")
def get_dataset_scenes():
    files = sorted(glob.glob(os.path.join(RAW_SAR_DIR, "*.*")))
    return {"scenes": [os.path.basename(f) for f in files]}


@app.post("/api/run-pipeline")
async def run_pipeline(
    mode: str = Form(...),
    selected_scene: str = Form(None),
    threshold: float = Form(0.50),
    file: UploadFile = File(None)
):
    # 1. Determine target image
    if mode == "dataset":
        if not selected_scene:
            raise HTTPException(status_code=400, detail="No scene selected.")
        target_path = os.path.join(RAW_SAR_DIR, selected_scene)
    else:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")
        if not is_supported_sar_extension(file.filename):
            for old in [SPILL_PATH, CULPRIT_PATH, LATEST_MASK_PATH]:
                if os.path.exists(old):
                    os.remove(old)
            return {"success": False, "validation_msg": "Invalid SAR image: only Sentinel-1 SAR files (.png, .jpg, .jpeg, .tif, .tiff) are accepted.", "status": "INVALID_INPUT"}
        target_path = os.path.join(OUTPUT_DIR, f"uploaded_{file.filename}")
        with open(target_path, "wb") as f:
            f.write(await file.read())

    # 2. Radiometric validation
    valid, msg = validate_sar_image(target_path)
    if not valid:
        for old in [SPILL_PATH, CULPRIT_PATH, LATEST_MASK_PATH]:
            if os.path.exists(old):
                os.remove(old)
        return {"success": False, "validation_msg": msg, "status": "INVALID_INPUT"}

    # 3. Clean and run inference
    for old in [SPILL_PATH, CULPRIT_PATH, LATEST_MASK_PATH]:
        if os.path.exists(old):
            os.remove(old)

    cmd_detect = [
         sys.executable, 
         "src/detect_and_vectorize.py",
        "--input", target_path,
        "--model", MODEL_PATH,
        "--output_dir", OUTPUT_DIR,
        "--threshold", str(threshold),
        "--save_mask"
    ]
    subprocess.run(cmd_detect, check=True)

    base_id = os.path.splitext(os.path.basename(target_path))[0]
    detected_geo = os.path.join(OUTPUT_DIR, f"{base_id}_detected.geojson")
    if os.path.exists(detected_geo):
        shutil.copy(detected_geo, SPILL_PATH)

    detected_mask = os.path.join(OUTPUT_DIR, f"{base_id}_mask.png")
    if os.path.exists(detected_mask):
        shutil.copy(detected_mask, LATEST_MASK_PATH)

    # 4. AIS Cross-Matching (canonical dynamic engine, config-driven)
    if os.path.exists(SPILL_PATH) and os.path.getsize(SPILL_PATH) > 0:
        subprocess.run(
            [PYTHON_EXE, "src/cross_match_ais.py", "--config", CONFIG_PATH], check=True
        )

    return get_current_status(target_path, msg)


@app.get("/api/status")
def get_status_endpoint():
    return get_current_status()


def get_current_status(target_path=None, validation_msg="Valid SAR Scene"):
    spill_exists = os.path.exists(SPILL_PATH) and os.path.getsize(SPILL_PATH) > 0
    culprit_exists = os.path.exists(CULPRIT_PATH) and os.path.getsize(CULPRIT_PATH) > 0

    spill_geojson = json.load(open(SPILL_PATH)) if spill_exists else None
    culprit_geojson = json.load(open(CULPRIT_PATH)) if culprit_exists else None

    dossier = None
    geo_stats = None

    if spill_exists:
        spill_gdf = gpd.read_file(SPILL_PATH)
        spill_union = spill_gdf.union_all() if hasattr(spill_gdf, "union_all") else spill_gdf.unary_union
        centroid = spill_union.centroid
        bounds = spill_gdf.total_bounds
        area_km2 = float((spill_gdf.to_crs(epsg=3857).area.sum()) / 1e6)

        # Extract location name and authority from GeoJSON columns
        location_name = spill_gdf["location_name"].iloc[0] if "location_name" in spill_gdf.columns else DEFAULT_LOCATION_NAME
        authority = spill_gdf["authority"].iloc[0] if "authority" in spill_gdf.columns else None

        geo_stats = {
            "location_name": location_name,
            "authority": authority,
            "centroid_lat": round(float(centroid.y), 5),
            "centroid_lon": round(float(centroid.x), 5),
            "bounds": [round(float(b), 4) for b in bounds],
            "area_km2": round(area_km2, 3),
            "slick_count": len(spill_gdf)
        }

    if culprit_exists:
        culprit_gdf = gpd.read_file(CULPRIT_PATH)
        if not culprit_gdf.empty:
            row = culprit_gdf.iloc[0]
            dossier = {
                "vessel_name": row.get("vessel_name", "Unknown"),
                "mmsi": int(row.get("mmsi", 0)),
                "vessel_type": row.get("vessel_type", "Unknown"),
                "avg_speed_knots": float(row.get("avg_speed_knots", 0.0)),
                "min_distance_m": float(row.get("min_distance_m", 0.0)),
                "last_seen": str(row.get("last_seen", "N/A"))
            }

    return {
        "success": True,
        "validation_msg": validation_msg,
        "has_spill": spill_exists,
        "spill_geojson": spill_geojson,
        "culprit_geojson": culprit_geojson,
        "geo_stats": geo_stats,
        "dossier": dossier,
        "target_scene_url": f"{BASE_URL}/raw_sar/{os.path.basename(target_path)}" if (target_path and "raw_sar" in target_path) else (f"{BASE_URL}/results/{os.path.basename(target_path)}" if target_path else None),
        "mask_url": f"{BASE_URL}/results/latest_mask.png" if os.path.exists(LATEST_MASK_PATH) else None
    }