import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import geopandas as gpd
import numpy as np
import pydeck as pdk
import streamlit as st
import yaml

st.set_page_config(
    page_title="Maritime Oil Spill & AIS Surveillance",
    page_icon="🛰️",
    layout="wide"
)

# 1. Configuration & Paths
CONFIG_PATH = "config.yaml"
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

paths_cfg = config.get("paths") or {}
map_cfg = config.get("map_dashboard") or {}
RAW_SAR_DIR = paths_cfg.get("test_sar_scene", "data/raw/SAR_training/sentinel/image")
OUTPUT_DIR = paths_cfg.get("output_dir", "results")
MODEL_PATH = paths_cfg.get("model_checkpoint", "models/oil_spill_segmentation_model.pth")
SPILL_PATH = os.path.join(OUTPUT_DIR, "spill_polygon.geojson")
CULPRIT_PATH = os.path.join(OUTPUT_DIR, "culprit_vessel.geojson")
LATEST_MASK_PATH = os.path.join(OUTPUT_DIR, "latest_mask.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
            font-weight: 700;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetricContainer"]) {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 0.8rem;
            padding: 0.5rem 0.75rem;
        }
        .status-chip {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            background: rgba(22, 163, 74, 0.12);
            color: #7ef0b6;
            border: 1px solid rgba(110, 231, 183, 0.35);
        }
        .warning-chip {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            background: rgba(245, 158, 11, 0.12);
            color: #fbd38d;
            border: 1px solid rgba(251, 191, 36, 0.35);
        }
        .danger-chip {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            background: rgba(239, 68, 68, 0.12);
            color: #fca5a5;
            border: 1px solid rgba(252, 165, 165, 0.4);
        }
        .info-panel {
            background: linear-gradient(135deg, rgba(17,24,39,0.88), rgba(15,23,42,0.72));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
        }
        .subtle-label {
            color: #9ca3af;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def validate_sar_image(image_path):
    """Validates radiometric characteristics to reject regular optical/camera photos."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "Unreadable image file format."

    if len(img.shape) == 3 and img.shape[2] >= 3:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))
        if diff_rg > 8.0 or diff_gb > 8.0:
            return False, "Optical/RGB photo detected. Model requires Sentinel-1 SAR backscatter imagery."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if np.std(gray) < 5.0:
        return False, "Insufficient dynamic range or blank image."

    return True, "Valid SAR backscatter scene."


# 2. Sidebar Controls
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Automate SAR segmentation and AIS correlation across scenes.")

show_spill = st.sidebar.checkbox("Show Spill Layer", value=True)
show_vessel = st.sidebar.checkbox("Show Vessel Route", value=True)
show_mask = st.sidebar.checkbox("Show Mask Overlay", value=False)

input_mode = st.sidebar.radio("Input Source", ["Select from Dataset", "Upload Custom SAR Scene"])
selected_image_path = None

if input_mode == "Select from Dataset":
    available_images = sorted(glob.glob(os.path.join(RAW_SAR_DIR, "*.*")))
    image_names = [os.path.basename(p) for p in available_images]
    if image_names:
        selected_name = st.sidebar.selectbox("Choose SAR Scene", image_names)
        selected_image_path = os.path.join(RAW_SAR_DIR, selected_name)
        st.session_state["active_image"] = selected_image_path
    else:
        st.sidebar.warning("No images found in raw dataset directory.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload SAR Image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "PNG", "JPG", "JPEG", "TIF", "TIFF"],
        key="sar_uploader"
    )
    if uploaded_file is not None:
        save_dest = os.path.join(OUTPUT_DIR, f"uploaded_{uploaded_file.name}")
        with open(save_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["active_image"] = save_dest
        selected_image_path = save_dest

if "active_image" in st.session_state and os.path.exists(st.session_state["active_image"]):
    selected_image_path = st.session_state["active_image"]

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# 3. Execution Trigger
if st.sidebar.button("🚀 Run Full Pipeline", type="primary"):
    if selected_image_path and os.path.exists(selected_image_path):
        # Step 0: Validate SAR Data Authenticity
        is_valid_sar, message = validate_sar_image(selected_image_path)

        if not is_valid_sar:
            # Purge any stale outputs
            for old_file in [SPILL_PATH, CULPRIT_PATH, LATEST_MASK_PATH]:
                if os.path.exists(old_file):
                    os.remove(old_file)
            st.sidebar.error(f"❌ Input Rejected: {message}")
        else:
            with st.spinner("Processing valid SAR scene and correlating AIS tracks..."):
                for old_file in [SPILL_PATH, CULPRIT_PATH, LATEST_MASK_PATH]:
                    if os.path.exists(old_file):
                        os.remove(old_file)

                # Step A: Run Model Detection
                cmd_detect = [
                    "python3", "src/detect_and_vectorize.py",
                    "--input", selected_image_path,
                    "--model", MODEL_PATH,
                    "--output_dir", OUTPUT_DIR,
                    "--threshold", str(conf_threshold),
                    "--save_mask"
                ]
                subprocess.run(cmd_detect, check=True)

                # Step B: Copy Output Files
                base_id = os.path.splitext(os.path.basename(selected_image_path))[0]
                detected_geo = os.path.join(OUTPUT_DIR, f"{base_id}_detected.geojson")
                if os.path.exists(detected_geo):
                    shutil.copy(detected_geo, SPILL_PATH)

                detected_mask = os.path.join(OUTPUT_DIR, f"{base_id}_mask.png")
                if os.path.exists(detected_mask):
                    shutil.copy(detected_mask, LATEST_MASK_PATH)

                # Step C: AIS Cross Matching (canonical dynamic engine)
                if os.path.exists(SPILL_PATH) and os.path.getsize(SPILL_PATH) > 0:
                    subprocess.run(
                        [sys.executable, "src/cross_match_ais.py", "--config", CONFIG_PATH], check=True
                    )
                    st.sidebar.success("Analysis complete.")
                else:
                    st.sidebar.info("No slicks detected in SAR scene.")
                st.rerun()
    else:
        st.sidebar.error("Please upload or select an image file first.")

# 4. Status Metrics
inject_custom_css()

st.markdown(
    """
    <div class="info-panel">
        <div class="subtle-label">Maritime Intelligence Dashboard</div>
        <h2 style="margin: 0.2rem 0 0.4rem 0;">🛰️ Sentinel-1 SAR Oil Spill & AIS Culprit Tracking</h2>
        <div style="color:#cbd5e1; font-size:0.95rem;">
            Monitoring spill signatures and correlating vessel activity for high-risk maritime incidents.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

spill_detected = os.path.exists(SPILL_PATH) and os.path.getsize(SPILL_PATH) > 0
culprit_detected = os.path.exists(CULPRIT_PATH) and os.path.getsize(CULPRIT_PATH) > 0

spill_gdf = gpd.read_file(SPILL_PATH) if spill_detected else None
culprit_gdf = gpd.read_file(CULPRIT_PATH) if culprit_detected else None

has_spill = spill_gdf is not None and not spill_gdf.empty
scene_valid = True
scene_message = "Ready"
if selected_image_path and os.path.exists(selected_image_path):
    scene_valid, scene_message = validate_sar_image(selected_image_path)

status_chip = "<span class='status-chip'>HEALTHY</span>" if scene_valid else "<span class='danger-chip'>INVALID INPUT</span>"
if not selected_image_path or not os.path.exists(selected_image_path):
    status_chip = "<span class='warning-chip'>WAITING FOR IMAGE</span>"

st.markdown(
    f"""
    <div class="info-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
                <div class="subtle-label">Scene Validation</div>
                <div style="font-size:1.05rem; color:#e2e8f0; margin-top:0.25rem;">{scene_message}</div>
            </div>
            <div>{status_chip}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spill Status", "CONFIRMED" if has_spill else "CLEAR / INVALID")
with col2:
    st.metric("Detected Slick Polygons", f"{len(spill_gdf)}" if has_spill else "0")
with col3:
    vessel_name = culprit_gdf["vessel_name"].iloc[0] if (culprit_gdf is not None and not culprit_gdf.empty) else "None"
    st.metric("Top Suspect Vessel", vessel_name)
with col4:
    min_dist = culprit_gdf["min_distance_m"].iloc[0] if (culprit_gdf is not None and not culprit_gdf.empty) else "N/A"
    st.metric("Trajectory Proximity", f"{min_dist} m" if min_dist != "N/A" else "N/A")

st.divider()

# 5. Map & Incident Display
left_pane, right_pane = st.columns([3, 1])

with left_pane:
    layers = []
    center_lat = float(map_cfg.get("center_lat", 13.0827))
    center_lon = float(map_cfg.get("center_lon", 80.2707))

    if has_spill and show_spill:
        with open(SPILL_PATH, "r") as f:
            spill_geojson_data = json.load(f)
        center_point = spill_gdf.union_all().centroid if hasattr(spill_gdf, "union_all") else spill_gdf.unary_union.centroid
        center_lat, center_lon = center_point.y, center_point.x

        layers.append(pdk.Layer(
            "GeoJsonLayer",
            spill_geojson_data,
            opacity=0.78,
            stroked=True,
            filled=True,
            get_fill_color=[230, 57, 70, 180],
            get_line_color=[255, 255, 255, 255],
            get_line_width=2,
            pickable=True
        ))

    if culprit_gdf is not None and not culprit_gdf.empty and show_vessel:
        with open(CULPRIT_PATH, "r") as f:
            culprit_geojson_data = json.load(f)

        layers.append(pdk.Layer(
            "GeoJsonLayer",
            culprit_geojson_data,
            opacity=0.95,
            stroked=True,
            filled=False,
            get_line_color=[0, 245, 212, 255],
            get_line_width=4,
            pickable=True
        ))

    if not has_spill and (culprit_gdf is None or culprit_gdf.empty):
        st.info("Select or upload a SAR image and run the pipeline to populate the map.")

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=float(map_cfg.get("initial_zoom", 10)),
        pitch=float(map_cfg.get("pitch", 30)),
        bearing=0,
        transition_duration=800,
    )

    # Keep map visible without requiring a Mapbox token; style is config-driven.
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style=str(map_cfg.get("pydeck_style", "dark_no_labels")),
        tooltip={"text": "Vessel / Spill Geometry"}
    )
    st.pydeck_chart(r, width="stretch")

with right_pane:
    st.subheader("🖼️ Target Scene")
    if selected_image_path and os.path.exists(selected_image_path):
        st.image(selected_image_path, caption=os.path.basename(selected_image_path), width="stretch")

    if show_mask and os.path.exists(LATEST_MASK_PATH):
        st.subheader("🧭 Model Binary Mask")
        st.image(LATEST_MASK_PATH, caption="Output Mask", width="stretch")

    st.subheader("� Spill Geo-Location")
    if has_spill and spill_gdf is not None:
        spill_union = spill_gdf.union_all() if hasattr(spill_gdf, "union_all") else spill_gdf.unary_union
        spill_centroid = spill_union.centroid
        min_lon, min_lat, max_lon, max_lat = spill_gdf.total_bounds

        area_km2 = (spill_gdf.to_crs(epsg=3857).area.sum()) / 1e6

        st.markdown(f"**Centroid Lat/Lon:** `{spill_centroid.y:.5f}° N, {spill_centroid.x:.5f}° E`")
        st.markdown("**Bounding Box:**")
        st.caption(f"SW: `{min_lat:.4f}°, {min_lon:.4f}°` | NE: `{max_lat:.4f}°, {max_lon:.4f}°`")
        st.markdown(f"**Estimated Slick Area:** `{area_km2:.3f} km²`")
    else:
        st.info("No active spill location coordinates.")

    st.divider()

    st.subheader("�📋 Incident Dossier")
    if culprit_gdf is not None and not culprit_gdf.empty:
        info = culprit_gdf.iloc[0]
        st.write(f"**Vessel Name:** {info.get('vessel_name', 'Unknown')}")
        st.write(f"**MMSI:** `{info.get('mmsi', 'Unknown')}`")
        st.write(f"**Vessel Type:** {info.get('vessel_type', 'Unknown')}")
        st.write(f"**Avg Speed:** {info.get('avg_speed_knots', 'N/A')} knots")
        st.write(f"**Nearest Approach:** {info.get('min_distance_m', 'N/A')} m")
        st.write(f"**Last Ping (UTC):** {info.get('last_seen', 'N/A')}")
    else:
        st.info("No active incident details.")

    st.markdown("---")
    st.caption("Operational notes")
    if has_spill:
        st.success("Spill polygon detected and mapped to the incident area.")
    else:
        st.warning("No confirmed spill polygon for the current SAR scene.")
