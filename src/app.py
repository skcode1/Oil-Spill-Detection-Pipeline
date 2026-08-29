import os
import glob
import json
import yaml
import subprocess
import shutil
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from PIL import Image

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
RAW_SAR_DIR = paths_cfg.get("test_sar_scene", "data/raw/SAR_training/sentinel/image")
OUTPUT_DIR = paths_cfg.get("output_dir", "results")
MODEL_PATH = paths_cfg.get("model_checkpoint", "models/sar_unet_best.pth")
SPILL_PATH = os.path.join(OUTPUT_DIR, "spill_polygon.geojson")
CULPRIT_PATH = os.path.join(OUTPUT_DIR, "culprit_vessel.geojson")

# Map defaults from config
map_cfg = config.get("map_dashboard", {}) or {}
DEFAULT_LAT = map_cfg.get("center_lat", 13.0827)
DEFAULT_LON = map_cfg.get("center_lon", 80.2707)
DEFAULT_ZOOM = map_cfg.get("initial_zoom", 9)
MAPBOX_STYLE = map_cfg.get("mapbox_style", "carto-positron")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Sidebar: Automated Pipeline Controller
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Automate SAR segmentation and AIS correlation across scenes.")

# Layer visibility toggles
st.sidebar.markdown("---")
show_spill = st.sidebar.checkbox("Show Spill Layer", value=True)
show_vessel = st.sidebar.checkbox("Show Vessel Route", value=True)
show_mask = st.sidebar.checkbox("Show Mask Overlay (if available)", value=False)

available_images = sorted(glob.glob(os.path.join(RAW_SAR_DIR, "*.*")))
image_names = [os.path.basename(p) for p in available_images]

input_mode = st.sidebar.radio("Input Source", ["Select from Dataset", "Upload Custom SAR Scene"])
selected_image_path = None

if input_mode == "Select from Dataset":
    if image_names:
        selected_name = st.sidebar.selectbox("Choose SAR Scene", image_names)
        selected_image_path = os.path.join(RAW_SAR_DIR, selected_name)
    else:
        st.sidebar.warning("No images found in raw dataset directory.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload SAR image (.png, .jpg, .tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded_file is not None:
        temp_path = os.path.join(OUTPUT_DIR, "temp_uploaded_scene.png")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_image_path = temp_path

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Trigger Automated End-to-End Execution
if st.sidebar.button("🚀 Run Full Pipeline", type="primary"):
    if selected_image_path and os.path.exists(selected_image_path):
        with st.spinner("Executing U-Net inference, vectorization, and AIS cross-matching..."):
            # Step A: Run detection & vectorization
            cmd_detect = [
                "python3", "src/detect_and_vectorize.py",
                "--input", selected_image_path,
                "--model", MODEL_PATH,
                "--output_dir", OUTPUT_DIR,
                "--threshold", str(conf_threshold),
                "--save_mask"
            ]
            subprocess.run(cmd_detect, check=True)

            # Step B: Standardize output filename to spill_polygon.geojson
            base_id = os.path.splitext(os.path.basename(selected_image_path))[0]
            detected_geo = os.path.join(OUTPUT_DIR, f"{base_id}_detected.geojson")
            if os.path.exists(detected_geo):
                shutil.copy(detected_geo, SPILL_PATH)

            # also copy mask image if produced (convention: <base>_mask.png)
            detected_mask = os.path.join(OUTPUT_DIR, f"{base_id}_mask.png")
            if os.path.exists(detected_mask):
                shutil.copy(detected_mask, os.path.join(OUTPUT_DIR, "latest_mask.png"))

            # Step C: Run AIS Cross-Matching
            subprocess.run(["python3", "src/cross_match_ais.py"], check=True)
            st.sidebar.success("Analysis complete!")
    else:
        st.sidebar.error("Please select or upload a valid SAR image.")

# 3. Main Dashboard Header & Metrics
st.title("🛰️ Sentinel-1 SAR Oil Spill & AIS Culprit Tracking")

spill_detected = os.path.exists(SPILL_PATH)
culprit_detected = os.path.exists(CULPRIT_PATH)

spill_gdf = gpd.read_file(SPILL_PATH) if spill_detected else None
culprit_gdf = gpd.read_file(CULPRIT_PATH) if culprit_detected else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    has_spill = spill_detected and spill_gdf is not None and not spill_gdf.empty
    st.metric("Spill Status", "CONFIRMED" if has_spill else "CLEAR")
with col2:
    slick_count = len(spill_gdf) if has_spill else 0
    st.metric("Detected Slick Polygons", f"{slick_count}")
with col3:
    vessel_name = culprit_gdf["vessel_name"].iloc[0] if (culprit_detected and culprit_gdf is not None and not culprit_gdf.empty) else "N/A"
    st.metric("Top Suspect Vessel", vessel_name)
with col4:
    min_dist = culprit_gdf["min_distance_m"].iloc[0] if (culprit_detected and culprit_gdf is not None and not culprit_gdf.empty) else "N/A"
    st.metric("Trajectory Proximity", f"{min_dist} m" if min_dist != "N/A" else "N/A")

st.divider()

# 4. Map & Dossier Visualization
left_pane, right_pane = st.columns([3, 1])

with left_pane:
    layers = []
    # Determine map center: spill centroid > culprit centroid > config defaults
    center_lat, center_lon = DEFAULT_LAT, DEFAULT_LON
    view_zoom = DEFAULT_ZOOM
    # Render Spill GeoJSON (Red)
    if has_spill and show_spill:
        with open(SPILL_PATH, "r") as f:
            spill_geojson_data = json.load(f)
        # robust centroid extraction
        try:
            center_point = spill_gdf.unary_union.centroid
            center_lat, center_lon = center_point.y, center_point.x
        except Exception:
            center_lat, center_lon = DEFAULT_LAT, DEFAULT_LON

        spill_layer = pdk.Layer(
            "GeoJsonLayer",
            spill_geojson_data,
            opacity=0.7,
            stroked=True,
            filled=True,
            get_fill_color=[230, 57, 70, 180],
            get_line_color=[255, 255, 255, 255],
            get_line_width=2,
            pickable=True
        )
        layers.append(spill_layer)

    # Render Culprit Vessel Route (Cyan Line)
    if culprit_detected and culprit_gdf is not None and not culprit_gdf.empty and show_vessel:
        with open(CULPRIT_PATH, "r") as f:
            culprit_geojson_data = json.load(f)

        # if no spill center chosen yet, use culprit centroid
        if center_lat == DEFAULT_LAT and center_lon == DEFAULT_LON:
            try:
                cpt = culprit_gdf.unary_union.centroid
                center_lat, center_lon = cpt.y, cpt.x
            except Exception:
                pass

        vessel_layer = pdk.Layer(
            "GeoJsonLayer",
            culprit_geojson_data,
            opacity=0.9,
            stroked=True,
            filled=False,
            get_line_color=[0, 245, 212, 255],
            get_line_width=4,
            pickable=True
        )
        layers.append(vessel_layer)

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=view_zoom,
        pitch=30
    )

    # Choose a map style; support either Mapbox style name or Carto style strings from config
    map_style = MAPBOX_STYLE if isinstance(MAPBOX_STYLE, str) else "mapbox://styles/mapbox/dark-v11"

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={"text": "Vessel / Spill Geometry"}
    )
    st.pydeck_chart(r)

with right_pane:
    st.subheader("🖼️ Selected SAR Scene")
    if selected_image_path and os.path.exists(selected_image_path):
        st.image(selected_image_path, caption=os.path.basename(selected_image_path), use_container_width=True)

    # Mask preview (if available and requested)
    latest_mask = os.path.join(OUTPUT_DIR, "latest_mask.png")
    if show_mask and os.path.exists(latest_mask):
        st.subheader("🧭 Detected Mask Preview")
        st.image(latest_mask, caption="Segmentation mask (latest)", use_container_width=True)

    st.subheader("📋 Incident Dossier")
    if culprit_detected and culprit_gdf is not None and not culprit_gdf.empty:
        culprit_info = culprit_gdf.iloc[0]
        st.write(f"**Vessel Name:** {culprit_info.get('vessel_name', 'Unknown')}")
        st.write(f"**MMSI:** `{culprit_info.get('mmsi', 'Unknown')}`")
        st.write(f"**Vessel Type:** {culprit_info.get('vessel_type', 'Unknown')}")
        st.write(f"**Avg Speed:** {culprit_info.get('avg_speed_knots', 'N/A')} knots")
        st.write(f"**Nearest Approach:** {culprit_info.get('min_distance_m', 'N/A')} m")
        st.write(f"**Last Ping (UTC):** {culprit_info.get('last_seen', 'N/A')}")
    else:
        st.info("No culprit vessel data available.")

    st.subheader("⚙️ Active Legend")
    st.markdown("🔴 **Red Polygon**: Oil Spill Segmentation")
    st.markdown("🟢/🔵 **Cyan Line**: Vessel Trajectory")

    # Download buttons
    st.markdown("---")
    if has_spill:
        try:
            with open(SPILL_PATH, "r") as f:
                geojson_text = f.read()
            st.download_button("Download Spill GeoJSON", data=geojson_text, file_name="spill_polygon.geojson")
        except Exception:
            pass

    if culprit_detected:
        try:
            with open(CULPRIT_PATH, "r") as f:
                geojson_text = f.read()
            st.download_button("Download Culprit GeoJSON", data=geojson_text, file_name="culprit_vessel.geojson")
        except Exception:
            pass

    # Basic spill stats (area) if CRS supports metric reprojection
    if has_spill:
        try:
            metric_crs = config.get('ais', {}).get('metric_crs', 'EPSG:3857')
            areas = (spill_gdf.to_crs(metric_crs).area / 1.0).round(1)
            stats_df = spill_gdf.copy()
            stats_df['area_m2'] = areas
            st.subheader('📊 Detection Summary')
            st.dataframe(stats_df[[col for col in stats_df.columns if col != 'geometry']].fillna(''))
        except Exception:
            pass