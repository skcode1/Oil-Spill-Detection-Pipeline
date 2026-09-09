# Maritime Oil Spill Detection and AIS Attribution Pipeline

## Overview

This project develops an end-to-end maritime intelligence system for detecting oil spills from synthetic aperture radar (SAR) imagery and identifying the most likely responsible vessel using AIS trajectory correlation. The solution combines deep learning-based segmentation, geospatial vectorization, and ship attribution logic into a single operational workflow.

The repository currently contains:

- A PyTorch segmentation pipeline for SAR spill detection
- AIS-based culprit-vessel ranking and hindcast correlation
- A FastAPI backend for process orchestration and API exposure
- A React + Next.js dashboard for monitoring detections and vessel results
- A Streamlit prototype dashboard for local experimentation
- Config-driven project settings and result exports

This project is designed for scenarios such as offshore monitoring, maritime safety analysis, environmental response planning, and suspicious vessel activity investigation.

---

## Problem Statement

Oil spills at sea are difficult to detect and attribute quickly using only a single data source. SAR imagery is highly effective for identifying spill patterns, but translating those detections into actionable intelligence requires linking the slick back to vessel traffic patterns and likely source behavior.

This solution addresses that by:

1. Validating and processing SAR imagery to ensure the input matches authentic backscatter data
2. Running a segmentation model to detect oil-like regions
3. Converting segment masks to geospatial polygons
4. Performing AIS correlation using temporal and spatial proximity heuristics
5. Ranking likely culprit vessels and presenting the results through a dashboard

---

## Core Features

### SAR spill detection
- Sentinel-1-style SAR scene validation
- U-Net-based segmentation architecture using ResNet34 encoder
- Threshold-based inference with mask export and vector conversion
- GeoJSON output for detected slick polygons

### Geospatial processing
- Conversion of binary masks into polygon geometries
- Optional georeferencing support for GeoTIFF inputs
- Output stored in GeoJSON formats for downstream analytics and map visualization

### AIS-based vessel attribution
- Hindcast-based origin estimation from oil slick centroid and drift assumptions
- Search and scoring of nearby vessel trajectories
- Risk scoring combining:
  - spatial proximity
  - temporal alignment
  - vessel type risk
  - trajectory anomalies
  - AIS blackout / dark vessel indicators

### Web interfaces
- FastAPI backend for model execution and data orchestration
- Next.js dashboard for interactive operation and mapping
- Streamlit dashboard as a lightweight prototype interface

---

## Project Architecture

The repository is organized into the following major components:

```text
SIH26143/
├── README.md
├── config.yaml
├── requirements.txt
├── backend/
│   └── main.py                 # FastAPI backend API
├── src/
│   ├── app.py                  # Streamlit prototype dashboard
│   ├── dataset.py              # SAR dataset loader
│   ├── detect_and_vectorize.py # SAR inference + vectorization
│   ├── cross_match_ais.py      # AIS correlation and ranking
│   ├── environment_fetcher.py  # Metocean data retrieval utilities
│   ├── hindcast_engine.py      # Drift/hindcast logic
│   ├── metadata_extractor.py   # Metadata support utilities
│   ├── preprocess.py           # Preprocessing helpers
│   ├── split_and_patch.py      # Patch-based processing utilities
│   ├── simulation_generator.py # Synthetic/simulated data generation
│   ├── train.py                # Training pipeline
│   └── ...
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── models/
│   └── sar_unet_best.pth       # Trained U-Net checkpoint
├── outputs/
│   └── generated GeoJSON files
├── results/
│   └── final detection and correlation outputs
├── maritime-surveillance/
│   ├── app/
│   ├── src/
│   ├── package.json
│   └── README.md
└── train_gpu.ipynb
```

---

## Core Workflow

### 1. Data preparation and validation
The pipeline starts by validating the uploaded or selected SAR scene to ensure it is an authentic backscatter image rather than a normal optical photograph. The validation logic checks dimensions, grayscale intensity, and SAR-like variance patterns.

### 2. Model inference
The deep learning model is loaded from the checkpoint at `models/sar_unet_best.pth` and applied to the uploaded SAR scene. A U-Net architecture using a ResNet34 encoder is used for binary segmentation of potential oil spill regions.

### 3. Mask-to-vector conversion
Detected pixels are converted into polygon geometries and exported as GeoJSON. The output can include a mask image for visualization and analysis.

### 4. AIS cross-matching
The system reads AIS traffic records and correlates them with the spill origin. It estimates a probable origin based on combined drift and hindcast assumptions and ranks vessels according to the probability that they caused the event.

### 5. Dashboard display
The map dashboard surfaces:
- detected slick polygons
- vessel route lines
- classification and anomaly details
- a ranked suspect vessel dossier

---

## Model and Training Setup

The training pipeline is implemented in `src/train.py` and uses:

- `segmentation-models-pytorch`
- `torch` and `torchvision`
- binary segmentation objective based on Dice loss
- dataset loader in `src/dataset.py`
- training on SAR image-label pairs from the configured dataset directory

The default model configuration is defined in `config.yaml` and includes the following settings:

```yaml
model:
  architecture: "unet"
  encoder: "resnet34"
  in_channels: 1
  classes: 1
  input_size: [256, 256]
  confidence_threshold: 0.50
```

---

## Configuration

The main project configuration is stored in `config.yaml` and includes:

- dataset and checkpoint paths
- output directories
- model thresholds
- inference parameters
- AIS search radius and time window
- map center settings for the dashboard

Key parameters include:

- `raw_sar_dir`: raw SAR dataset path
- `model_checkpoint`: trained model path
- `output_dir`: result export directory
- `ais_csv`: AIS data file
- `confidence_threshold`: segmentation threshold
- `time_window_hours`: AIS correlation window
- `buffer_distance_m`: spatial search radius for candidate vessels

---

## Setup and Installation

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd maritime-surveillance
npm install
```

---

## Running the Project

### Run the backend API

From the project root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

This exposes endpoints such as:

- `/api/scenes`
- `/api/status`
- `/api/run-pipeline`

### Run the dashboard

From the frontend folder:

```bash
cd maritime-surveillance
npm run dev
```

Then open:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`

### Run the Streamlit prototype

From the root folder:

```bash
python src/app.py
```

### Run inference directly

```bash
python src/detect_and_vectorize.py \
  --input data/raw/SAR_training/sentinel/image/sample_scene.png \
  --model models/sar_unet_best.pth \
  --output_dir results \
  --threshold 0.5 \
  --save_mask
```

---

## Outputs Produced

The software exports the following artifacts:

- `results/spill_polygon.geojson` — detected spill polygons
- `results/culprit_vessel.geojson` — ranked likely culprit vessel records
- `results/latest_mask.png` — latest segmentation mask
- `results/*.png` and `results/*.geojson` — per-scene outputs
- `data/processed/metocean_environment.json` — metocean drift summary (when generated)

These outputs are used both for geospatial reporting and frontend visualization.

---

## Current Status of the Project

This repository already demonstrates a working proof-of-concept pipeline with the following implemented components:

- SAR validation for realistic backscatter input
- segmentation inference using a trained U-Net
- polygon vectorization from masks
- GIS-ready output generation
- AIS candidate selection and attribution scoring
- map-based user interface and API orchestration

The system is functional as a prototype and is suitable for demo use, scenario testing, and further iterative development.

---

## Possible Future Adjustments

The following improvements are the most valuable next steps for productionizing this project:

### 1. Improve model robustness
- Use larger and more diverse SAR datasets
- Add data augmentation for class imbalance and noisy scenes
- Train with better spill annotation quality and multi-scene validation
- Tune thresholding based on region-specific SAR distributions

### 2. Strengthen geospatial accuracy
- Add proper georeferencing for all scene inputs, not only GeoTIFF files
- Improve polygon cleanup and contour filtering
- Introduce post-processing steps like morphological operations and connected-component pruning
- Add confidence intervals for detected spill boundaries

### 3. Upgrade AIS attribution logic
- Replace heuristic scoring with probabilistic or model-based attribution
- Integrate vessel route history and port departure information
- Add more advanced anomaly detection for irregular vessel behavior
- Use real-time AIS streams instead of static CSV ingestion

### 4. Improve metocean modeling
- Integrate live ERA5 and Copernicus Marine APIs for better hindcast accuracy
- Add region-specific drift calibration for Gulf and coastal marine environments
- Include uncertainty ranges for spill origin estimation

### 5. Productionize APIs and deployment
- Add authentication and authorization for the backend
- Add structured logging, monitoring, and health checks
- Containerize the stack with Docker and Docker Compose
- Deploy backend and frontend separately for production hosting

### 6. Improve frontend experience
- Add timeline controls and scene browsing filters
- Add map layer toggling for wind/current overlays
- Support historical archive browsing and saved incident reports
- Improve responsive behavior and mobile UI support

### 7. Operational analytics and reporting
- Generate incident reports PDF/JSON exports
- Add incident IDs and event history tracking
- Connect to GIS dashboards or national maritime monitoring systems

---

## Suggested Next Development Roadmap

1. Expand the SAR training dataset and improve model validation metrics
2. Harden the inference and geospatial output pipeline
3. Upgrade the attribution engine with richer vessel intelligence
4. Connect live environmental data and real AIS feeds
5. Deploy the solution with monitoring and operational APIs
6. Add reporting and incident management workflows

---

## Summary

This project represents a meaningful end-to-end prototype for maritime oil spill detection and vessel attribution using SAR + AIS intelligence. It combines machine learning, geospatial analytics, and a web-based visualization layer in a single system that is ready for demonstration, experimentation, and future deployment-grade enhancement.

The codebase already demonstrates the major building blocks required for a useful maritime monitoring platform, and the roadmap above outlines the most important improvements to move from a strong prototype to a robust field-ready solution.

