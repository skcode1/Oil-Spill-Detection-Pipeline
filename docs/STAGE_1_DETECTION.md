# Stage 1 — 5-class SAR Detection (detection_service.py)

## What changed
The binary oil-mask U-Net is replaced by a 5-class DeepLabV3+ / ResNet-34 segmentation
(`segmentation_models_pytorch`), trained on the Krestenitis et al. SAR taxonomy:

- `0` Sea Surface
- `1` Oil Spill
- `2` Look-Alike
- `3` Ship
- `4` Land

## Inference pipeline
1. SAR GeoTIFF is converted to a 3-channel composite `[VV, VH, VV/VH ratio]`, normalised to `[0, 1]`.
2. Sliding-window (512 px, 256 px stride) inference with overlap averaging removes tile seams.
3. Each class mask is vectorised via `rasterio.features.shapes` into WGS84 (EPSG:4326) polygons,
   preserving the scene affine transform and CRS.
4. Geodesic area is computed in EPSG:6933 (equal-area); mean softmax confidence is sampled per polygon.

## Outputs
- `slicks` (Class 1) — seed for drift & attribution.
- `look_alikes` (Class 2) — dossier warning only; never fed into drift modelling.
- `detected_ships` (Class 3) — co-located within `ship_buffer_km` (default 2.0 km) of each slick,
  attached as `slick.colocated_ships` with `visual_ship_detected`.

## Config (`config.yaml -> model`)
```
architecture: deeplabv3plus
encoder: resnet34
classes: 5
class_names: [sea_surface, oil_spill, look_alike, ship, land]
oil_class_index: 1
ship_class_index: 3
confidence_threshold: 0.90
```

## Training (`src/train.py`)
DeepLabV3Plus + Hybrid Cross-Entropy/Dice loss, per-class IoU tracked, best checkpoint
selected on Oil Spill IoU. Class weights emphasise Oil (3.0), Look-Alike (2.0), Ship (3.5).

## Contract preserved
`DetectionService.detect_path/detect_bytes/detect_scene` return `DetectionOutcome` with a
`SlickDetectionResponse` (now extended with `look_alikes` and `detected_ships`).
