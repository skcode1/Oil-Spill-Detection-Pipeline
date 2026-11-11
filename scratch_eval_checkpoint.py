#!/usr/bin/env python
"""Score a segmentation checkpoint against the authentic test corpus.

Runs the same inference path the API uses (``ModelService`` sliding window with
per-tile normalisation) over N scenes per class and reports, for several
confidence thresholds, what the dashboard would actually show:

* ``Oil``        -> mean oil IoU + slick count  (must be > 0 slicks)
* ``No oil``     -> false oil pixels + slicks   (must be 0 slicks)
* ``Lookalike``  -> false oil pixels + slicks   (must be 0 slicks)

Usage::

    venv/bin/python scratch_eval_checkpoint.py [checkpoint] [--scenes N]
                                               [--thresholds 0.5,0.7,0.9]
                                               [--normalisation per_tile|scene]
                                               [--downscale N]

Takes ~1 min per scene on CPU (less with --downscale).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tifffile as tiff

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "authentic testing" / "02_Test_images_and_ground_truth"
CLASSES = ("Oil", "No oil", "Lookalike")
NormalisationMode = Literal["per_tile", "scene"]


def _slick_count(mask: np.ndarray, min_area_px: int) -> int:
    if mask.sum() == 0:
        return 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area_px)


def _iou(mask: np.ndarray, gt: np.ndarray) -> float:
    if gt.shape != mask.shape:
        gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    union = np.logical_or(mask, gt).sum()
    if union == 0:
        return 1.0 if mask.sum() == 0 else 0.0
    return float(np.logical_and(mask, gt).sum()) / float(union)


def _precision_recall_f1(mask: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for binary oil detection."""
    if gt.shape != mask.shape:
        gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    tp = np.logical_and(mask, gt).sum()
    fp = np.logical_and(mask, ~gt).sum()
    fn = np.logical_and(~mask, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _apply_normalisation(image: np.ndarray, mode: NormalisationMode) -> np.ndarray:
    """Apply min-max normalisation to image channels."""
    img = image.astype(np.float32)
    if mode == "per_tile":
        # This is applied per-tile in the service; here we approximate by doing
        # whole-scene per-channel min-max as a proxy for the "scene" mode.
        # The actual per-tile normalisation happens in ModelService.
        pass  # No-op: service handles per-tile normalisation
    elif mode == "scene":
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                img[:, :, c] = (ch - mn) / (mx - mn)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a checkpoint on the corpus")
    parser.add_argument("checkpoint", nargs="?",
                        default="models/oil_spill_segmentation_model.pth.bak")
    parser.add_argument("--scenes", type=int, default=2, help="scenes per class")
    parser.add_argument("--thresholds", type=str, default="0.5,0.7,0.9")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--normalisation", type=str, choices=["per_tile", "scene"],
                        default="per_tile", help="Normalisation mode")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor for speed")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"[!] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    if not CORPUS.is_dir():
        print(f"[!] corpus not found: {CORPUS}", file=sys.stderr)
        return 2

    # ModelService resolves its checkpoint through the same env var the API
    # honours, so this harness scores exactly what the server would run.
    os.environ["OILSPILL_MODEL_PATH"] = str(checkpoint)
    os.environ["OILSPILL_NORMALISATION"] = args.normalisation
    sys.path.insert(0, str(ROOT))
    from app.api.deps import build_services
    from app.config import load_settings

    settings = load_settings(project_root=ROOT)
    services = build_services(settings, load_model=True)
    if not services.model.ready:
        print("[!] checkpoint refused to load", file=sys.stderr)
        return 2

    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Thresholds: {thresholds}  tile={args.tile_size}/{args.stride}  "
          f"scenes/class={args.scenes}  normalisation={args.normalisation}  downscale={args.downscale}\n")

    results: dict[str, dict[float, list]] = {}
    json_results = {
        "checkpoint": checkpoint.name,
        "thresholds": thresholds,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "normalisation": args.normalisation,
        "downscale": args.downscale,
        "classes": {}
    }
    try:
        for cls in CLASSES:
            scenes = sorted((CORPUS / "Images" / cls).glob("*.tif"))[: args.scenes]
            if not scenes:
                print(f"[!] no scenes for class {cls}", file=sys.stderr)
                return 2
            results[cls] = {thr: [] for thr in thresholds}
            json_results["classes"][cls] = {}
            for image_path in scenes:
                mask_path = CORPUS / "Mask" / cls / f"{image_path.stem}_segmentation.tif"
                gt = (tiff.imread(str(mask_path)) > 0).astype(np.uint8)
                image = tiff.imread(str(image_path))
                
                if args.downscale > 1:
                    h, w = image.shape[:2]
                    image = cv2.resize(image, (w // args.downscale, h // args.downscale),
                                       interpolation=cv2.INTER_AREA)
                    gt = cv2.resize(gt, (gt.shape[1] // args.downscale, gt.shape[0] // args.downscale),
                                    interpolation=cv2.INTER_NEAREST)
                
                started = time.time()
                inference = services.model.infer_probabilities(
                    image,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    tile_batch_size=4,
                )
                probs = inference.prob_maps
                label = probs.argmax(axis=0)
                for thr in thresholds:
                    mask = ((label == 1) & (probs[1] >= thr)).astype(np.uint8)
                    iou = _iou(mask, gt)
                    oil_px = int(mask.sum())
                    slicks = _slick_count(mask, 50)
                    precision, recall, f1 = _precision_recall_f1(mask, gt)
                    results[cls][thr].append((iou, oil_px, slicks, precision, recall, f1))
                    print(f"  {cls}/{image_path.name}  "
                          f"({time.time() - started:.0f}s, {inference.tiles_evaluated} tiles) "
                          f"IoU={iou:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} slicks={slicks}")
                
                # Store per-scene results for JSON
                for thr in thresholds:
                    rows = results[cls][thr]
                    if image_path.name not in json_results["classes"][cls]:
                        json_results["classes"][cls][image_path.name] = {}
                    json_results["classes"][cls][image_path.name][f"thr_{thr:.2f}"] = {
                        "iou": rows[-1][0],
                        "oil_pixels": rows[-1][1],
                        "slicks": rows[-1][2],
                        "precision": rows[-1][3],
                        "recall": rows[-1][4],
                        "f1": rows[-1][5]
                    }
    finally:
        services.close()

    print(f"\n{'class':<10} {'thr':>5} {'mean IoU':>9} {'mean P':>8} {'mean R':>8} {'mean F1':>8} "
          f"{'mean oil px':>12} {'slicks/scene':>13}")
    for cls in CLASSES:
        for thr in thresholds:
            rows = results[cls][thr]
            iou = float(np.mean([r[0] for r in rows]))
            precision = float(np.mean([r[3] for r in rows]))
            recall = float(np.mean([r[4] for r in rows]))
            f1 = float(np.mean([r[5] for r in rows]))
            pixels = int(np.mean([r[1] for r in rows]))
            slicks = [r[2] for r in rows]
            print(f"{cls:<10} {thr:>5.2f} {iou:>9.3f} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} "
                  f"{pixels:>12} {str(slicks):>13}")
            
            # Add summary to JSON
            if "summary" not in json_results["classes"][cls]:
                json_results["classes"][cls]["summary"] = {}
            json_results["classes"][cls]["summary"][f"thr_{thr:.2f}"] = {
                "mean_iou": iou,
                "mean_precision": precision,
                "mean_recall": recall,
                "mean_f1": f1,
                "mean_oil_pixels": pixels,
                "slicks_per_scene": slicks
            }

    print()
    best = max(thresholds)  # cleanest operating point in the calibration sweep
    oil_detected = any(s > 0 for _, _, s, _, _, _ in results["Oil"][best])
    clean_clear = all(s == 0 for _, _, s, _, _, _ in results["No oil"][best])
    lookalike_clear = all(s == 0 for _, _, s, _, _, _ in results["Lookalike"][best])
    print(f"VERDICT Oil       @ {best:.2f}: {'DETECTED' if oil_detected else 'MISSED'}")
    print(f"VERDICT No oil    @ {best:.2f}: {'CLEAR' if clean_clear else 'FALSE POSITIVES'}")
    print(f"VERDICT Lookalike @ {best:.2f}: {'CLEAR' if lookalike_clear else 'FALSE POSITIVES'}")
    
    # Add verdicts to JSON
    json_results["verdicts"] = {
        "oil_detected": oil_detected,
        "no_oil_clear": clean_clear,
        "lookalike_clear": lookalike_clear,
        "threshold_used": best
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results written to {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

