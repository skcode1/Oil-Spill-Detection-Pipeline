"""Production wrapper around the trained 3-class SAR segmentation U-Net.

Responsibilities
----------------
* **Lifecycle** - load the ``.pth`` checkpoint once at startup, keep the module
  in ``eval()`` mode on the best available device, and expose a thread-safe
  ``close()`` for the ASGI shutdown phase.
* **Fault tolerance** - a failed load never crashes the process: the error is
  recorded, ``ready`` stays ``False``, ``/health`` reports ``degraded`` and the
  inference endpoints answer ``503 model_unavailable`` with the original reason.
* **Fidelity** - the tile/sliding-window maths, channel layout
  (``[VV, VH, VV+VH]``), ``[0, 1]`` min-max pre-normalisation and the
  ``(x - mean) / std`` standardisation are the same recipe used by
  ``src/detect_and_vectorize.py`` and ``src/dataset.py``; only the execution is
  batched for throughput.

The class deliberately exposes NumPy in / NumPy out: no FastAPI types leak into
the ML layer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch

from app.config import AppSettings
from app.errors import ModelNotReadyError
from app.schemas import ModelInfo

# Reuse the canonical, already-validated helpers from the existing codebase.
from src.detect_and_vectorize import load_trained_model, sar_to_three_channel

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Probability maps and diagnostics produced by one sliding-window pass."""

    prob_maps: np.ndarray  # (num_classes, H, W) float32
    label_map: np.ndarray  # (H, W) int64 argmax over classes
    tiles_evaluated: int
    duration_ms: float
    #: (H, W) per-class confidence maps.
    oil_confidence: np.ndarray
    lookalike_confidence: np.ndarray
    background_confidence: np.ndarray
    ship_confidence: np.ndarray
    land_confidence: np.ndarray
    warnings: List[str] = field(default_factory=list)


class SpillSegmentationModel:
    """Thread-safe, lazily-loadable holder of the oil-spill segmentation model.

    Args:
        settings: Resolved application settings (checkpoint path, architecture,
            class indices, tiling defaults, normalisation statistics).
        logger_: Optional logger override (useful in tests).
    """

    def __init__(self, settings: AppSettings, logger_: Optional[logging.Logger] = None):
        self.settings = settings
        self.log = logger_ or logger
        self._lock = threading.RLock()
        self._model: Optional[torch.nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._load_error: Optional[str] = None
        self._checkpoint_size_bytes = 0
        self._loaded_at: Optional[str] = None
        self._load_duration_ms = 0.0
        self._resolved: Dict[str, Any] = {}
        self._norm_mean, self._norm_std = self._normalisation_stats()
        self._val_oil_iou: Optional[float] = None

    # ------------------------------------------------------------------ load
    @property
    def ready(self) -> bool:
        """True when a checkpoint is loaded and usable."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Device string the model currently runs on."""
        return str(self._device)

    def _resolve_device(self) -> torch.device:
        """Resolve ``auto``/``cpu``/``cuda``/``cuda:N`` into a torch device."""
        requested = str(self.settings.device or "auto").lower()
        if requested in ("auto", ""):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.log.warning(
                "CUDA requested (%s) but unavailable; falling back to CPU.", requested
            )
            return torch.device("cpu")
        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - bad env var
            self.log.warning("Invalid device %r (%s); using CPU.", requested, exc)
            return torch.device("cpu")

    @staticmethod
    def _detect_architecture(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
        """Infer the saved architecture from checkpoint state-dict keys.

        Returns ``'deeplabv3plus'`` when DeepLabV3Plus head/ASPP keys are present,
        ``'unet'`` when U-Net decoder/head keys are present, or ``None`` when the
        architecture cannot be determined.
        """
        keys = set(state_dict.keys())
        if any(k.startswith("decoder.aspp") for k in keys):
            return "deeplabv3plus"
        if any(k.startswith("decoder.blocks") for k in keys):
            return "unet"
        if any(k.startswith("segmentation_head.0") for k in keys):
            return "deeplabv3plus"
        if any(k.startswith("segmentation_head.conv") for k in keys):
            return "unet"
        return None

    def _normalisation_stats(self) -> Tuple[List[float], List[float]]:
        """Per-channel normalisation statistics (config.yaml -> augmentation)."""
        augmentation = self.settings.config.get("augmentation") or {}
        mean = [float(value) for value in (augmentation.get("norm_mean") or [0.5, 0.5, 0.5])]
        std = [float(value) for value in (augmentation.get("norm_std") or [0.5, 0.5, 0.5])]
        while len(mean) < 3:
            mean.append(mean[-1] if mean else 0.5)
        while len(std) < 3:
            std.append(std[-1] if std else 0.5)
        std = [value if value not in (0.0,) else 1.0 for value in std]
        return mean[:3], std[:3]

    def load(self, *, force: bool = False) -> ModelInfo:
        """Load the checkpoint (idempotent) and return the model descriptor.

        Args:
            force: Reload even when a model is already present.  With ``force``
                a failure raises :class:`ModelNotReadyError`; without it the
                error is recorded and reported through :meth:`info`.

        Raises:
            ModelNotReadyError: On a failed ``force`` reload.
        """
        with self._lock:
            if self._model is not None and not force:
                return self.info()

            checkpoint = self.settings.model_checkpoint
            if not checkpoint.is_file():
                self._load_error = f"Checkpoint not found at {checkpoint}"
                self.log.error("Model load skipped: %s", self._load_error)
                if force:
                    raise ModelNotReadyError(self._load_error)
                return self.info()

            self._device = self._resolve_device()
            started = time.perf_counter()
            try:
                # Load the raw checkpoint first so we can detect the architecture
                # it was saved for (3-class U-Net vs 5-class DeepLabV3Plus).  This
                # keeps the existing production checkpoint working while also
                # supporting the new 5-class DeepLabV3Plus checkpoints.
                checkpoint_state = torch.load(
                    str(checkpoint), map_location="cpu", weights_only=True
                )
                state_dict = None
                if isinstance(checkpoint_state, dict):
                    state_dict = (
                        checkpoint_state.get("model_state_dict")
                        or checkpoint_state.get("state_dict")
                        or checkpoint_state
                    )
                else:
                    state_dict = checkpoint_state

                detected_arch = self._detect_architecture(state_dict)
                requested_arch = str(self.settings.model.architecture).lower()
                if detected_arch:
                    architecture = detected_arch
                    self.log.info(
                        "Checkpoint architecture detected as %s (requested: %s)",
                        architecture,
                        requested_arch,
                    )
                else:
                    architecture = requested_arch
                    self.log.info(
                        "No architecture metadata in checkpoint; using requested %s",
                        architecture,
                    )

                num_classes = int(self.settings.model.classes)
                ckpt_classes = None
                if isinstance(checkpoint_state, dict) and "num_classes" in checkpoint_state:
                    ckpt_classes = int(checkpoint_state["num_classes"])
                if ckpt_classes is None and isinstance(state_dict, dict):
                    head_weight = state_dict.get("segmentation_head.0.weight")
                    if head_weight is not None:
                        ckpt_classes = int(head_weight.shape[0])
                if ckpt_classes:
                    num_classes = ckpt_classes
                    self.log.info(
                        "Inferred %d classes from checkpoint segmentation head.", num_classes
                    )

                if isinstance(checkpoint_state, dict) and "val_oil_iou" in checkpoint_state:
                    self._val_oil_iou = float(checkpoint_state["val_oil_iou"])
                    self.log.info(
                        "Checkpoint validation Oil Spill IoU: %.4f", self._val_oil_iou
                    )

                if architecture == "deeplabv3plus":
                    model = smp.DeepLabV3Plus(
                        encoder_name=self.settings.model.encoder,
                        encoder_weights=None,
                        in_channels=int(self.settings.model.in_channels),
                        classes=num_classes,
                    )
                else:
                    model = load_trained_model(
                        str(checkpoint),
                        self._device,
                        num_classes=num_classes,
                        in_channels=self.settings.model.in_channels,
                        encoder_name=self.settings.model.encoder,
                    )

                model.load_state_dict(state_dict, strict=False)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self._model = model
                self._load_error = None
            except Exception as exc:  # noqa: BLE001 - startup must never crash
                self._model = None
                self._load_error = f"{type(exc).__name__}: {exc}"
                self.log.error(
                    "Failed to load segmentation checkpoint %s: %s",
                    checkpoint,
                    exc,
                    exc_info=True,
                )
                if force:
                    raise ModelNotReadyError(
                        f"Could not load checkpoint {checkpoint}: {exc}"
                    ) from exc
                return self.info()

            self._load_duration_ms = (time.perf_counter() - started) * 1000.0
            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._checkpoint_size_bytes = int(checkpoint.stat().st_size)
            self._resolved = {
                "architecture": architecture,
                "encoder": self.settings.model.encoder,
                "in_channels": self.settings.model.in_channels,
                "num_classes": num_classes,
                "parameters": int(sum(p.numel() for p in model.parameters())),
            }
            info = self.info()
            self.log.info(
                "Model ready | %s/%s | in_channels=%s | classes=%s %s | device=%s | "
                "params=%s | load=%.0fms | checkpoint=%s",
                info.architecture,
                info.encoder,
                info.in_channels,
                info.num_classes,
                info.class_names,
                info.device,
                f"{info.parameters:,}",
                info.load_duration_ms,
                checkpoint.name,
            )
            return info

    def info(self) -> ModelInfo:
        """Current model descriptor (also returned when loading failed)."""
        model = self.settings.model
        inference = self.settings.inference
        return ModelInfo(
            ready=self.ready,
            architecture=str(self._resolved.get("architecture", model.architecture)),
            encoder=str(self._resolved.get("encoder", model.encoder)),
            in_channels=int(self._resolved.get("in_channels", model.in_channels)),
            num_classes=int(self._resolved.get("num_classes", model.classes)),
            class_names=list(model.class_names),
            oil_class_index=model.oil_class_index,
            oil_class_name=model.oil_class_name,
            device=self.device,
            checkpoint_path=str(self.settings.model_checkpoint),
            checkpoint_size_bytes=self._checkpoint_size_bytes,
            input_size=list(model.input_size),
            confidence_threshold=model.confidence_threshold,
            tile_size=inference.tile_size,
            stride=inference.effective_stride(),
            tile_batch_size=inference.tile_batch_size,
            loaded_at_utc=self._loaded_at,
            load_duration_ms=round(self._load_duration_ms, 2),
            parameters=int(self._resolved.get("parameters", 0)),
            error=self._load_error,
            val_oil_iou=self._val_oil_iou,
        )

    def close(self) -> None:
        """Release the model and free device memory (idempotent)."""
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self._resolved = {}
            if torch.cuda.is_available():  # pragma: no cover - GPU hosts only
                torch.cuda.empty_cache()
            self.log.info("Segmentation model released.")


    # ------------------------------------------------------------- inference
    def _tile_origins(
        self, height: int, width: int, tile: int, stride: int
    ) -> List[Tuple[int, int]]:
        """Sliding-window tile origins, identical to ``src/detect_and_vectorize``.

        Origins step from ``0``, so the final partial tile is zero-padded during
        the forward pass and cropped afterwards.  This keeps parity with the
        reference implementation (a determinism contract asserted by the test
        suite).
        """
        tile_h = min(tile, height)
        tile_w = min(tile, width)
        step = max(1, min(stride, tile_h, tile_w))
        return [
            (row, col) for row in range(0, height, step) for col in range(0, width, step)
        ]

    def _prepare_tensor(self, channels_hw_c: np.ndarray) -> torch.Tensor:
        """HWC float32 [0, 1] -> normalised CHW tensor on the model device."""
        chw = np.transpose(channels_hw_c, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).float()
        mean = torch.tensor(self._norm_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self._norm_std, dtype=torch.float32).view(-1, 1, 1)
        return ((tensor - mean) / std).to(self._device)

    def _select_channels(self, image_hw_c: np.ndarray) -> np.ndarray:
        """Trim the 3-channel composite to the channel count the model expects."""
        expected = int(self._resolved.get("in_channels", self.settings.model.in_channels))
        if expected in (1, 2, 3) and expected != image_hw_c.shape[2]:
            return image_hw_c[:, :, :expected]
        return image_hw_c


    def infer_probabilities(
        self,
        image: np.ndarray,
        *,
        tile_size: Optional[int] = None,
        stride: Optional[int] = None,
        tile_batch_size: Optional[int] = None,
    ) -> InferenceResult:
        """Run 3-class sliding-window inference over a full scene.

        Args:
            image: Raw scene array (``HxW``, ``HxWx2`` VV+VH or ``HxWx3``).
            tile_size: Tile edge in pixels (default: settings, 256).
            stride: Window step in pixels (default: settings, 128).  Using
                ``stride == tile_size`` gives the fastest, non-overlapping pass.
            tile_batch_size: Tiles per forward pass (default: settings, 4).

        Returns:
            :class:`InferenceResult` with averaged softmax probability maps.

        Raises:
            ModelNotReadyError: When no checkpoint is loaded.
        """
        with self._lock:
            if self._model is None:
                raise ModelNotReadyError(
                    self._load_error or "Segmentation model is not loaded."
                )

            inference = self.settings.inference
            tile = int(tile_size or inference.tile_size)
            step = int(stride or inference.effective_stride())
            batch_size = max(1, int(tile_batch_size or inference.tile_batch_size))
            normalisation = getattr(inference, "normalisation", "per_tile")

            # Channel layout must match training: physics-informed 3-channel
            # tensor (dB VV, local texture variance, VH/VV ratio).
            # Normalisation is applied inside sar_to_three_channel per channel.
            height, width = int(image.shape[0]), int(image.shape[1])

            num_classes = int(
                self._resolved.get("num_classes", self.settings.model.classes)
            )
            prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
            count_map = np.zeros((height, width), dtype=np.float32)

            origins = self._tile_origins(height, width, tile, step)
            tile_h = min(tile, height)
            tile_w = min(tile, width)
            started = time.perf_counter()

            for start in range(0, len(origins), batch_size):
                chunk = origins[start : start + batch_size]
                batch = np.zeros((len(chunk), tile_h, tile_w, 3), dtype=np.float32)
                spans: List[Tuple[int, int, int, int]] = []
                for index, (row, col) in enumerate(chunk):
                    end_row = min(row + tile_h, height)
                    end_col = min(col + tile_w, width)
                    # Normalisation is either done per-tile (default) or
                    # pre-applied to the whole scene ("scene" mode).
                    patch = self._select_channels(
                        sar_to_three_channel(
                            image[row:end_row, col:end_col],
                            normalisation=normalisation,
                        )
                    )
                    batch[index, : end_row - row, : end_col - col, :] = patch
                    spans.append((row, end_row, col, end_col))

                stacked = torch.stack(
                    [self._prepare_tensor(batch[i]) for i in range(len(chunk))]
                )
                with torch.no_grad():
                    logits = self._model(stacked)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                for index, (row, end_row, col, end_col) in enumerate(spans):
                    prob_sum[:, row:end_row, col:end_col] += probs[index][
                        :, : end_row - row, : end_col - col
                    ]
                    count_map[row:end_row, col:end_col] += 1.0

            prob_avg = prob_sum / np.maximum(count_map, 1.0)[None, :, :]
            label_map = np.argmax(prob_avg, axis=0).astype(np.int64)
            duration_ms = (time.perf_counter() - started) * 1000.0

            model_cfg = self.settings.model
            num_classes = int(
                self._resolved.get("num_classes", model_cfg.classes)
            )
            oil_index = min(max(model_cfg.oil_class_index, 0), num_classes - 1)
            lookalike_index = min(max(model_cfg.lookalike_class_index, 0), num_classes - 1)
            background_index = min(max(model_cfg.background_class_index, 0), num_classes - 1)
            ship_index = min(max(getattr(model_cfg, "ship_class_index", 3), 0), num_classes - 1)
            land_index = min(max(getattr(model_cfg, "land_class_index", 4), 0), num_classes - 1)

            self.log.info(
                "Inference complete | %dx%d | tiles=%d | tile=%d stride=%d batch=%d | classes=%d | %.0fms",
                width,
                height,
                len(origins),
                tile_h,
                step,
                batch_size,
                num_classes,
                duration_ms,
            )
            return InferenceResult(
                prob_maps=prob_avg,
                label_map=label_map,
                tiles_evaluated=len(origins),
                duration_ms=duration_ms,
                oil_confidence=prob_avg[oil_index],
                lookalike_confidence=prob_avg[lookalike_index],
                background_confidence=prob_avg[background_index],
                ship_confidence=prob_avg[ship_index],
                land_confidence=prob_avg[land_index],
            )

