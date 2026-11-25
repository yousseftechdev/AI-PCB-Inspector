"""Full-board PCB inspector runtime (TFLite, numpy + Pillow only).

This module is the single source of truth for inference. It is copied unchanged into
every exported run bundle, so the model you test locally behaves exactly like the
model evaluated in Colab.

Pipeline
--------
1. Letterbox the *entire* board image to the detector input size.
2. YOLO-style detector finds solder-joint boxes (one class: solder_joint).
3. Each box is expanded, cropped from the ORIGINAL image and classified.
4. Boxes are mapped back to original-image coordinates, NMS is applied, and a
   board-level verdict is produced.

No crop is ever supplied by the caller: the caller supplies a whole board.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["FullBoardInspector", "InspectionResult", "JointResult", "decode_yolo", "nms"]


# ---------------------------------------------------------------------------
# TFLite loading / tensor helpers
# ---------------------------------------------------------------------------


def _interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter

            return Interpreter
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "No TFLite runtime found. Install 'tflite-runtime' on the Pi, "
                "or 'tensorflow' on a PC (both expose tflite.Interpreter)."
            ) from exc


def _quantize(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert float values to the model input dtype (int8/uint8/float32)."""
    dtype = np.dtype(detail["dtype"])
    if dtype == np.float32:
        return values.astype(np.float32, copy=False)
    if dtype in (np.int8, np.uint8):
        scale = float(detail["quantization"][0] or 1.0)
        zero = int(detail["quantization"][1] or 0)
        q = np.round(values / scale) + zero
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)
    raise RuntimeError(f"Unsupported TFLite input dtype: {dtype}")


def _dequantize(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert a model output tensor back to float."""
    values = np.asarray(values)
    dtype = np.dtype(detail["dtype"])
    if dtype == np.float32:
        return values.astype(np.float32, copy=False)
    if dtype in (np.int8, np.uint8):
        scale = float(detail["quantization"][0] or 1.0)
        zero = int(detail["quantization"][1] or 0)
        return (values.astype(np.float32) - zero) * scale
    raise RuntimeError(f"Unsupported TFLite output dtype: {dtype}")


def _set_input(interpreter, values: np.ndarray) -> None:
    detail = interpreter.get_input_details()[0]
    values = _quantize(values, detail)
    expected = tuple(detail["shape"])
    if tuple(values.shape) != expected:
        # Most TFLite exports are NHWC [1,H,W,3]; a few YOLO exports are NCHW.
        if len(expected) == 4 and expected[0] == 1 and expected[3] == 3:
            values = np.transpose(values, (0, 3, 1, 2))
        elif len(expected) == 4 and expected[0] == 1 and expected[1] == 3:
            values = np.transpose(values, (0, 2, 3, 1))
    interpreter.set_tensor(detail["index"], np.ascontiguousarray(values))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def letterbox(
    image: np.ndarray, size: int, fill: int = 114
) -> tuple[np.ndarray, float, float, float]:
    """Resize with unchanged aspect ratio and pad to (size, size).

    Returns (resized_rgb_0_255, scale, pad_x, pad_y) where an input pixel (x, y)
    maps to (x * scale + pad_x, y * scale + pad_y) in the padded image.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    padded = np.full((size, size, 3), fill, dtype=np.float32)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    padded[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return padded, scale, pad_x, pad_y


def _square_crop(image: np.ndarray, box: Sequence[float], expand: float = 0.20) -> np.ndarray:
    """Crop a (possibly expanded) square around a joint from the original image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * expand)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x0 + side))), min(h, int(np.ceil(y0 + side)))
    if x1i <= x0i or y1i <= y0i:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return image[y0i:y1i, x0i:x1i]


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_b = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_o = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_b + area_o - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int
) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns indices in descending score order."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size and len(keep) < max_detections:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_one_to_many(boxes[best], boxes[rest])
        order = rest[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(
    output: np.ndarray, num_classes: int = 1, conf_threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLO11n/v8 export tensor into boxes, scores and class ids.

    Accepts either [1, 4+nc, N] or [1, N, 4+nc]. Boxes are returned in the
    detector's input-pixel coordinate space as xyxy.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Unexpected YOLO output shape {arr.shape}; expected [1,C,N] or [1,N,C]")
    c = 4 + num_classes
    if arr.shape[1] == c:
        pred = arr[0]  # [C, N]
    elif arr.shape[2] == c:
        pred = np.transpose(arr[0], (1, 0))  # [N, C]
    else:
        raise ValueError(
            f"Unexpected YOLO output shape {arr.shape}; expected one dimension to be {c}"
        )

    cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
    cls_scores = pred[4 : 4 + num_classes] if num_classes > 1 else pred[4:5]
    class_ids = np.argmax(cls_scores, axis=0)
    scores = cls_scores[class_ids, np.arange(cls_scores.shape[1])]
    mask = scores >= conf_threshold
    if not np.any(mask):
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    boxes = np.stack(
        [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1
    ).astype(np.float32)
    return boxes, scores[mask].astype(np.float32), class_ids[mask].astype(np.int64)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class JointResult:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # original-image pixels, xyxy
    detector_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box_xyxy": [int(v) for v in self.box],
            "detector_confidence": round(float(self.detector_confidence), 4),
        }


@dataclass
class InspectionResult:
    verdict: str  # PASS | FAIL | REVIEW
    joints: list[JointResult] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    annotated_image: np.ndarray | None = None
    image_size: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "joint_count": len(self.joints),
            "counts": self.counts,
            "joints": [joint.to_dict() for joint in self.joints],
            "timings_ms": {key: round(value, 2) for key, value in self.timings_ms.items()},
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Main inspector
# ---------------------------------------------------------------------------


class FullBoardInspector:
    """Load a downloaded run bundle and inspect complete board images."""

    FAIL_LABELS = ("solder_bridge", "insufficient_solder")
    REVIEW_LABELS = ("other", "low_confidence")
    COLORS = {
        "good": (40, 190, 90),
        "solder_bridge": (235, 55, 55),
        "insufficient_solder": (245, 145, 40),
        "other": (240, 190, 50),
        "low_confidence": (150, 150, 150),
    }

    def __init__(self, bundle_dir: str | Path, num_threads: int = 2):
        self.bundle = Path(bundle_dir)
        self.config = json.loads((self.bundle / "preprocess.json").read_text(encoding="utf-8"))
        self.labels = [
            line.strip()
            for line in (self.bundle / "labels.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.labels:
            raise ValueError("labels.txt is empty")

        Interpreter = _interpreter_class()
        self.detector = Interpreter(
            model_path=str(self.bundle / self.config["detector"]["file"]), num_threads=num_threads
        )
        self.classifier = Interpreter(
            model_path=str(self.bundle / self.config["classifier"]["file"]), num_threads=num_threads
        )
        self.detector.allocate_tensors()
        self.classifier.allocate_tensors()
        self.detector_size = int(self.config["detector"]["input_size"])
        self.classifier_size = int(self.config["classifier"]["input_size"])
        self.detector_input_range = self.config["detector"].get("input_range", [0, 1])
        self.classifier_input_range = self.config["classifier"].get("input_range", [0, 1])
        self.conf_threshold = float(self.config.get("score_threshold", 0.25))
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))
        self.crop_expand = float(self.config.get("crop_expand", 0.20))
        self.max_detections = int(self.config.get("max_detections", 200))
        self.detector_num_classes = int(self.config["detector"].get("num_classes", 1))
        self.det_output_layout = self.config["detector"].get("output_layout", "auto")
        self.supported_classes = set(self.labels)

        det_out = self.detector.get_output_details()
        if len(det_out) != 1:
            raise RuntimeError(
                f"Detector must have exactly one output tensor, got {len(det_out)}. "
                "Re-export with the Colab notebook's supported YOLO export path."
            )
        self.det_output_detail = det_out[0]
        shape = tuple(self.det_output_detail["shape"])
        expected = 4 + self.detector_num_classes
        if self.det_output_layout == "auto":
            if len(shape) == 3 and (shape[1] == expected or shape[2] == expected):
                self.det_output_layout = "c_first" if shape[1] == expected else "n_first"
            else:
                raise RuntimeError(
                    f"Unrecognised detector output shape {shape}; expected dimension {expected}. "
                    "Check preprocess.json detector.output_layout."
                )
        self.notes: list[str] = []
        for label in FullBoardInspector.FAIL_LABELS:
            if label not in self.supported_classes:
                self.notes.append(
                    f"'{label}' has no verified training examples in this bundle; "
                    "it can never be predicted."
                )

    # ---- internals ---------------------------------------------------------

    def _scale_for_model(self, values: np.ndarray, input_range: Sequence[float]) -> np.ndarray:
        lo, hi = float(input_range[0]), float(input_range[1])
        return values * ((hi - lo) / 255.0) + lo

    def _detect(self, full_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded, scale, pad_x, pad_y = letterbox(full_image, self.detector_size)
        padded = self._scale_for_model(padded, self.detector_input_range)
        _set_input(self.detector, padded)
        self.detector.invoke()
        raw = self.detector.get_tensor(self.det_output_detail["index"])
        raw = _dequantize(raw, self.det_output_detail)
        if self.det_output_layout == "c_first":
            raw = np.transpose(raw, (0, 2, 1))
        # raw is now [1, N, 4+nc]
        num_classes = self.detector_num_classes
        cx, cy, bw, bh = raw[0, :, 0], raw[0, :, 1], raw[0, :, 2], raw[0, :, 3]
        scores = np.max(raw[0, :, 4 : 4 + num_classes], axis=1)
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        boxes = np.stack(
            [cx[keep] - bw[keep] / 2, cy[keep] - bh[keep] / 2,
             cx[keep] + bw[keep] / 2, cy[keep] + bh[keep] / 2],
            axis=1,
        )
        boxes[:, 0::2] = (boxes[:, 0::2] - pad_x) / scale
        boxes[:, 1::2] = (boxes[:, 1::2] - pad_y) / scale
        h, w = full_image.shape[:2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        return boxes.astype(np.float32), scores[keep].astype(np.float32)

    def _classify(self, crop: np.ndarray) -> tuple[str, float]:
        resized = np.asarray(
            Image.fromarray(crop).resize(
                (self.classifier_size, self.classifier_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        resized = self._scale_for_model(resized, self.classifier_input_range)
        _set_input(self.classifier, resized)
        self.classifier.invoke()
        detail = self.classifier.get_output_details()[0]
        probs = _dequantize(self.classifier.get_tensor(detail["index"]), detail).reshape(-1)
        if probs.size != len(self.labels):
            raise RuntimeError(
                f"Classifier output has {probs.size} values but labels.txt has {len(self.labels)}"
            )
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])

    def _annotate(self, image: np.ndarray, joints: Iterable[JointResult]) -> np.ndarray:
        out = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(out)
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None
        for joint in joints:
            color = self.COLORS.get(joint.label, (200, 200, 200))
            x1, y1, x2, y2 = joint.box
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, out.width // 400))
            text = f"{joint.label} {joint.confidence:.2f}"
            draw.text((x1, max(0, y1 - 14)), text, fill=color, font=font)
        return np.asarray(out)

    # ---- public API --------------------------------------------------------

    def inspect(self, image: Image.Image | np.ndarray, annotate: bool = True) -> InspectionResult:
        started = time.perf_counter()
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            rgb = arr[..., :3].astype(np.uint8)
        h, w = rgb.shape[:2]

        t0 = time.perf_counter()
        raw_boxes, det_scores = self._detect(rgb)
        t_detect = (time.perf_counter() - t0) * 1000.0

        if len(raw_boxes):
            kept = nms(raw_boxes, det_scores, self.iou_threshold, self.max_detections)
            raw_boxes, det_scores = raw_boxes[kept], det_scores[kept]

        joints: list[JointResult] = []
        t0 = time.perf_counter()
        for box, det_score in zip(raw_boxes, det_scores):
            crop = _square_crop(rgb, box, self.crop_expand)
            if crop.size == 0:
                continue
            label, confidence = self._classify(crop)
            if confidence < self.conf_threshold:
                label = "low_confidence"
            joints.append(
                JointResult(
                    label=label,
                    confidence=confidence if label != "low_confidence" else det_score,
                    box=tuple(int(round(v)) for v in box),
                    detector_confidence=float(det_score),
                )
            )
        t_classify = (time.perf_counter() - t0) * 1000.0

        counts: dict[str, int] = {}
        for joint in joints:
            counts[joint.label] = counts.get(joint.label, 0) + 1
        if any(joint.label in FullBoardInspector.FAIL_LABELS for joint in joints):
            verdict = "FAIL"
        elif any(joint.label in FullBoardInspector.REVIEW_LABELS for joint in joints):
            verdict = "REVIEW"
        elif not joints:
            verdict = "REVIEW"
        else:
            verdict = "PASS"

        total = (time.perf_counter() - started) * 1000.0
        result = InspectionResult(
            verdict=verdict,
            joints=joints,
            counts=counts,
            timings_ms={
                "detect": t_detect,
                "classify": t_classify,
                "total": total,
            },
            image_size=(w, h),
            notes=list(self.notes),
        )
        if annotate:
            t0 = time.perf_counter()
            result.annotated_image = self._annotate(rgb, joints)
            result.timings_ms["annotate"] = (time.perf_counter() - t0) * 1000.0
        return result
