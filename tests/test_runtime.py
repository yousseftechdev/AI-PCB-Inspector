"""CPU-only contract/regression tests.

Run from the repo root:  python -m pytest tests -q
No model weights, dataset or Pi hardware required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runtime"))
sys.path.insert(0, str(REPO / "training"))

from inspector import decode_yolo, letterbox, nms  # noqa: E402
import fullboard_colab as pipeline  # noqa: E402


def test_no_good_is_never_a_bridge():
    config = pipeline.load_config(REPO / "training" / "configs" / "fullboard_config.json")
    soldef = next(dataset for dataset in config["datasets"] if dataset["name"] == "soldef")
    mapping = pipeline.class_mapping(config, soldef)
    assert mapping["no_good"]["label"] == "other"
    assert all(entry["label"] != "solder_bridge" for entry in mapping.values())

    warnings = pipeline.check_label_policy(config, {"soldef": mapping}, strict=False)
    assert any("visually reviewed" in warning for warning in warnings)
    with pytest.raises(RuntimeError, match="NOT visually reviewed"):
        pipeline.check_label_policy(config, {"soldef": mapping}, strict=True)


def test_bridge_class_can_be_absent_without_a_reviewed_source():
    config = pipeline.load_config(REPO / "training" / "configs" / "fullboard_config.json")
    mapping = pipeline.class_mapping(config, config["datasets"][0])
    warnings = pipeline.check_label_policy(config, {"soldef": mapping}, strict=False)
    assert any("No verified solder_bridge source" in warning for warning in warnings)


def test_decode_yolo_both_layouts():
    # One object: centre (100, 80), size 40x60, confidence 0.9.
    pred_c_first = np.zeros((1, 5, 10), dtype=np.float32)
    pred_c_first[0, 0, 0], pred_c_first[0, 1, 0] = 100, 80
    pred_c_first[0, 2, 0], pred_c_first[0, 3, 0] = 40, 60
    pred_c_first[0, 4, 0] = 0.9
    boxes, scores, _ = decode_yolo(pred_c_first, num_classes=1, conf_threshold=0.25)
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [80, 50, 120, 110])
    assert scores[0] == pytest.approx(0.9)

    pred_n_first = np.transpose(pred_c_first, (0, 2, 1))
    boxes2, scores2, _ = decode_yolo(pred_n_first, num_classes=1, conf_threshold=0.25)
    np.testing.assert_allclose(boxes, boxes2)
    np.testing.assert_allclose(scores, scores2)


def test_nms_suppresses_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.7, 0.9, 0.6], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.4, max_detections=10)
    assert keep.tolist() == [1, 2]


def test_letterbox_preserves_aspect_and_maps_back():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(image, 640)
    assert scale == pytest.approx(1.6)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(160.0)


def test_board_family_regex():
    assert pipeline.board_family("CS2/R3/V1.1/Setup1/img_004.png", "soldef").endswith(
        "CS2_R3_V1.1_SETUP1"
    )
    assert pipeline.board_family("random_export_12.png", "roboflow") == "roboflow:random_export_12"


def test_notebook_and_config_are_valid_json():
    notebook = json.loads((REPO / "training" / "PCB_Inspector_Colab.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    config = json.loads((REPO / "training" / "configs" / "fullboard_config.json").read_text())
    assert "solder_bridge" in config["target_classes"]
