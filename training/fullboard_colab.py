#!/usr/bin/env python3
"""Colab-only full-board training pipeline for the PCB defect inspector.

Run from one Colab notebook (see training/PCB_Inspector_Colab.ipynb):

    python training/fullboard_colab.py audit --config training/configs/fullboard_config.json
    # human: review audit/*_contact.jpg and flip reviewed flags in the config
    python training/fullboard_colab.py prepare --config ... --output /content/pcb_fullboard_run
    python training/fullboard_colab.py train --work /content/pcb_fullboard_run --device auto
    python training/fullboard_colab.py export --work /content/pcb_fullboard_run
    python training/fullboard_colab.py publish --run-dir <bundle dir>
    python training/fullboard_colab.py push

Design rules enforced here
--------------------------
* No local-agent step. Everything (download, audit, prepare, train, export,
  handoff, git push) happens in Colab.
* The model consumes FULL BOARD images. Joints are found by a one-class
  detector; there is no "please send me a crop" step.
* Label semantics are explicit. SolDef `no_good` is NEVER a solder bridge.
  Unverified mappings cannot be used with --strict-labels (the default).
* Every run gets a new immutable UTC id, a saved config, metrics and checksums.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_ROOT = Path("/content/drive/MyDrive/AI-PCB-Inspector")
DATA_CACHE = DRIVE_ROOT / "data_cache"
RUNS_ROOT = DRIVE_ROOT / "runs"
HANDOFF_PATH = DRIVE_ROOT / "handoff" / "HANDOFF.json"
CHANGELOG_PATH = DRIVE_ROOT / "handoff" / "CHANGELOG.md"
GITHUB_REMOTE = "https://github.com/yousseftechdev/AI-PCB-Inspector.git"
GITHUB_REPO = "yousseftechdev/AI-PCB-Inspector"

BOARD_FAMILY_RE = re.compile(r"(CS\d+|R\d+|V\d+(?:\.\d+)?|Setup\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_secret(name: str) -> str | None:
    """Read a secret from Colab Secrets first, then from the environment."""
    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name)


def require_drive() -> None:
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive. Mount it in the notebook and retry."
        )
    for path in (DATA_CACHE, RUNS_ROOT, HANDOFF_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = config.get("target_classes")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("target_classes must be a non-empty list of unique names")
    return config


def find_coco_files(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    results: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return results
    # Read obvious COCO exports first: the Kaggle cache also contains hundreds of
    # labelme JSONs with megabytes of embedded imageData.
    paths = sorted(
        root.rglob("*.json"),
        key=lambda path: (0 if "coco" in path.name.lower() else 1, str(path)),
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if all(isinstance(payload.get(key), list) for key in ("images", "annotations", "categories")):
            results.append((path, payload))
    return results


def load_labelme_files(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load per-image labelme/VIA JSON exports (``shapes`` + ``imagePath``)."""
    results: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return results
    for path in sorted(root.rglob("*.json")):
        if "coco" in path.name.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("shapes"), list) and payload.get("imagePath"):
            results.append((path, payload))
    return results


def find_labelme_files(root: Path) -> list[Path]:
    return [path for path, _ in load_labelme_files(root)]


def convert_labelme_to_coco(root: Path) -> Path | None:
    """Convert SolDef's labelme JSONs into one COCO file at the cache root.

    The Kaggle archive ships 428 per-image labelme JSONs in ``SolDef_AI/Labeled``
    and the same images under ``SolDef_AI/Dataset/CS*/R*/V*/Setup*/``.  We point
    each COCO image at the Dataset copy so the board-family split can use the
    path tokens instead of falling back to the (non-unique) WIN_* basenames.
    """
    output = root / "_annotations.coco.json"
    if output.exists():
        return output
    labelme_files = load_labelme_files(root)
    if not labelme_files:
        return None

    labels = sorted(
        {
            str(shape.get("label", ""))
            for _, payload in labelme_files
            for shape in payload["shapes"]
            if shape.get("label")
        }
    )
    if not labels:
        raise RuntimeError(f"No labels found in labelme annotations under {root}")
    category_ids = {name: index + 1 for index, name in enumerate(labels)}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 0
    for image_id, (json_path, payload) in enumerate(labelme_files, start=1):
        image_name = Path(str(payload["imagePath"])).name
        sibling = json_path.parent / image_name
        candidates = [path for path in root.rglob(image_name) if path.is_file()]
        dataset_copies = [
            path for path in candidates if "dataset" in path.relative_to(root).as_posix().lower()
        ]
        if len(dataset_copies) == 1:
            image_path = dataset_copies[0]
        elif dataset_copies:
            digest = sha256_file(sibling) if sibling.is_file() else None
            matches = [path for path in dataset_copies if digest and sha256_file(path) == digest]
            image_path = matches[0] if len(matches) == 1 else dataset_copies[0]
        elif sibling.is_file():
            image_path = sibling
        elif len(candidates) == 1:
            image_path = candidates[0]
        else:
            print(f"WARN: cannot resolve image {image_name!r} for {json_path.name}; skipped")
            continue

        width = int(payload.get("imageWidth", 0))
        height = int(payload.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(root).as_posix(),
                "width": width,
                "height": height,
                "license": 1,
            }
        )
        for shape in payload["shapes"]:
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            try:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
            except (TypeError, ValueError):
                continue
            x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
            x1, y1 = min(width, max(xs)), min(height, max(ys))
            if x1 <= x0 or y1 <= y0:
                continue
            annotation_id += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[str(shape["label"])],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                    "segmentation": [[float(value) for point in points for value in point]],
                }
            )

    coco = {
        "info": {
            "description": "SolDef_AI labelme annotations converted to COCO by fullboard_colab.py",
            "converted_at": utc_id(),
            "source": "https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        },
        "licenses": [{"id": 1, "name": "GPL-3.0 (Kaggle API metadata)"}],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "solder_joint"}
            for name in labels
        ],
    }
    output.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(
        f"Converted {len(images)} labelme images / {len(annotations)} annotations "
        f"to {output}"
    )
    return output


def ensure_annotations(config: dict[str, Any]) -> None:
    """Make sure every configured dataset has a COCO file to audit/prepare."""
    for dataset in config["datasets"]:
        root = Path(dataset["root"])
        if not root.is_dir():
            continue
        if dataset.get("annotation_format") == "labelme":
            convert_labelme_to_coco(root)  # idempotent; cheap once the COCO file exists
            continue
        if find_coco_files(root):
            continue
        if find_labelme_files(root):
            convert_labelme_to_coco(root)


def resolve_image(root: Path, file_name: str, annotation_dir: Path | None = None) -> Path:
    candidates = [root / file_name]
    if annotation_dir is not None:
        candidates.append(annotation_dir / file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve image {file_name!r} under {root} ({len(matches)} matches)")


def board_family(file_name: str, source: str) -> str:
    tokens = BOARD_FAMILY_RE.findall(file_name)
    if tokens:
        return source + ":" + "_".join(token.upper() for token in tokens)
    # Roboflow exports often use random names; fall back to the whole image id.
    return source + ":" + Path(file_name).stem


def class_mapping(config: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Return category -> {label, reviewed, reason}; None label means excluded."""
    configured = dataset.get("class_map", {})
    mapping: dict[str, Any] = {}
    for category in configured:
        entry = configured[category]
        if isinstance(entry, str):
            entry = {"label": entry, "reviewed": False, "reason": "legacy string mapping"}
        label = entry.get("label")
        if label is not None and label not in config["target_classes"]:
            raise ValueError(f"{dataset['name']}: unknown target label {label!r}")
        mapping[category] = {
            "label": label,
            "reviewed": bool(entry.get("reviewed", False)),
            "reason": entry.get("reason", ""),
        }
    return mapping


def check_label_policy(
    config: dict[str, Any], used_mappings: dict[str, dict[str, Any]], strict: bool
) -> list[str]:
    warnings: list[str] = []
    for name, mapping in used_mappings.items():
        for category, entry in mapping.items():
            if entry["label"] is None:
                continue
            if not entry["reviewed"]:
                message = f"{name}:{category} -> {entry['label']} is NOT visually reviewed"
                if strict:
                    raise RuntimeError(
                        message + ". Look at audit/<source>/<category>_contact.jpg and set "
                        "'reviewed': true in the config, or pass --allow-provisional explicitly."
                    )
                warnings.append(message + " (provisional)")
    if "solder_bridge" in config["target_classes"]:
        bridge_sources = [
            name
            for name, mapping in used_mappings.items()
            if any(entry["label"] == "solder_bridge" for entry in mapping.values())
        ]
        if not bridge_sources:
            warnings.append(
                "No verified solder_bridge source. The 'solder_bridge' class will have zero "
                "examples and will be REMOVED from labels.txt so it can never be predicted."
            )
    return warnings


# ---------------------------------------------------------------------------
# 1. Download + audit
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    require_drive()
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    datasets = {dataset["name"]: dataset for dataset in config["datasets"]}

    soldef = datasets.get("soldef")
    if soldef:
        soldef_dir = Path(soldef["root"])
        soldef_dir.mkdir(parents=True, exist_ok=True)
        if find_coco_files(soldef_dir) or find_labelme_files(soldef_dir):
            print(f"SolDef cached at {soldef_dir}")
        else:
            kaggle_user = get_secret("KAGGLE_USERNAME")
            kaggle_key = get_secret("KAGGLE_KEY")
            if not (kaggle_user and kaggle_key):
                raise RuntimeError(
                    "SolDef is not cached. Add KAGGLE_USERNAME and KAGGLE_KEY to Colab "
                    "Secrets and rerun download."
                )
            env = os.environ.copy()
            env["KAGGLE_USERNAME"] = kaggle_user
            env["KAGGLE_KEY"] = kaggle_key
            subprocess.run(
                ["kaggle", "datasets", "download", "-d",
                 soldef.get("kaggle_slug",
                            "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection"),
                 "-p", str(soldef_dir), "--unzip"],
                check=True,
                env=env,
            )

    # Roboflow is opt-in: only a dataset with a configured workspace/project is
    # downloaded, and only when an API key exists. No project is configured for
    # this project, so the hardcoded workspace/project call is gone.
    roboflow_datasets = [
        dataset for dataset in config["datasets"]
        if dataset.get("workspace") and dataset.get("project")
    ]
    if not roboflow_datasets:
        print("Roboflow: disabled (no workspace/project configured); skipping.")
    else:
        rf_key = get_secret("ROBOFLOW_API_KEY")
        if not rf_key:
            print("ROBOFLOW_API_KEY missing; configured Roboflow dataset(s) skipped.")
        else:
            from roboflow import Roboflow

            rf = Roboflow(api_key=rf_key)
            for dataset in roboflow_datasets:
                target = Path(dataset["root"])
                target.mkdir(parents=True, exist_ok=True)
                if find_coco_files(target):
                    print(f"Roboflow cached at {target}")
                    continue
                project = rf.workspace(dataset["workspace"]).project(dataset["project"])
                project.version(int(dataset.get("version", 1))).download(
                    "coco", location=str(target)
                )

    ensure_annotations(config)
    print(f"Data cache: {DATA_CACHE}")


def cmd_audit(args: argparse.Namespace) -> None:
    require_drive()
    config = load_config(args.config)
    ensure_annotations(config)
    from PIL import Image, ImageDraw

    (RUNS_ROOT / "audit").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"generated_at": utc_id(), "sources": {}}
    total_entries = 0
    for dataset in config["datasets"]:
        root = Path(dataset["root"])
        entries = find_coco_files(root)
        total_entries += len(entries)
        print(f"\n[{dataset['name']}] {len(entries)} COCO file(s) under {root}")
        source_summary: list[dict[str, Any]] = []
        for annotation_path, coco in entries:
            categories = {item["id"]: item["name"] for item in coco["categories"]}
            counts = Counter(
                categories.get(item["category_id"], str(item["category_id"]))
                for item in coco["annotations"]
            )
            images = {item["id"]: item for item in coco["images"]}
            mapping = class_mapping(config, dataset)
            item = {
                "annotation_file": str(annotation_path),
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "categories": categories,
                "counts": dict(counts),
                "mapping": mapping,
            }
            source_summary.append(item)
            print(json.dumps({"file": annotation_path.name, "counts": dict(counts)}, indent=2))

            samples: dict[str, list[Image.Image]] = defaultdict(list)
            for ann in coco["annotations"]:
                category = categories.get(ann["category_id"], str(ann["category_id"]))
                if len(samples[category]) >= args.previews:
                    continue
                info = images.get(ann["image_id"])
                if not info:
                    continue
                image_path = resolve_image(root, info["file_name"], annotation_path.parent)
                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                x, y, bw, bh = map(float, ann.get("bbox", [0, 0, image.width, image.height]))
                pad = max(bw, bh) * 0.35
                box = (
                    max(0, int(x - pad)), max(0, int(y - pad)),
                    min(image.width, int(x + bw + pad)), min(image.height, int(y + bh + pad)),
                )
                crop = image.crop(box)
                crop.thumbnail((260, 260))
                samples[category].append(crop.copy())

            audit_dir = RUNS_ROOT / "audit" / dataset["name"]
            audit_dir.mkdir(parents=True, exist_ok=True)
            for category, crops in samples.items():
                if not crops:
                    continue
                cols = min(4, len(crops))
                rows = (len(crops) + cols - 1) // cols
                sheet = Image.new("RGB", (cols * 260, rows * 250), (255, 255, 255))
                draw = ImageDraw.Draw(sheet)
                for index, crop in enumerate(crops):
                    x0, y0 = (index % cols) * 260, (index // cols) * 250
                    sheet.paste(crop, (x0, y0 + 20))
                    draw.text((x0 + 4, y0 + 3), category, fill=(0, 0, 0))
                out = audit_dir / f"{category}_contact.jpg"
                sheet.save(out, quality=90)
                print(f"  contact sheet: {out}")
        summary["sources"][dataset["name"]] = source_summary
    if total_entries == 0:
        raise RuntimeError(
            "No COCO annotation files found. Run `download` first and check the dataset "
            "roots in training/configs/fullboard_config.json."
        )
    summary_path = RUNS_ROOT / "audit" / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAudit summary: {summary_path}")
    print("Review the contact sheets, then set 'reviewed': true only for mappings you can defend.")


# ---------------------------------------------------------------------------
# 2. Prepare detector + classifier datasets
# ---------------------------------------------------------------------------


def _bbox_from_annotation(ann: dict[str, Any], image_size: tuple[int, int]) -> list[float] | None:
    bbox = ann.get("bbox")
    if bbox and len(bbox) == 4:
        x, y, w, h = map(float, bbox)
        if w > 1 and h > 1:
            return [x, y, w, h]
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        points = segmentation[0]
        if len(points) >= 6:
            xs, ys = points[0::2], points[1::2]
            return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return None


def cmd_prepare(args: argparse.Namespace) -> None:
    require_drive()
    config = load_config(args.config)
    ensure_annotations(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    strict = not args.allow_provisional

    detector_records: list[dict[str, Any]] = []
    classifier_records: list[dict[str, Any]] = []
    used_mappings: dict[str, dict[str, Any]] = {}
    dataset_summaries: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        root = Path(dataset["root"])
        mapping = class_mapping(config, dataset)
        used_mappings[dataset["name"]] = mapping
        counts: Counter[str] = Counter()
        for annotation_path, coco in find_coco_files(root):
            categories = {item["id"]: item["name"] for item in coco["categories"]}
            images = {item["id"]: item for item in coco["images"]}
            for ann in coco["annotations"]:
                category = categories.get(ann["category_id"])
                if category is None or ann.get("iscrowd", 0):
                    continue
                info = images.get(ann["image_id"])
                if not info:
                    continue
                image_path = resolve_image(root, info["file_name"], annotation_path.parent)
                bbox = _bbox_from_annotation(ann, (0, 0))
                if bbox is None:
                    continue
                family = board_family(info["file_name"], dataset["name"])
                detector_records.append(
                    {
                        "image_path": str(image_path), "bbox": bbox,
                        "group": family, "source": dataset["name"],
                        "source_image": info["file_name"],
                    }
                )
                entry = mapping.get(category)
                if entry and entry["label"]:
                    counts[entry["label"]] += 1
                    classifier_records.append(
                        {
                            "image_path": str(image_path), "bbox": bbox,
                            "group": family, "source": dataset["name"],
                            "source_image": info["file_name"], "category": category,
                            "label": entry["label"],
                        }
                    )
        dataset_summaries.append(
            {"name": dataset["name"], "class_counts": dict(counts), "root": str(root)}
        )

    warnings = check_label_policy(config, used_mappings, strict)
    available_labels = {record["label"] for record in classifier_records}
    final_classes = [name for name in config["target_classes"] if name in available_labels]
    if not final_classes:
        raise RuntimeError("No usable classifier examples after applying the class map.")
    dropped = [name for name in config["target_classes"] if name not in final_classes]
    warnings.extend(f"Class '{name}' has no verified examples and was dropped" for name in dropped)
    for warning in warnings:
        print("WARN:", warning)

    # Split by board family when possible; fall back to source image with a flag.
    groups = sorted({record["group"] for record in detector_records})
    if len(groups) < 5:
        raise RuntimeError(f"Only {len(groups)} distinct groups; need at least 5 for an honest split.")
    rng = random.Random(config["training"]["seed"])
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * config["training"]["test_fraction"]))
    n_val = max(1, round(len(groups) * config["training"]["val_fraction"]))
    split_of_group: dict[str, str] = {}
    for group in groups[:n_test]:
        split_of_group[group] = "test"
    for group in groups[n_test : n_test + n_val]:
        split_of_group[group] = "val"
    for group in groups[n_test + n_val :]:
        split_of_group[group] = "train"

    # Build YOLO detector dataset (one label file per image).
    det_root = output / "detector_dataset"
    for split in ("train", "val", "test"):
        (det_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in detector_records:
        by_image[record["image_path"]].append(record)
    from PIL import Image

    for image_path, annotations in by_image.items():
        split = split_of_group[annotations[0]["group"]]
        with Image.open(image_path) as img:
            width, height = img.size
        image_name = f"{Path(image_path).parent.name}_{Path(image_path).stem}.jpg"
        target_image = det_root / "images" / split / image_name
        if not target_image.exists():
            try:
                target_image.symlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, target_image)
        label_name = Path(image_name).stem + ".txt"
        lines = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            cx, cy = (x + w / 2) / width, (y + h / 2) / height
            bw, bh = w / width, h / height
            if 0 < cx < 1 and 0 < cy < 1 and bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (det_root / "labels" / split / label_name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    names = {index: name for index, name in enumerate(final_classes)}
    (det_root / "data.yaml").write_text(
        json.dumps(
            {
                "path": str(det_root.resolve()),
                "train": "images/train", "val": "images/val", "test": "images/test",
                "nc": 1, "names": {0: "solder_joint"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Build classifier crops.
    cls_root = output / "classifier_dataset"
    for split in ("train", "val", "test"):
        for label in final_classes:
            (cls_root / split / label).mkdir(parents=True, exist_ok=True)
    crop_index = 0
    for record in classifier_records:
        if record["label"] not in final_classes:
            continue
        split = split_of_group[record["group"]]
        with Image.open(record["image_path"]) as img:
            image = img.convert("RGB")
        x, y, w, h = record["bbox"]
        pad_x, pad_y = w * 0.2, h * 0.2
        box = (
            max(0, int(x - pad_x)), max(0, int(y - pad_y)),
            min(image.width, int(x + w + pad_x)), min(image.height, int(y + h + pad_y)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        target = cls_root / split / record["label"] / f"{crop_index:06d}.jpg"
        image.crop(box).save(target, quality=95)
        crop_index += 1

    counts_by_split: dict[str, Counter[str]] = {}
    for split in ("train", "val", "test"):
        counts_by_split[split] = Counter(
            {
                label: len(list((cls_root / split / label).glob("*.jpg")))
                for label in final_classes
            }
        )
        print(f"{split}: {dict(counts_by_split[split])}")
    for split in ("train", "val", "test"):
        if sum(counts_by_split[split].values()) == 0:
            raise RuntimeError(f"{split} split has no classifier crops; adjust split fractions.")

    summary = {
        "generated_at": utc_id(),
        "seed": config["training"]["seed"],
        "split_unit": "board family (CS*/R*/V*/Setup*) where available, else source image",
        "groups": len(groups),
        "split_of_group": split_of_group,
        "final_classes": final_classes,
        "dropped_classes": dropped,
        "warnings": warnings,
        "used_mappings": used_mappings,
        "datasets": dataset_summaries,
        "classifier_counts": {split: dict(counts) for split, counts in counts_by_split.items()},
        "detector_images": len(by_image),
        "detector_annotations": len(detector_records),
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "saved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Dataset prepared at {output}")


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------


def _select_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available in this Colab runtime.")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return "0"  # ultralytics device id
    return "cpu"


def _train_detector(config: dict[str, Any], work: Path, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    training = config["training"]
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(work / "detector_dataset" / "data.yaml"),
        epochs=training["detector_epochs"],
        imgsz=training["detector_image_size"],
        batch=training["detector_batch_size"],
        device=device,
        seed=training["seed"],
        project=str(work),
        name="detector",
        exist_ok=True,
        verbose=False,
        plots=False,
        patience=6,
    )
    metrics = model.val(
        data=str(work / "detector_dataset" / "data.yaml"),
        device=device,
        verbose=False,
        plots=False,
    )
    summary = {
        "epochs": training["detector_epochs"],
        "imgsz": training["detector_image_size"],
        "train_summary": str(results),
        "map50": float(getattr(metrics, "map50", 0.0)),
        "map50_95": float(getattr(metrics, "map", 0.0)),
    }
    export_path: Path | None = None
    for int8 in (bool(training.get("int8", True)), False):
        try:
            candidate = model.export(
                format="tflite",
                int8=int8,
                data=str(work / "detector_dataset" / "data.yaml"),
                imgsz=training["detector_image_size"],
            )
            export_path = Path(candidate)
            summary["export_int8"] = int8
            break
        except Exception as exc:  # noqa: BLE001 - export backends vary across Colab versions
            summary.setdefault("export_errors", []).append(f"int8={int8}: {type(exc).__name__}: {exc}")
    if export_path is None or not export_path.exists():
        raise RuntimeError("YOLO TFLite export failed; see metrics.json export_errors.")
    shutil.copy2(export_path, work / "detector.tflite")
    return summary


def _train_classifier(config: dict[str, Any], work: Path) -> dict[str, Any]:
    import tensorflow as tf

    training = config["training"]
    size = training["classifier_image_size"]
    batch = training["classifier_batch_size"]
    summary = json.loads((work / "dataset_summary.json").read_text(encoding="utf-8"))
    classes = summary["final_classes"]
    root = work / "classifier_dataset"

    def make(split: str):
        return tf.keras.utils.image_dataset_from_directory(
            root / split,
            labels="inferred",
            label_mode="categorical",
            class_names=classes,
            image_size=(size, size),
            batch_size=batch,
            shuffle=(split == "train"),
            seed=training["seed"] if split == "train" else None,
        )

    train_ds, val_ds, test_ds = make("train"), make("val"), make("test")
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    counts = {
        label: len(list((root / "train" / label).glob("*.jpg"))) for label in classes
    }
    total = sum(counts.values())
    class_weights = {
        index: total / (len(classes) * max(1, counts[label]))
        for index, label in enumerate(classes)
    }

    tf.keras.utils.set_random_seed(training["seed"])
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(size, size, 3), alpha=0.75, include_top=False,
        weights="imagenet", pooling="avg",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(training["classifier_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=training["classifier_epochs"], class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    representative_images = []

    def representative_dataset():
        for images, _ in train_ds.take(8):
            for image in images:
                if len(representative_images) < 200:
                    representative_images.append(image.numpy())
                yield [tf.cast(image, tf.float32)[None, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    export_int8 = bool(training.get("int8", True))
    if export_int8:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    try:
        tflite_bytes = converter.convert()
        used_int8 = export_int8
    except Exception as exc:  # noqa: BLE001
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        used_int8 = False
        print("INT8 classifier export failed, using dynamic-range export:", exc)
    (work / "classifier.tflite").write_bytes(tflite_bytes)
    (work / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    return {
        "classes": classes,
        "counts": counts,
        "class_weights": {classes[k]: float(v) for k, v in class_weights.items()},
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "export_int8": used_int8,
    }


def cmd_train(args: argparse.Namespace) -> None:
    require_drive()
    work = Path(args.work)
    config = json.loads((work / "saved_config.json").read_text(encoding="utf-8"))
    device = _select_device(args.device)
    print(f"Training device: {device}")
    detector_summary = _train_detector(config, work, device)
    classifier_summary = _train_classifier(config, work)
    dataset_summary = json.loads((work / "dataset_summary.json").read_text(encoding="utf-8"))
    metrics = {
        "run_id": work.name,
        "generated_at": utc_id(),
        "scope": "public_data_feasibility_only",
        "device": device,
        "split_unit": dataset_summary["split_unit"],
        "split_counts": dataset_summary["classifier_counts"],
        "label_warnings": dataset_summary["warnings"],
        "classes": classifier_summary["classes"],
        "detector": detector_summary,
        "classifier": classifier_summary,
        "claim": (
            "This number is measured on public-data crops/boards and is NOT the project's "
            ">=80% held-out-board accuracy."
        ),
    }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "detector_map50": detector_summary["map50"],
        "classifier_test_accuracy": classifier_summary["test_accuracy"],
        "classes": classifier_summary["classes"],
    }, indent=2))


# ---------------------------------------------------------------------------
# 4. Export + publish + push
# ---------------------------------------------------------------------------


def _tensor_metadata(path: Path) -> dict[str, Any]:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow.lite as tflite

        Interpreter = getattr(tflite, "Interpreter")
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "input_shape": [int(v) for v in inp["shape"]],
        "input_dtype": str(np_dtype_name(inp["dtype"])),
        "output_shape": [int(v) for v in out["shape"]],
        "output_dtype": str(np_dtype_name(out["dtype"])),
    }


def np_dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype))


def input_size_from_shape(shape: list[int]) -> int:
    """Return the spatial size of a square NHWC or NCHW model input."""
    if len(shape) != 4:
        raise ValueError(f"Expected a 4-D model input shape, got {shape}")
    if shape[3] == 3:  # NHWC [1, H, W, 3]
        return int(shape[1])
    if shape[1] == 3:  # NCHW [1, 3, H, W]
        return int(shape[2])
    raise ValueError(f"Cannot find the 3-channel axis in input shape {shape}")


def load_labels(path: Path) -> list[str]:
    """Read labels.txt into a clean list (it is plain text, not JSON)."""
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels


def cmd_export(args: argparse.Namespace) -> None:
    require_drive()
    work = Path(args.work)
    run_id = utc_id()
    config = json.loads((work / "saved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
    bundle = work / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    for name in ("detector.tflite", "classifier.tflite", "labels.txt"):
        shutil.copy2(work / name, bundle / name)
    for name in ("inspector.py", "test_board.py"):
        shutil.copy2(REPO_ROOT / "runtime" / name, bundle / name)

    det_meta = _tensor_metadata(bundle / "detector.tflite")
    cls_meta = _tensor_metadata(bundle / "classifier.tflite")
    det_h = input_size_from_shape(det_meta["input_shape"])
    cls_h = input_size_from_shape(cls_meta["input_shape"])
    output_shape = det_meta["output_shape"]
    layout = "auto"
    if len(output_shape) == 3:
        layout = "c_first" if output_shape[1] == 5 else "n_first"
    preprocess = {
        "schema_version": 1,
        "pipeline": "yolo11n_joint_detector + mobilenetv3_small_classifier",
        "accepts": "full board image of any resolution (no caller-supplied crops)",
        "detector": {
            "file": "detector.tflite", "input_size": int(det_h),
            "input_range": [0, 1], "num_classes": 1, "output_layout": layout,
            "model_metadata": det_meta,
        },
        "classifier": {
            "file": "classifier.tflite", "input_size": int(cls_h),
            "input_range": [0, 255], "model_metadata": cls_meta,
        },
        "score_threshold": config["training"]["conf_threshold"],
        "iou_threshold": config["training"]["iou_threshold"],
        "crop_expand": config["training"]["crop_expand"],
        "max_detections": config["training"]["max_detections"],
    }
    (bundle / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")

    (bundle / "README.txt").write_text(
        "Full-board PCB inspector bundle\n"
        "Run: python test_board.py --bundle . --image <whole_board_image>\n"
        "Read model_card.json for the exact classes and limitations.\n",
        encoding="utf-8",
    )
    artifact_names = sorted(
        p.name for p in bundle.iterdir()
        if p.name not in {"manifest.sha256", "model_card.json"}
    )
    artifact_checksums = {name: sha256_file(bundle / name) for name in artifact_names}
    model_card = {
        "run_id": run_id,
        "created_at": utc_id(),
        "status": "ready_for_local_test",
        "scope": metrics["scope"],
        "classes": load_labels(bundle / "labels.txt"),
        "label_semantics": metrics["label_warnings"],
        "split_policy": metrics["split_unit"],
        "metrics": metrics,
        "checksums": artifact_checksums,
        "licences": {
            "soldef": "GPL-3.0 per Kaggle API metadata",
            "roboflow": "verify before adding data",
            "ultralytics": "AGPL-3.0 (YOLO11 export; check current licence for deployment)",
            "tensorflow": "Apache-2.0",
        },
        "limitations": [
            "Public-data feasibility model; not the >=80% held-out-board claim.",
            "Full-board pipeline validated on public joint close-ups and synthetic composite boards.",
            "Any class absent from labels.txt cannot be predicted; check notes in the JSON report.",
        ],
    }
    (bundle / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    manifest_names = sorted(p.name for p in bundle.iterdir() if p.name != "manifest.sha256")
    manifest_checksums = {name: sha256_file(bundle / name) for name in manifest_names}
    (bundle / "manifest.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in manifest_checksums.items()),
        encoding="utf-8",
    )
    destination = RUNS_ROOT / run_id
    if destination.exists():
        raise RuntimeError(f"Run {run_id} already exists; refusing to overwrite.")
    shutil.copytree(bundle, destination)
    current = DRIVE_ROOT / "exports" / "current"
    if current.exists():
        shutil.rmtree(current)
    current.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(destination, current)
    print(f"Bundle published to {destination}")
    print(f"Stable copy at {current}")
    print(f"Run id: {run_id}")
    return destination


def cmd_publish(args: argparse.Namespace) -> None:
    require_drive()
    run_dir = Path(args.run_dir)
    card = json.loads((run_dir / "model_card.json").read_text(encoding="utf-8"))
    handoff = HANDOFF_PATH.read_text(encoding="utf-8") if HANDOFF_PATH.exists() else "{}"
    state = json.loads(handoff)
    previous = state.get("latest_run")
    state["updated_at"] = utc_id()
    state["author"] = "colab-agent"
    state["latest_run"] = {
        "run_id": card["run_id"],
        "drive_path": f"drive/MyDrive/AI-PCB-Inspector/runs/{card['run_id']}",
        "stable_path": "drive/MyDrive/AI-PCB-Inspector/exports/current",
        "status": "ready_for_local_test",
        "scope": card["scope"],
        "classes": card["classes"],
        "label_semantics": card["label_semantics"],
        "checksums": card["checksums"],
        "requested_action": "Download exports/current, unzip, and run test_board.py on a whole board image.",
    }
    if previous:
        state.setdefault("previous_runs", []).append(previous)
    state["workflow"] = "Colab-only: no local sync and no local training steps."
    state["known_issues"] = [
        "Public-data feasibility only; not the project's held-out-board accuracy.",
        "solder_bridge is absent unless a reviewed bridge source or physical board data exists "
        "(no Roboflow project is configured).",
    ]
    state["next_action"] = "User tests the full-board bundle locally and reports confusion examples; then collect our-board images."
    HANDOFF_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    with CHANGELOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {utc_id()}: published full-board run {card['run_id']} "
            f"({card['scope']}); local test requested.\n"
        )
    print(f"Updated {HANDOFF_PATH}")


def cmd_push(args: argparse.Namespace) -> None:
    token = get_secret("GH_TOKEN") or get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Add GH_TOKEN (a GitHub PAT with repo write access) to Colab Secrets, then rerun. "
            "The token is never printed or written to a file."
        )
    if not (REPO_ROOT / ".git").is_dir():
        subprocess.run(["git", "init", "-b", "main"], cwd=REPO_ROOT, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Colab Agent", "-c", "user.email=colab-agent@local",
         "add", "-A"], cwd=REPO_ROOT, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        subprocess.run(
            ["git", "-c", "user.name=Colab Agent", "-c", "user.email=colab-agent@local",
             "commit", "-m", args.message], cwd=REPO_ROOT, check=True
        )
    else:
        print("No local changes to commit.")

    remote = GITHUB_REMOTE
    current = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    if current != remote:
        if current:
            subprocess.run(["git", "remote", "set-url", "origin", remote], cwd=REPO_ROOT, check=True)
        else:
            subprocess.run(["git", "remote", "add", "origin", remote], cwd=REPO_ROOT, check=True)

    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": f"http.{remote}.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: basic {encoded}",
        "GIT_CONFIG_KEY_1": "core.askPass",
        "GIT_CONFIG_VALUE_1": "/bin/true",
    })
    # Keep the existing remote history but let this cleaned tree replace it.
    fetched = subprocess.run(
        ["git", "fetch", "--depth=1", "origin", "main"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if fetched.returncode == 0:
        subprocess.run(
            ["git", "-c", "user.name=Colab Agent", "-c", "user.email=colab-agent@local",
             "merge", "--allow-unrelated-histories", "-s", "ours", "FETCH_HEAD",
             "-m", "Overhaul repository (Colab-only workflow)"],
            cwd=REPO_ROOT, check=True,
        )
    else:
        print("Shallow fetch failed; pushing the cleaned tree without merging remote history.")
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO_ROOT, check=True, env=env)
    print("Pushed to GitHub (token redacted).")


def cmd_all(args: argparse.Namespace) -> None:
    print("\n===== download + audit =====")
    cmd_download(args)
    cmd_audit(args)
    print("\n===== prepare =====")
    cmd_prepare(args)
    print("\n===== train =====")
    cmd_train(args)
    print("\n===== export =====")
    run_dir = cmd_export(args)
    print("\n===== publish + push =====")
    cmd_publish(argparse.Namespace(run_dir=str(run_dir)))
    cmd_push(args)


GLOBALS = globals()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_config = str(REPO_ROOT / "training" / "configs" / "fullboard_config.json")

    for name in ("download", "audit"):
        child = sub.add_parser(name)
        child.add_argument("--config", default=default_config)
        if name == "audit":
            child.add_argument("--previews", type=int, default=6)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", default=default_config)
    prepare.add_argument("--output", default="/content/pcb_fullboard_run")
    prepare.add_argument("--allow-provisional", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--work", default="/content/pcb_fullboard_run")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    export = sub.add_parser("export")
    export.add_argument("--work", default="/content/pcb_fullboard_run")
    publish = sub.add_parser("publish")
    publish.add_argument("--run-dir", required=True)
    push = sub.add_parser("push")
    push.add_argument("--message", default="Colab: full-board inspector overhaul")
    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--config", default=default_config)
    all_cmd.add_argument("--output", default="/content/pcb_fullboard_run")
    all_cmd.add_argument("--allow-provisional", action="store_true")
    all_cmd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_cmd.add_argument("--message", default="Colab: full-board inspector overhaul")
    all_cmd.add_argument("--previews", type=int, default=6)
    all_cmd.add_argument("--run-dir", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "all":
        cmd_all(args)
        return
    GLOBALS[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
