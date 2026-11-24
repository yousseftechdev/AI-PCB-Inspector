import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path


SOLDEF_SLUG = "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection"
ROBOFLOW_WORKSPACE = "work-6qkmv"
ROBOFLOW_PROJECT = "pcb-solder-joint"


def find_coco_files(root):
    results = []
    for path in Path(root).rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if all(isinstance(data.get(key), list) for key in ("images", "annotations", "categories")):
            results.append((path, data))
    return results


def download(args):
    root = Path(args.root)
    soldef = root / "soldef"
    roboflow_dir = root / "roboflow"
    soldef.mkdir(parents=True, exist_ok=True)
    roboflow_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets before downloading.")
    if not find_coco_files(soldef):
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                SOLDEF_SLUG,
                "-p",
                str(soldef),
                "--unzip",
            ],
            check=True,
        )
    else:
        print(f"SolDef_AI directory already has files; skipping download: {soldef}")

    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise RuntimeError("Set ROBOFLOW_API_KEY in Colab Secrets before downloading.")
    if not find_coco_files(roboflow_dir):
        from roboflow import Roboflow

        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        project.version(args.roboflow_version).download("coco", location=str(roboflow_dir))
    else:
        print(f"Roboflow directory already has files; skipping download: {roboflow_dir}")
    print(f"Dataset downloads are in {root}")


def resolve_image(root, file_name, preferred_dir=None):
    root = Path(root)
    candidate = root / file_name
    if candidate.is_file():
        return candidate
    if preferred_dir is not None:
        candidate = Path(preferred_dir) / file_name
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely resolve {file_name!r} under {root}; found {len(matches)} basename matches."
    )


def inspect(args):
    from PIL import Image, ImageDraw

    root = Path(args.root)
    summary = {}
    preview_dir = root / "audit_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for source_root in (root / "soldef", root / "roboflow"):
        entries = find_coco_files(source_root)
        print(f"\n{source_root.name}: {len(entries)} COCO annotation file(s)")
        summary[source_root.name] = []
        for annotation_path, coco in entries:
            categories = {item["id"]: item["name"] for item in coco["categories"]}
            counts = Counter(categories.get(item["category_id"], str(item["category_id"])) for item in coco["annotations"])
            item = {
                "annotation": str(annotation_path),
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "categories": categories,
                "counts": dict(counts),
            }
            summary[source_root.name].append(item)
            print(json.dumps(item, indent=2))

            image_by_id = {image["id"]: image for image in coco["images"]}
            shown = Counter()
            for annotation in coco["annotations"]:
                category = categories.get(annotation["category_id"], str(annotation["category_id"]))
                if shown[category] >= args.previews_per_class or not annotation.get("bbox"):
                    continue
                image_info = image_by_id.get(annotation["image_id"])
                if not image_info:
                    continue
                image_path = resolve_image(source_root, image_info["file_name"], annotation_path.parent)
                image = Image.open(image_path).convert("RGB")
                x, y, width, height = map(float, annotation["bbox"])
                draw = ImageDraw.Draw(image)
                draw.rectangle((x, y, x + width, y + height), outline="red", width=max(2, image.width // 500))
                draw.text((x, max(0, y - 18)), category, fill="red")
                output = preview_dir / f"{source_root.name}_{len(summary[source_root.name])}_{category}_{shown[category]}.jpg"
                image.thumbnail((1200, 900))
                image.save(output, quality=90)
                shown[category] += 1
            print("  Previews:", preview_dir)
        if not entries:
            print("  No COCO annotations found; inspect this export and adapt the parser.")
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAudit summary: {root / 'audit_summary.json'}")


def load_records(config, work, target_classes):
    from PIL import Image

    records = []
    crop_dir = work / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for dataset in config["datasets"]:
        root = Path(dataset["root"])
        invalid = set(dataset["class_map"].values()) - set(target_classes)
        if invalid:
            raise ValueError(f"{dataset['name']}: unknown target labels {sorted(invalid)}")
        annotation_paths = dataset.get("annotations", [dataset.get("annotation")])
        annotation_paths = [Path(path) for path in annotation_paths if path]
        if not annotation_paths:
            raise ValueError(f"{dataset['name']}: no COCO annotation files configured")

        dataset_classes = set()
        for annotation_path in annotation_paths:
            coco = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {item["id"]: item["name"] for item in coco["categories"]}
            dataset_classes.update(
                name for name, target in dataset["class_map"].items() if name in categories.values()
            )
            selected = {
                category_id: dataset["class_map"][name]
                for category_id, name in categories.items()
                if name in dataset["class_map"]
            }
            images = {image["id"]: image for image in coco["images"]}
            image_cache = OrderedDict()
            for annotation in coco["annotations"]:
                target = selected.get(annotation["category_id"])
                if target is None or annotation.get("iscrowd", 0):
                    continue
                image_info = images.get(annotation["image_id"])
                bbox = annotation.get("bbox")
                if not image_info or not bbox or len(bbox) != 4:
                    continue
                image_path = resolve_image(root, image_info["file_name"], annotation_path.parent)
                image_key = image_path.name
                if image_key not in image_cache:
                    with Image.open(image_path) as image:
                        image_cache[image_key] = image.convert("RGB")
                    if len(image_cache) > 8:
                        oldest_id = next(iter(image_cache))
                        image_cache.pop(oldest_id).close()
                else:
                    image_cache.move_to_end(image_key)
                image = image_cache[image_key]
                x, y, width, height = map(float, bbox)
                if width <= 1 or height <= 1:
                    continue
                pad_x, pad_y = width * 0.2, height * 0.2
                box = (
                    max(0, int(x - pad_x)),
                    max(0, int(y - pad_y)),
                    min(image.width, int(x + width + pad_x)),
                    min(image.height, int(y + height + pad_y)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                crop_name = f"{annotation_path.parent.name}_{annotation_path.stem}_{image_path.stem}_{annotation['id']}.jpg"
                crop_path = crop_dir / target / dataset["name"] / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop(box).save(crop_path, quality=95)
                records.append(
                    {
                        "path": str(crop_path),
                        "label": target,
                        "group": f"{dataset['name']}:{image_path.name}",
                        "source": dataset["name"],
                        "source_image": image_info["file_name"],
                        "annotation_file": str(annotation_path),
                    }
                )
            for cached_image in image_cache.values():
                cached_image.close()

        if not dataset_classes:
            raise ValueError(
                f"{dataset['name']}: class_map does not match categories in any annotation file"
            )
    return records


def grouped_splits(records, target_classes, seed):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    labels = np.asarray([target_classes.index(record["label"]) for record in records])
    groups = np.asarray([record["group"] for record in records])
    indices = np.arange(len(records))
    for label in target_classes:
        group_count = len({record["group"] for record in records if record["label"] == label})
        if group_count < 5:
            raise ValueError(f"{label}: only {group_count} distinct source-image groups; at least 5 are needed for this pilot split.")
    for attempt in range(100):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        train_val, test = next(outer.split(indices, labels, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000 + attempt)
        train_rel, val_rel = next(inner.split(train_val, labels[train_val], groups[train_val]))
        train, val = train_val[train_rel], train_val[val_rel]
        all_classes = set(range(len(target_classes)))
        if all(set(labels[split]) == all_classes for split in (train, val, test)):
            return (train, val, test), labels, groups
    raise ValueError("Could not make image-grouped train/validation/test splits containing all classes.")


class CropDataset:
    def __init__(self, records, indices, classes, transform):
        self.records = records
        self.indices = indices
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[int(self.indices[index])]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.classes.index(record["label"])


def train(args):
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from torchvision.models import MobileNet_V3_Small_Weights

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target_classes = config["target_classes"]
    if len(target_classes) != 3 or len(set(target_classes)) != 3:
        raise ValueError("This pilot expects exactly three distinct target classes.")
    if not config.get("datasets"):
        raise ValueError("Add at least one dataset with audited category mappings to training_config.json.")
    for dataset in config["datasets"]:
        if not dataset.get("license_confirmed", False):
            raise ValueError(f"{dataset.get('name')}: confirm the dataset licence in its source account before training.")
        if not dataset.get("class_map"):
            raise ValueError(f"{dataset.get('name')}: class_map is empty; verify labels and configure explicit mappings.")

    work = Path(args.output)
    if work.exists():
        work = work.with_name(f"{work.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    work.mkdir(parents=True)
    records = load_records(config, work, target_classes)
    counts = Counter(record["label"] for record in records)
    group_counts = {
        label: len({record["group"] for record in records if record["label"] == label})
        for label in target_classes
    }
    print("Crops by class:", dict(counts))
    print("Source-image groups by class:", group_counts)
    if any(counts[label] == 0 for label in target_classes):
        raise ValueError("At least one target class has no annotated crops after mapping.")

    (train_indices, val_indices, test_indices), labels, groups = grouped_splits(records, target_classes, args.seed)
    with (work / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "source", "source_image", "annotation_file", "label", "group", "crop_path"])
        for split_name, indices in (("train", train_indices), ("val", val_indices), ("test", test_indices)):
            for index in indices:
                record = records[int(index)]
                writer.writerow([split_name, record["source"], record["source_image"], record["annotation_file"],
                                 record["label"], record["group"], record["path"]])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the Colab runtime has no available GPU.")
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), normalize]
    )
    train_loader = DataLoader(CropDataset(records, train_indices, target_classes, train_transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(CropDataset(records, val_indices, target_classes, eval_transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(CropDataset(records, test_indices, target_classes, eval_transform),
                             batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(target_classes))
    model.to(device)
    train_counts = np.bincount(labels[train_indices], minlength=len(target_classes))
    if np.any(train_counts == 0):
        raise ValueError("Training split is missing a class.")
    class_weights = torch.tensor(len(train_indices) / (len(target_classes) * train_counts),
                                 dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    def evaluate(loader):
        model.eval()
        truth, predictions = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels_batch in loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(images)
                loss_sum += criterion(logits, labels_batch).item() * len(labels_batch)
                truth.extend(labels_batch.cpu().tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
        accuracy = float(np.mean(np.asarray(truth) == np.asarray(predictions)))
        return loss_sum / max(1, len(truth)), accuracy, truth, predictions

    history = []
    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels_batch)
            correct += (logits.argmax(1) == labels_batch).sum().item()
            seen += len(labels_batch)
        val_loss, val_accuracy, _, _ = evaluate(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "train_accuracy": correct / max(1, seen),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(row)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    test_loss, test_accuracy, y_true, y_pred = evaluate(test_loader)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(target_classes))),
        target_names=target_classes, output_dict=True, zero_division=0,
    )
    metrics = {
        "pilot_only": True,
        "split_unit": "source image; physical board IDs unavailable",
        "device": str(device),
        "target_classes": target_classes,
        "datasets": [
            {
                "name": dataset["name"],
                "annotations": dataset.get("annotations", [dataset.get("annotation")]),
                "source_version": dataset.get("source_version"),
                "license": dataset.get("license", "not recorded"),
                "license_confirmed": dataset.get("license_confirmed", False),
                "source_url": dataset.get("source_url"),
                "class_map": dataset["class_map"],
            }
            for dataset in config["datasets"]
        ],
        "class_maps": {dataset["name"]: dataset["class_map"] for dataset in config["datasets"]},
        "crop_counts": dict(counts),
        "source_image_groups_by_class": group_counts,
        "split_sizes": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
        "seed": args.seed,
        "epochs": args.epochs,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(args.config, work / "training_config.json")
    torch.save(
        {"model_state_dict": best_state, "class_names": target_classes, "image_size": args.image_size},
        work / "option_b_mobilenetv3_small.pt",
    )
    print("Public-data pilot test accuracy (not project accuracy):", round(test_accuracy, 4))
    print(json.dumps(metrics["confusion_matrix"], indent=2))

    destination = Path(args.drive_output) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("option_b_mobilenetv3_small.pt", "metrics.json", "split_manifest.csv", "training_config.json"):
        shutil.copy2(work / name, destination / name)
    print(f"Model and reports saved to Google Drive: {destination}")


def main():
    parser = argparse.ArgumentParser(description="Option B annotated-ROI classifier pilot for Google Colab.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--root", default="/content/pcb_inspector")
    download_parser.add_argument("--roboflow-version", type=int, default=1)
    download_parser.set_defaults(func=download)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--root", default="/content/pcb_inspector")
    inspect_parser.add_argument("--previews-per-class", type=int, default=3)
    inspect_parser.set_defaults(func=inspect)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output", default="/content/option_b_run")
    train_parser.add_argument("--drive-output", required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
