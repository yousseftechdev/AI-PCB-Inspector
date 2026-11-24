import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


SCRIPT = Path(__file__).parents[1] / "training" / "notebooks" / "option_b_colab.py"
SPEC = importlib.util.spec_from_file_location("option_b_colab", SCRIPT)
OPTION_B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTION_B)


def write_coco(split_dir, category, color):
    split_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color).save(split_dir / "board.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "board.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 20, 20]}],
        "categories": [{"id": 1, "name": category}],
    }
    annotation_path = split_dir / "_annotations.coco.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    return annotation_path


class OptionBColabTests(unittest.TestCase):
    def test_find_coco_files_ignores_non_coco_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "metadata.json").write_text('{"title":"not coco"}', encoding="utf-8")
            annotations = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "good"}],
            }
            (root / "labels.json").write_text(json.dumps(annotations), encoding="utf-8")

            found = OPTION_B.find_coco_files(root)

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0][1]["categories"][0]["name"], "good")

    def test_resolve_image_prefers_split_local_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train"
            valid = root / "valid"
            train.mkdir()
            valid.mkdir()
            (train / "board.jpg").write_bytes(b"train")
            (valid / "board.jpg").write_bytes(b"valid")

            resolved = OPTION_B.resolve_image(root, "board.jpg", valid)

            self.assertEqual(resolved.read_bytes(), b"valid")

    @unittest.skipIf(Image is None, "Pillow is required for crop-pipeline tests")
    def test_load_records_reads_split_local_images_and_groups_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_annotations = write_coco(root / "train", "good", (0, 255, 0))
            val_annotations = write_coco(root / "valid", "bridge", (255, 0, 0))
            config = {
                "datasets": [
                    {
                        "name": "sample",
                        "root": str(root),
                        "annotations": [str(train_annotations), str(val_annotations)],
                        "class_map": {"good": "good", "bridge": "solder_bridge"},
                    }
                ]
            }

            records = OPTION_B.load_records(
                config,
                root / "output",
                ["good", "solder_bridge", "insufficient_solder"],
            )

            self.assertEqual(len(records), 2)
            self.assertEqual({record["label"] for record in records}, {"good", "solder_bridge"})
            self.assertEqual(records[0]["group"], records[1]["group"])
            self.assertNotEqual(records[0]["path"], records[1]["path"])
            self.assertTrue(all(Path(record["path"]).is_file() for record in records))
            self.assertEqual(
                {Path(record["annotation_file"]).parent.name for record in records},
                {"train", "valid"},
            )

    @unittest.skipIf(Image is None, "Pillow is required for crop-pipeline tests")
    def test_unmapped_categories_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation = write_coco(root / "train", "unexpected", (10, 20, 30))
            config = {
                "datasets": [
                    {
                        "name": "sample",
                        "root": str(root),
                        "annotations": [str(annotation)],
                        "class_map": {"good": "good"},
                    }
                ]
            }

            with self.assertRaisesRegex(ValueError, "does not match categories"):
                OPTION_B.load_records(config, root / "output", ["good"])


if __name__ == "__main__":
    unittest.main()
