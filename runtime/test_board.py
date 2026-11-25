#!/usr/bin/env python3
"""Local full-board test script.

Download a run bundle from Google Drive, unzip it, then:

    python test_board.py --bundle ./bundle --image board_001.jpg

The script accepts a *whole board image* of any size. It writes an annotated
image and a JSON report next to the input image and prints the board verdict.

Dependencies: numpy, Pillow, and either tflite-runtime (Pi) or tensorflow (PC).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# The runtime module travels inside the bundle. When running straight from the
# repository, make its directory importable as a fallback.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspector import FullBoardInspector  # noqa: E402


def _demo_image(path: Path) -> None:
    """Create a synthetic board image so --self-test works before a model exists."""
    image = Image.new("RGB", (640, 480), (22, 74, 52))
    draw = ImageDraw.Draw(image)
    for box in [(40, 40, 600, 440)]:
        draw.rectangle(box, outline=(180, 180, 160), width=4)
    for x in range(90, 580, 90):
        for y in range(90, 420, 80):
            draw.rectangle((x - 14, y - 9, x + 14, y + 9), fill=(205, 205, 195))
            draw.ellipse((x - 6, y - 5, x + 6, y + 5), fill=(225, 190, 120))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a full-board PCB inspector bundle.")
    parser.add_argument("--bundle", type=Path, help="Folder containing the downloaded model bundle")
    parser.add_argument("--image", type=Path, help="Full board image (any resolution)")
    parser.add_argument("--output", type=Path, help="Annotated image output (default: <image>_annotated.jpg)")
    parser.add_argument("--json", dest="json_path", type=Path, help="JSON report output")
    parser.add_argument("--self-test", action="store_true", help="Run on a generated synthetic board")
    args = parser.parse_args()

    if args.self_test:
        probe = Path("self_test_board.jpg")
        _demo_image(probe)
        image_path = probe
    elif args.image is not None:
        image_path = args.image
    else:
        parser.error("provide --image or --self-test")

    if args.bundle is None:
        # Common case: bundle is the folder containing this script.
        args.bundle = Path(__file__).resolve().parent

    started = time.perf_counter()
    inspector = FullBoardInspector(args.bundle)
    with Image.open(image_path) as image:
        result = inspector.inspect(image, annotate=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    output = args.output or image_path.with_name(image_path.stem + "_annotated.jpg")
    if result.annotated_image is not None:
        Image.fromarray(result.annotated_image).save(output, quality=92)
    report = result.to_dict()
    report["total_wall_ms"] = round(elapsed_ms, 2)
    report["bundle"] = str(args.bundle)
    report["image"] = str(image_path)
    report["annotated_image"] = str(output)

    json_path = args.json_path or image_path.with_name(image_path.stem + "_report.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"VERDICT: {result.verdict}  ({len(result.joints)} joints found)")
    print(f"Counts: {json.dumps(result.counts)}")
    for note in report["notes"]:
        print(f"NOTE: {note}")
    print(f"Annotated image: {output}")
    print(f"JSON report:     {json_path}")
    print(f"Latency: {elapsed_ms:.1f} ms total, "
          f"{result.timings_ms.get('detect', 0):.1f} ms detect, "
          f"{result.timings_ms.get('classify', 0):.1f} ms classify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
