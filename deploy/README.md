# Local test bundle — download and run

The bundle is in Drive: `AI-PCB-Inspector/exports/current/`. Zip that folder in Drive if you want
one file, then download it to your PC or Pi and unzip.

## Contents

```
detector.tflite       one-class joint detector
classifier.tflite     joint classifier
labels.txt            classifier class order
preprocess.json       sizes, ranges, thresholds, output layout
model_card.json       classes, scope, warnings, metrics, checksums
inspector.py          runtime (imported by test_board.py)
test_board.py         the CLI you run
README.txt
manifest.sha256
```

## Install

On the Pi (or any Linux board):

```bash
python -m pip install numpy pillow tflite-runtime
```

On a Windows/Linux PC without `tflite-runtime`:

```bash
python -m pip install numpy pillow tensorflow
```

`tensorflow.lite.Interpreter` is used automatically if `tflite_runtime` is not installed.

## Verify the download

Linux/macOS:

```bash
cd bundle
sha256sum -c manifest.sha256
```

Windows PowerShell:

```powershell
cd .\bundle
Get-Content manifest.sha256 | ForEach-Object {
  $hash, $name = $_ -split '  ', 2
  $actual = (Get-FileHash -Algorithm SHA256 $name).Hash.ToLower()
  if ($actual -ne $hash) { throw "CHECKSUM MISMATCH: $name" }
}
"All checksums OK"
```

Do not run the model if any line fails. Delete that file and download it again.

## Run on a whole board

```bash
python test_board.py --bundle . --image board_001.jpg
```

Outputs:

- `board_001_annotated.jpg` — boxes, class names and confidences on the original image;
- `board_001_report.json` — verdict, per-joint boxes, counts, timings, and any notes.

Verdicts: `PASS` (all joints good), `FAIL` (an `insufficient_solder` or `solder_bridge` joint),
`REVIEW` (unknown / low confidence / no joint found). A `REVIEW` is not a pass.

If `solder_bridge` is not listed in `model_card.json -> classes`, the bundle has no reviewed bridge
training data; every report will say so, and the model cannot falsely call a bridge.

## Sanity check without an image

```bash
python test_board.py --self-test
```

This generates a synthetic board and runs the full detector → classifier → verdict path. It proves
the install works; it is **not** an accuracy measurement.
