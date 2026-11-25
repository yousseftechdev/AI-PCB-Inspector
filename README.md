# AI PCB Defect Inspector

A portable AOI station for the Grade 12 (2026/27) capstone: a Raspberry Pi 4 photographs a board
under locked lighting, an on-device model returns **PASS / FAIL / REVIEW** with joint boxes, and the
result is logged and pushed to a dashboard.

**Workflow change (2026-11-25):** everything is developed and trained in **Google Colab**. The
local machine/Pi only downloads a finished bundle and runs one test script. There is no local
sync step and no local training.

---

## Current state

| Area | State |
|---|---|
| Architecture | ✅ full-board hybrid (detector + classifier) adopted — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Label semantics | ✅ rulebook written — [`docs/LABELS.md`](docs/LABELS.md); `no_good` is no longer a bridge |
| Training pipeline | ✅ Colab notebook + CLI written — [`training/PCB_Inspector_Colab.ipynb`](training/PCB_Inspector_Colab.ipynb) |
| Local test script | ✅ [`runtime/test_board.py`](runtime/test_board.py) |
| Trained bundle | ⬜ pending a Colab run with Kaggle/Roboflow credentials in Colab Secrets |
| Real board accuracy | ⬜ not measured; public-data numbers are feasibility only |
| Hardware rig / cloud UI | ⬜ not built |

The old 0.7823 crop-classifier result was measured on a **defective label map** and is retired. It
is kept in Drive `option_b_public_pilot/` for the record, not as a project result.

---

## How it works

```
whole board image (any size)
  -> YOLO11n detector: solder_joint boxes
  -> per-joint crop from the original image
  -> MobileNetV3-Small classifier: good | insufficient_solder | solder_bridge? | other
  -> PASS / FAIL / REVIEW + annotated image + JSON
```

The detector has **one class** — it never calls something a bridge. The classifier cannot predict
a class that has no reviewed training examples and no entry in `labels.txt`. A tilted component
(`no_good`) maps to `other`, so it is reported for review, not as a bridge.

---

## Run it

1. Add `KAGGLE_USERNAME`, `KAGGLE_KEY`, optional `ROBOFLOW_API_KEY`, and `GH_TOKEN` to **Colab
   Secrets** (never in a notebook cell or chat).
2. Open `training/PCB_Inspector_Colab.ipynb` and Run all. It downloads data, shows label audit
   contact sheets, trains, exports, publishes to Drive, verifies checksums and pushes to GitHub.
3. After reviewing the contact sheets, set `reviewed: true` only for mappings you can defend in
   `training/configs/fullboard_config.json`, and rerun the train/export cells.
4. Download Drive `AI-PCB-Inspector/exports/current/` and run:

   ```bash
   python test_board.py --bundle . --image /path/to/whole_board.jpg
   ```

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) and [`deploy/README.md`](deploy/README.md).

---

## Repository layout

```
agent.md              project brief (what/why/rules)
colab-agent.md        operating instructions for the Colab agent
docs/
  ARCHITECTURE.md     full-board pipeline, contract, verdict rules
  LABELS.md           class definitions and source-mapping rulebook
  WORKFLOW.md         Colab-only run/download workflow
  DECISIONS/          dated decision records
  approaches.md       historic options survey (superseded by ARCHITECTURE.md)
  datasets.md         dataset survey (historic, with corrections)
training/
  PCB_Inspector_Colab.ipynb   single run-all notebook
  fullboard_colab.py          audit / prepare / train / export / publish / push CLI
  configs/fullboard_config.json
runtime/
  inspector.py        TFLite full-board runtime (shared by Colab and local test)
  test_board.py       the script you download and run locally
deploy/README.md      local download + checksum instructions (Windows & Linux)
tests/                contract and regression tests (CPU only)
handoff/              HANDOFF.json and CHANGELOG.md in Drive root (mirrors here)
```

---

## Rules that do not change

- No Arduino-branded controllers. No on-Pi training. No fabricated numbers.
- ≥4 measurable design requirements including response time and accuracy; documented sensor
  calibration; public data never in our held-out test set.
- Secrets stay in Colab Secrets / environment variables, never in git or Drive text files.
- Public-data metrics are labelled `public_data_feasibility_only`.
