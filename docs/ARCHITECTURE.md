# Full-board inspector architecture — 2026-11-25

Status: **adopted**. Supersedes the crop-only Option B pilot described in `approaches.md`.
Owner: Colab agent. The local machine/Pi only runs the exported bundle.

---

## 1. What the user gives the system

A **whole board image** of arbitrary resolution, captured under locked lighting. No crops, no joint
coordinates, no caller-supplied regions of interest.

```
full board image
   -> letterbox to detector size (640 px)
   -> YOLO11n joint detector (1 class: solder_joint)
   -> NMS -> joint boxes mapped back to original image pixels
   -> per-box crop from the ORIGINAL image (20% context padding)
   -> MobileNetV3-Small classifier -> {good, insufficient_solder, solder_bridge?, other}
   -> board verdict: PASS / FAIL / REVIEW
   -> annotated image + JSON report (boxes, labels, confidences, timings)
```

This is Option C ("hybrid") from `approaches.md`, with one deliberate simplification: the detector
has a **single class**. It answers only "where is a solder joint?", never "is this a bridge?".
That separation is what stops a tilted component or two unconnected pads from being called a
bridge: the detector has no bridge concept, and the classifier cannot predict a class that was
not verified and exported.

---

## 2. Why not the old crop classifier

The 2026-11-25 pilot took already-cropped joints and classified them. Three failures followed:

1. It could not inspect a board. A human had to find every joint first.
2. Its `no_good -> solder_bridge` mapping labelled component misplacement as a bridge. A tilted
   part was therefore *ground-truth* "bridge", not a model mistake.
3. Two pads that are simply not connected (`poor_solder` / `no_good`) also landed in bridge or
   good classes depending on the source label, so the learned "bridge" feature was incoherent.

`docs/LABELS.md` records the corrected mapping policy. `docs/DECISIONS/2026-11-25-full-board-overhaul.md`
records the decision and its evidence.

---

## 3. Components and files

| Component | Trained in | Exported as | Runtime file |
|---|---|---|---|
| Joint locator | Colab, Ultralytics YOLO11n | `detector.tflite` (INT8 where supported) | `runtime/inspector.py` |
| Joint classifier | Colab, tf.keras MobileNetV3-Small | `classifier.tflite` (INT8 where supported) | `runtime/inspector.py` |
| Local test CLI | — | copied into bundle | `runtime/test_board.py` |
| Contract | — | `preprocess.json`, `labels.txt` | read by `inspector.py` |

The entire runtime is numpy + Pillow + a TFLite interpreter (`tflite-runtime` on the Pi,
`tensorflow.lite` on a PC). No PyTorch, no Ultralytics, no OpenCV is needed locally.

### 3.1 Contract

`preprocess.json` declares input sizes, input ranges, detector output layout, NMS thresholds,
crop expansion and the scoring threshold. `labels.txt` is the index → class order used by the
classifier. `model_card.json` carries checksums, split policy, label-review warnings, metrics and
licences. `manifest.sha256` is `sha256sum -c` compatible.

A run bundle is immutable. A stable copy is kept at Drive `exports/current/`.

---

## 4. Verdict semantics

| Joint label | Meaning | Board effect |
|---|---|---|
| `good` | joint accepted in an otherwise ordinary image | no effect |
| `insufficient_solder` | too little solder / poor wetting (once visually reviewed) | FAIL |
| `solder_bridge` | confirmed pad-to-pad solder short (only if a verified source exists) | FAIL |
| `other` | a real annotation that is **not** in our taxonomy (misplacement, spike, excess solder) | REVIEW |
| `low_confidence` | classifier score below threshold | REVIEW |

Rules:

- Any `insufficient_solder` or `solder_bridge` → **FAIL**.
- Else any `other` / `low_confidence` / no joints found → **REVIEW** (human looks).
- Else → **PASS**.

`REVIEW` is not a silent pass. It is how we keep "I don't know" from becoming "bridge".

---

## 5. Splitting and evaluation

Crops from one physical board must never cross splits. The dataset preparer derives a board family
from SolDef path tokens (`CS*`, `R*`, `V*`, `Setup*`); if a source has no such tokens it falls back
to the source image. This is stricter than the old source-image-only split but still **not** a
claim of physical-board independence when public data are used.

Two metrics are reported separately:

- detector: mAP@0.5 on the held-out family group;
- classifier: per-class accuracy on the held-out family group.

Both are labelled `public_data_feasibility_only`. The ≥80% project requirement still needs our own
physically held-out boards (`agent.md` §6).

---

## 6. Known limits

- Public solder-joint datasets are close-ups, not full boards. The full-board path is exercised
  end-to-end by synthetic composite boards in `tests/`, but that is a plumbing test, not accuracy.
- The detector is trained on every annotated joint (including `other` categories). That is
  intentional: it should find joints regardless of defect taxonomy.
- The classifier sees only reviewed categories. If `solder_bridge` has no verified source it is
  dropped from `labels.txt`; the runtime prints a note in every JSON report saying it cannot
  predict that class. This is a feature, not a bug.
- Pi latency has not been measured. Do not quote a number until the bundle is run on the Pi.
