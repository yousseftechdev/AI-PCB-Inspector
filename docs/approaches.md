# Ways to build the PCB inspector

Status: **Option B selected for pilot (pilot executed 2026-11-24); Option C is the fallback** · 2026-11-24
Owner: TBC · Mechanical presentation remains undecided

This document lists the realistic ways to build the AI PCB Defect Inspector on the hardware we
already own (Pi 4 4 GB, Camera Module v1.3 / OV5647, no AI accelerator), scores them against the
capstone's actual grading criteria, and proposes one pilot to settle the decision with evidence
instead of opinion.

Nothing here is a number we measured. Every performance figure marked *benchmark* must be filled in
from our own runs on our own Pi before it goes into the portfolio (agent.md §11).

---

## 0. The decision is really two decisions

People usually conflate these. They are independent, and mixing them up is how capstone teams end up
with a conveyor they can't debug and a model that can't see the defect.

| # | Decision | Choices | Driven by |
|---|---|---|---|
| **D1** | **What the AI looks at and outputs** (the perception stack) | whole-board detector / two-stage crop classifier / hybrid / anomaly / classical-only | defect size in pixels, data volume, accuracy target |
| **D2** | **How a board arrives in front of the camera** (the mechanical rig) | static jig / sliding feeder / motorised conveyor + diverter | schedule, budget, demo value, R5/R7 risk |

You can build any D1 with any D2. D2 does not change model accuracy; D1 does not change the mechanics.
Pick D1 first, because it sets the dataset work, and the dataset is the long pole.

---

## 1. D1 — Perception options

All five assume: Pi 4 CPU inference via TFLite (`tflite-runtime`, INT8) or NCNN; training off-device in
Colab; fixed camera exposure/white balance so training images and live images match.

### Option A — Single-stage detector (one model, boxes + classes)

`YOLOv8n` / `YOLO11n` trained on our boards, exported to INT8 TFLite or NCNN, run at 320–416 px.

```
image -> YOLO -> [(class, box, conf), ...] -> NMS -> pass/fail rule -> result
```

- **Pros:** one model, one export, one benchmark. Gives defect *locations*, which is the visible
  "wow" on the dashboard and satisfies "outputs change as inputs change" very literally.
  It is also the approach the Hackster blueprint uses, so there is prior art to explain.
- **Cons:** a 0603 solder joint is a handful of pixels at 416 px. Either you tile the board (4–6
  crops per board, slower and more code) or you accept that only gross defects are detectable.
  Annotation cost is boxes for every defect, not one label per image. YOLO on a 640 px image of a
  busy board will produce small-object recall problems, and INT8 quantisation hurts small objects
  more than big ones.
- **Data need:** ~150–300 labelled defect instances *per class*, spread over many boards.
- **Accuracy risk:** medium-high. mAP@0.5 for a nano model on tiny joints is the number most likely
  to embarrass us; the ≥ 80 % requirement is then usually reported as *image-level* accuracy
  (any correct detection → image correct), which is defensible but must be stated precisely.

### Option B — Two-stage: CV localisation + small CNN classifier

OpenCV finds the board and the joints/component footprints; a small CNN (MobileNetV3-Small,
EfficientNet-Lite0) classifies each crop.

```
image -> board rectification -> joint/component ROIs (CV) -> crop -> CNN per crop -> aggregate
```

- **Pros:** a clearly defined *per-joint classification accuracy* is easy to measure and explain
  ("we classified 460 joints, 437 correct, 95.0 %"). The crops are
  small (e.g. 96×96), so a tiny CNN runs fast on the Pi 4 and INT8 quantisation barely hurts. Needs
  *far less* training data because the network never has to learn where the joints are.
  Transfer learning from ImageNet works well on solder-joint textures.
- **Cons:** the CV stage is the weak link. Finding joints by colour/shape is brittle across board
  colours (green/black/blue solder mask), silkscreen, and shadows, and a missed ROI is a silent
  miss, not a low-confidence box. Needs a real calibration/registration step
  (checkerboard or board-outline homography) — which is *good* for us, because §7 requires
  documented calibration anyway.
- **Data need:** enough labelled crops from multiple physical boards; repeated crops from one board
  are not independent samples. Split by board (and by board design/source where possible), and do not
  claim generalisation from image-level random splits.
- **Accuracy risk:** low on classification, medium on coverage. Mitigation: report "joints found"
  as its own metric (R-something) so a locator miss is visible rather than hidden.

### Option C — Hybrid: YOLO finds joints, CNN classifies them

`YOLO(k=1 class: "joint") -> crop each box -> CNN(class) -> aggregate`

- **Pros:** best of both. Locator is learned (so no brittle CV), classifier stays small and accurate,
  and each half can be evaluated separately. This is the architecture most likely to actually clear
  ≥ 80 % with confidence.
- **Cons:** two models to train, export, benchmark and explain; two failure modes; labeler must do
  both boxes and per-crop classes. Roughly 1.5× the work of A or B for maybe +5–10 points of accuracy.
- **Good as the *upgrade path*** from B if the CV locator turns out to be the bottleneck. Design the
  code so the locator is swappable (see §5, interface contract).

### Option D — Anomaly detection (learn only "good")

Train on defect-free boards only (e.g. a MobileNet feature extractor + Mahalanobis / PaDiM-style
embedding distance, or `anomalib` exported to ONNX).

```
image/crop -> embedding -> distance from "good" distribution -> PASS / ANOMALY + heatmap
```

- **Pros:** we can build this dataset in an afternoon (no defect fabrication needed to *train*), and
  it detects defect types we never anticipated. The heatmap is an excellent demo visual. Academically
  it is the strongest story ("we treat inspection as unsupervised anomaly detection because defect
  examples are rare in real production") and judges like a method chosen for a real reason.
- **Cons:** **it cannot name defect types**, only "different from good". Rules want classes and a
  rubric; a binary pangood/anomaly rubric is thin. Library-heavy for the Pi 4, and threshold
  calibration becomes the whole problem (per board family, per lighting).
- **Data need:** ~50–100 good images per board type — the cheapest to obtain.
- **Best use:** as a **second opinion / novelty warning** alongside A/B/C, or as the fallback if we
  cannot fabricate enough defect boards in time. Not the primary plan.

### Option E — Classical difference imaging (no AI)

Register the test image to a "golden board" reference and diff pixels/blobs.

- **Pros:** zero training, deterministic, extremely explainable, and brilliant as a *sanity baseline*.
- **Cons:** it is **not an AI model**, so it does not satisfy the AI-platform requirement on its own.
  Requires the same board model every time.
- **Verdict:** build it as the **baseline row** in our results table ("classical diff: 71 %; our CNN:
  93 %") — that comparison is exactly the kind of evidence judges reward, and it is ~100 lines.

### D1 comparison

Scores are 1–5, higher is better. "Risk" = chance we miss the ≥ 80 % milestone.

| | Accuracy headroom | Data cost | Build effort | Pi 4 speed | Explainability | Demo value | Risk |
|---|---|---|---|---|---|---|---|
| **A** whole-board detector | 3 | 4 (high) | 2 | 3 | 4 | 5 | **High** |
| **B** CV + classifier | 4 | 2 | 3 | 5 | 5 | 3 | **Low** |
| **C** hybrid | 5 | 3 | 4 | 4 | 4 | 5 | Medium |
| **D** anomaly | 3 | 1 | 3 | 3 | 3 | 4 | Medium |
| **E** classical only | 2 | 1 | 1 | 5 | 5 | 2 | disqualifying |

**Recommendation:** run the pilot as **B, with the locator behind an interface so C is a drop-in
upgrade.** B gets us a defensible accuracy number with the least data, earliest — which de-risks the
one requirement that cannot be faked at the exhibition. Report both per-joint metrics and board-level
PASS/FAIL results on physically held-out boards; per-joint accuracy alone must not be presented as
the system's overall accuracy. The pilot is for feasibility, not the final accuracy claim. If it shows the CV locator missing
joints, we spend the extra week on C, and we will have the data for it already (same crops, plus
boxes we can derive from the ROIs the CV stage got right).

Do **not** start with A. It looks like the obvious choice because prior art did it, but prior art
also published contradictory results and no dataset (agent.md §5).

---

## 2. D2 — Mechanical presentation options

| | Static jig (hand-placed) | Sliding feeder (manual push) | Motorised conveyor + diverter |
|---|---|---|---|
| Cost | ~0 | ~EGP 300 | EGP 1500+ |
| Build time | 1 day | 3–4 days | 2–3 weeks |
| Trigger | IR break-beam or button | IR break-beam | IR + encoder |
| "Outputs change with inputs" | LEDs + dashboard only | LEDs + dashboard | **+ physical reject bin** |
| Failure modes at exhibition | few | few | jams, belt slip, motor stall, power brownout |
| R5 (trigger reliability ≥ 98 %) | trivial | moderate | risky |
| R7 (2 h on battery) | easy | easy | motors eat the power budget |
| Judge impression | "a scanner" | "a station" | "a production line" ✅ |

**Recommendation:** build the **static light-tight jig first** (it is required either way — the shroud,
lighting and camera mount are shared), get the AI and cloud passing, then add the **sliding feeder**,
and treat the **motorised conveyor as a stretch goal with a demo-day fallback**. The rules ask for
"fully automated production lines" as a *pathway example*, not that our prototype be a factory. A
reliable station that visibly sorts PASS/REJECT beats a conveyor that jams while three judges watch.
A servo-driven **reject flap** on the output chute gets most of the demo value for a fraction of the
mechanical risk.

The rules do require the output to change visibly with the input. Satisfy that with: different label
on screen, red/green LED/buzzer, reject servo, and a live dashboard counter — four independent
signals, at least three of which cannot jam.

---

## 3. Optional amplifiers (cheap, high grading value)

- **Calibration artefact:** print a checkerboard/ruler target, photograph it through the same rig, and
  report px/mm and error before/after remount. Directly satisfies the calibration requirement, ~1 hour.
- **Latency instrumentation:** per-stage timestamps in the pipeline (capture/preprocess/infer/upload)
  written to JSONL. R1 is then a `pandas.describe()` away.
- **Confusion matrix + per-class table** in `results/`, regenerated by script so it is never stale.
- **Failure drill:** an "unplug the camera" and "kill the Wi-Fi" button in the demo script. Judges ask.

---

## 4. What we are explicitly not doing

- No Arduino-branded hardware (rule).
- No on-Pi training.
- No cloud inference (the model must run on-device; the cloud gets *results*).
- No 6-class detector race at the start — first milestone is **3 classes + good**, then grow.
- No second camera / no lens swap until the focus test proves v1.3 cannot focus at working distance.

---

## 5. Interface contract (so the choice stays reversible)

Whatever we pick, keep these seams. They are what let us switch D1 later without rewriting the demo.

```python
# src/inference.py — the only thing the pipeline knows about
class Inspector:
    def load(self, model_path: str) -> None: ...
    def inspect(self, image: np.ndarray) -> InspectionResult: ...

@dataclass
class Defect:
    label: str          # "good" | "insufficient_solder" | ...
    confidence: float
    box: tuple[int, int, int, int] | None   # None if the method has no localisation

@dataclass
class InspectionResult:
    verdict: str        # "PASS" | "FAIL"
    defects: list[Defect]
    image_annotated: np.ndarray
    timings_ms: dict[str, float]
```

A/B/C/D all implement `Inspector`. `pipeline.py`, `cloud.py` and the UI never import a model library
directly. Mock mode (`Inspector` that returns a scripted result) makes the whole system testable on a
laptop, as required by agent.md §11.

---

## 6. Current pilot plan — Option B in Colab

The user has no lab access or disposable boards, but has Kaggle and Roboflow accounts and will use
Google Colab Free. The initial week therefore tests the **annotated-crop classifier**, not the
complete Option B pipeline or its OpenCV locator. It cannot establish performance on our camera or
unseen physical boards.

1. Run `training/notebooks/option_b_colab.ipynb` in hosted Colab with credentials stored in Colab
   Secrets. Use Colab's provided PyTorch; the runner auto-selects an available GPU or CPU. Keep
   source data and checkpoints in Colab/Drive, not on the project PC. **Done 2026-11-24** —
   artifacts are committed in the repo under `data/external/` and
   `training/runs/option_b_public_pilot/run_20261124_224758/` (pilot CPU run, `test_accuracy`
   0.7903 on public data only).
2. Audit the exported COCO categories, annotations, and representative crops. Verify Roboflow
   licence and class definitions. Map only labels that genuinely correspond to `good`,
   `solder_bridge`, and `insufficient_solder`. If no public source has a defensible bridge label,
   do not fake one; train a narrower pilot and source real examples before expanding.
3. Train MobileNetV3-Small on annotated joint crops. The notebook groups crops by source image to
   avoid image-level leakage, but the public data do not establish physical-board separation.
4. Record class counts, split manifest, per-class precision/recall/F1, confusion matrix, and the
   public-data-only caveat. This is feasibility evidence, not the ≥80% project result.
5. Obtain expendable boards before doing the project-specific capture, CV locator, camera focus,
   Pi latency, and final board-grouped test. If the CV locator misses too many ROIs, move from B to C.

---

## 7. Open questions for the team

1. **Confirmed pilot classes:** `good`, `solder_bridge`, `insufficient_solder`; the label mapping still requires dataset-definition review.
2. **TBC** — conveyor or station? (recommendation above: station + reject flap first)
3. **Known limitation:** user has no lab access or disposable boards; user owns a soldering iron/solder and sensors/microcontrollers. Do not risk their existing hardware as defect samples.
4. **TBC** — can inexpensive expendable boards or scrap boards be obtained, and with what budget?
5. **TBC** — exhibition date, which sets the model-freeze date.
