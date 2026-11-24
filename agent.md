# agent.md — AI PCB Defect Inspector (Capstone 2026–2027)

Read this file first. It gives an AI coding/research agent (and any human teammate) the full context for our capstone: what the challenge is, what the rules are, what we are building, and how we want to work. When something here conflicts with the official challenge PDF or a teacher's instruction, the PDF/teacher wins — flag the conflict instead of guessing.

Last updated: 2026-11-24. Items marked **TBC** are unknown and must be confirmed with the team or school.

## Current handoff checkpoint — 2026-11-24

- **Repository state:** scaffold plus a Colab-only public dataset download/audit and Option B
  classifier pilot runner, now handling multiple split-specific COCO files and covered by local
  COCO discovery/path tests. The Colab download/audit and the public-data Option B pilot **ran to
  completion on Colab** (run `training/runs/option_b_public_pilot/run_20261124_224758`):
  SolDef_AI `_annotations.coco.json` (428 images, 671 annotations) with the audited class map, the
  15 audit preview images, crops and split manifest are now in the repo, and training produced
  `option_b_mobilenetv3_small.pt` + `metrics.json` (public-data pilot only — see below). There is
  still no local camera capture, OpenCV ROI locator, deployment inference, hardware, cloud or UI
  code, and no physical-board dataset has been captured.
- **Confirmed direction:** Option B first (`good`, `solder_bridge`, `insufficient_solder`), with an
  upgrade to Option C if measured ROI coverage requires a learned locator.
- **Dataset:** no public data can support the final accuracy claim. Kaggle SolDef_AI is downloadable
  in principle with the user's account; Kaggle API metadata reports GPL-3.0. Roboflow is a second
  candidate; verify the licence/class definitions in the logged-in account. Do not equate
  `exc_solder` to bridge without evidence. The unlicensed GitHub mirror remains excluded.
- **Evaluation integrity:** collect and test on our own physical boards. Group train/validation/test
  by physical board and, where possible, board design/source. Keep synthetic defect edits out of
  validation/test. Report per-joint and board-level metrics separately.
- **Dataset documentation corrections:** the PCBA-DET repository describes screw, wiring and scratch
  categories, not our solder-joint taxonomy. `docs/datasets.md` and `docs/approaches.md` record the
  current qualifications and candidate links.
- **User-confirmed 2026-11-24:** no lab access or disposable boards; user owns a soldering iron/solder
  and sensors/microcontrollers. Selected Option B with `good`, `solder_bridge`,
  `insufficient_solder`; will train in Google Colab Free and has Kaggle and Roboflow access.
- **Board update:** user has one perfboard and a very limited budget. Use it only as a limited
  proof-of-process sample. Track joint IDs, capture controlled before/after images, and do not treat
  multiple joints/views from that one board as independent board-level evaluation. Seek spare/scrap
  boards through teammates, school, or repair shops; do not damage sensors/microcontrollers.
- **Current milestone:** the Colab download/audit and the reproducible public-data Option B
  baseline were completed on 2026-11-24 (artifacts in the repo; pilot test accuracy 0.7903 —
  see `training/runs/option_b_public_pilot/run_20261124_224758/metrics.json`). The full
  physical-board capture, Option B/C decision data, board-grouped test and final ≥80% accuracy
  claim remain future milestones; this run is feasibility evidence only.
- **Dataset/legal status:** Kaggle API metadata for official SolDef_AI reports GPL-3.0 and archive
  size ~1.18 GB. Roboflow is a second candidate; verify the licence/class definitions in the
  account before mapping. Never assume SolDef `exc_solder` means bridge. No credentials were
  supplied or stored. Colab assets are in `training/notebooks/option_b_colab.ipynb` and its
  companion `.py` runner.
- **Current execution boundary:** no SSH/remote shell access to the user's Colab. The user must run
  the notebook in hosted Colab and may share non-sensitive logs, class summaries, screenshots, and
  metrics for follow-up. Never request or accept API keys/tokens in chat.
- **Current test coverage:** local tests live in `tests/test_option_b_colab.py`; runner syntax,
  notebook JSON/Python parsing and local tests pass. Image-cropping tests are skipped locally because
  Pillow is unavailable; we stopped a slow local package download at the user's request. Colab
  already provides PyTorch/torchvision; do not install them on the PC. Runner supports `--device
  auto|cpu|cuda` and auto-selects available Colab hardware.
- **Evaluation integrity:** public dataset splits do not establish physical-board independence.
  Report public-data results as feasibility only. Collect and test on our own physical boards for
  the final accuracy claim; keep synthetic edits out of validation/test.
- **Immediate blockers:** one perfboard only is not enough for board-independent generalization.
  Gather more expendable/scrap boards if possible; meanwhile the single board can support a clearly
  labeled procedure pilot. Verify Roboflow licence and class definitions inside its logged-in
  project.

---

## 1. TL;DR

- **What:** a portable, hardware-based **AI PCB inspector** that photographs printed circuit boards moving through a small "production line" (conveyor/feeder), classifies defects with an on-device AI model, reports results over Wi-Fi to the cloud, and shows pass/fail on a dashboard.
- **Why:** it fits **Pathway A – Industrial Base** ("fully automated production lines", "product quality inspection and control") of the *Information Technology and AI Applications* capstone challenge, Grade 12 Semester 1, 2026/2027.
- **Hardware on hand:** Raspberry Pi 4 (4 GB RAM) and a Raspberry Pi Camera Module **rev 1.3** (OV5647, 5 MP, fixed focus). **No Arduino boards** (banned by the rules).
- **Must-haves from the rules:** sensing, wireless data to the cloud, AI model with **≥ 80 % accuracy** shown with evidence, our own dataset and rubric, documented sensor calibration, ≥ 4 measurable design requirements (including response time and accuracy), testable/portable/reliable on exhibition day, printed logbooks.

---

## 2. What a capstone is in STEM schools (background)

Egypt's STEM schools are public high schools built around project-based learning. They were set up from 2011 through the USAID-funded Egypt STEM Schools Project together with the Ministry of Education, and the model is organized around **Egypt's Grand Challenges** (for example water, energy, pollution, waste, food and the industrial/agricultural base). The challenge document for this year comes from the **Central Unit for Gifted and Talented Students** (Ministry of Education and Technical Education).

The **capstone** is the school's signature project course:

- A **team-based, semester-long engineering project** in which students pick a real problem connected to one or more Grand Challenges and design, build, test and defend a solution. Public descriptions of the model mention six capstones across three years (one per semester), though details may differ by school and year.
- It follows the **Engineering Design Process (EDP)**: present and justify a problem → background research → specify measurable design requirements → brainstorm and select a solution → design and build a prototype → test/experiment → evaluate → iterate and improve.
- Typical deliverables (names vary by school): **journals/logbooks** (individual, kept throughout), a **portfolio** (the written engineering report), a **working prototype**, a **poster**, and an **exhibition** where teams present and defend the work, with external judges (often university faculty) asking questions.
- It carries significant weight in the final grade, but the exact percentage differs by grade and by source (some student guides say 60 % in Grades 10–11 and 20 % in Grade 12; others say 60 % overall). **TBC with our teacher** — do not quote a number in official documents without checking.
- What judges look for, in practice: a clearly justified problem, **testable design requirements**, a prototype that actually works and is demonstrably tested, honest data and analysis, evidence of iteration, and that **every member understands every part** of the project.

Practical implication for an agent: this is graded as an engineering-process project, not just a coding task. The documentation trail (requirements, test data, calibration records, iteration notes) matters as much as the code. Always help produce evidence, and never fabricate results.

---

## 3. The official challenge (summary of the PDF)

**Challenge title:** Information Technology and Artificial Intelligence (AI) Applications
**Level:** Grade 12, Semester 1, 2026/2027
**Theme:** Communication, Sensing, Information, Informatics

**Egypt Grand Challenges named in the document**
- Increase the industrial and agricultural base of Egypt.
- Public health issues and diseases.
- Improve the scientific and technological environment.

**Big idea:** integrate AI-driven edge computing with IoT for next-generation industrial automation, educational robotics, or assistive medical technologies. Students design a system that analyzes the current situation (sensing, measurements, communication with data sources) and formulates AI-driven solutions. The prototype **must include sensing, wireless data transfer to the cloud, advanced communication tools, and AI platforms**.

**Essential question:** How can integrating AI tools with information technology help develop reliable and intelligent devices that improve the industrial base, provide medical services, or enhance the learning environment?

**Pathways (examples; students are not restricted to these)**
- **A. Industrial Base** — fully automated production lines; product quality inspection and control; any other application with school-board approval. ← *our pathway*
- **B. Educational Environment** — language-learning device, automatic homework grading, programming-assistant device, etc.
- **C. Medical Services** — monitoring device for people with disabilities, AI smart glasses for the blind, AI wheelchair with obstacle avoidance, etc.

### Design requirements (must satisfy)
1. Identify **at least four measurable design requirements**, **including response time and system accuracy**, plus **two additional measurable requirements** chosen for the application.
2. Prototype must use **ICT, IoT and AI platforms**.
3. Prototype must contain **hardware components**.
4. Provide **datasets and a rubric** for the machine/deep-learning algorithms **during the exhibition**.
5. Provide **evidence of AI model accuracy, at least 80 %**.
6. **Conduct and document sensor calibration.**
7. Functional prototype must be **testable, operational, reliable and portable** on exhibition day. Bring a **hard copy of each team member's logbook**.

### Constraints
- The whole system (software and hardware) must be testable.
- **Arduino boards are not allowed**; more advanced controllers are encouraged.
- Students must be able to **show how system outputs/states change as inputs change**.
- **Lab equipment or materials must never leave the laboratories**; tests needing them happen in the labs.
- Projects involving chronic diseases are not considered; the medical pathway allows no hazards. (Not relevant to us, but keep it in mind if the scope ever drifts.)

---

## 4. Our project: AI PCB Defect Inspector

### 4.1 Concept
A compact automated inspection station. A PCB enters the inspection zone on a small conveyor or sliding feeder. A presence sensor detects it, lighting is held constant, the camera captures an image, the Pi runs the AI model, and the result (pass / fail + defect types + locations + confidence) is shown locally, logged, and pushed to a cloud dashboard. Optionally an actuator (servo diverter/pusher) routes failed boards to a reject bin, so the output visibly changes with the input during the demo.

> "Pipeline" in the original request is interpreted as a **production line** (board flow through stages). **TBC:** confirm with the team that a small conveyor/feeder plus reject mechanism is what we want, versus a static inspection rig.

### 4.2 Why this satisfies the rules
| Rule | How we meet it |
|---|---|
| Sensing | Camera (image sensing), IR/ToF presence sensor, ambient light sensor (BH1750 or similar); optional temperature |
| Wireless data to the cloud | MQTT over Wi-Fi to a cloud broker; results/images logged to a cloud database and dashboard |
| Advanced communication tools | MQTT, HTTPS/REST, web dashboard, optional Telegram/e-mail alert on repeated failures |
| AI platform | On-device detection/classification model (TFLite or NCNN), trained with a standard framework |
| Hardware | Pi 4, camera, sensors, lighting, conveyor/feeder, actuator, enclosure |
| No Arduino | Pi 4 only (optionally a Pi Pico/ESP32 is *not* planned; avoid anything Arduino-branded) |
| ≥ 80 % accuracy | Evaluated on a held-out test set built from *separate physical boards* |
| Dataset + rubric | Our own images, labels, split files, and a scoring rubric, all printed/available at the exhibition |
| Calibration | Camera scale (px/mm), lighting (lux), presence-sensor threshold/distance — each documented with before/after error |
| Outputs vs inputs | Different boards/defects produce different labels, dashboard states, LEDs and reject-gate actions live |

### 4.3 Hardware

**Confirmed**
- Raspberry Pi 4 Model B, 4 GB RAM. Use a proper 5 V / 3 A supply; add a heatsink or small fan because inference keeps the CPU busy. For portability, use a power bank rated for 5 V / 3 A.
- Raspberry Pi Camera Module **v1.3** (OV5647). Pi 4 has the standard 15-pin camera connector, so the original ribbon cable fits directly.

**Camera v1.3 notes (verify in testing)**
- 5 MP sensor; stills up to 2592×1944; common video modes include 1920×1080 and lower. Do not assume 1920×1440 exists as a native mode.
- Field of view is roughly 54° × 41°. At a mount height of ~80 mm the view is about 80 × 60 mm, so a larger board needs a taller mount or several shots stitched/tiled.
- **Fixed focus**, factory-set for distance. For close-up inspection the lens usually has to be re-focused by twisting it; it may be held with glue. Test early whether sharp focus is achievable at the planned distance, and have a plan B (a higher mount with cropping, or a macro lens).
- Fixed exposure/white balance during data capture: **lock these** so training and deployment images look the same.

**Planned (TBC — finalize BOM and pin map)**
- Lighting: diffused LED ring or panel inside a shaded enclosure (constant, shadow-free light is the single biggest accuracy lever).
- Presence sensor: IR break-beam or ToF (e.g., VL53L0X/VL53L1X) to trigger capture.
- Ambient light sensor (e.g., BH1750) to log and calibrate illumination.
- Conveyor/feeder: small belt or slider driven by a DC/stepper motor with a motor driver (driven from Pi GPIO).
- Reject diverter/pusher: micro servo.
- Status LEDs/buzzer, optional small display.
- Enclosure/rig: aluminium extrusion or 3D-printed/laser-cut frame, light-tight shroud.
- Test boards: simple through-hole and SMD practice boards, assembled with deliberate defects (see §6).

**Safety/process:** soldering and any lab equipment stay in the lab (rule: nothing leaves the labs). Plan defect-board fabrication during lab sessions.

### 4.4 Software architecture

```
[IR/ToF trigger] -> [Capture (Picamera2, locked exposure)] -> [Preprocess/crop/ROI]
        -> [AI model on Pi 4 (TFLite INT8 or NCNN)] -> [Post-process: NMS, thresholds, pass/fail]
        -> [Local UI + LEDs + reject servo] -> [Log: JSON + saved image]
        -> [MQTT publish -> cloud broker -> database -> dashboard]
```

- OS: current 64-bit Raspberry Pi OS. Use a Python virtual environment.
- Camera: **Picamera2** (libcamera). Do not rely on `cv2.VideoCapture(0)` for the CSI camera.
- Inference: `tflite-runtime` or NCNN/ONNX Runtime; OpenCV for image handling; NumPy.
- Training happens **off-device** (Google Colab or a PC), never on the Pi.
- Messaging: `paho-mqtt` to a cloud broker (e.g., a free-tier hosted broker). Queue messages locally and **sync when Wi-Fi returns**, so the demo survives flaky exhibition Wi-Fi (bring a phone hotspot as backup).
- Cloud storage/dashboard: a free-tier database and dashboard service (TBC: Supabase/Firebase/Grafana Cloud/ThingSpeak, etc.). Stored fields: timestamp, board ID, pass/fail, defect list with class/confidence/box, latency, sensor readings, thumbnail.
- Local UI: small Flask/FastAPI page on the Pi's hotspot/LAN showing the live image, annotated result and counters, so the system is fully demonstrable offline.

### 4.5 AI approach

Two candidate pipelines. Decide after a pilot with real images; record the decision and the evidence in the logbook.

**Option A — single-stage detector.** YOLOv8n or YOLOv5n trained in Colab, exported to INT8 TFLite or NCNN, run on the Pi 4 at 320–416 px input. Gives boxes + classes in one pass.

**Option B — two-stage (classical CV + small classifier).** OpenCV localizes the board and crops regions of interest (joints/components); a small CNN (e.g., MobileNetV2 / EfficientNet-Lite) classifies each crop. Often needs less data, is easier to explain, and makes the ≥ 80 % classification accuracy easier to measure.

**Candidate classes (start small, then grow):** Good, Insufficient solder, Solder bridge, Cold joint, Tombstoning, Misalignment/missing component. For the first milestone, **limit to 3–4 classes** and reach target accuracy before adding more.

**Speed expectation:** the Pi 4 has no AI accelerator, so inference runs on the CPU. A small INT8 model is plausibly in the few-hundred-millisecond range per image, but **benchmark on our actual Pi and record the numbers** — do not quote figures from the internet as ours.

### 4.6 Measurable design requirements (draft — team must confirm targets)

At least four are required, including response time and accuracy. Draft set:

| # | Requirement | Draft target | How measured |
|---|---|---|---|
| R1 | Response time: trigger → result displayed | ≤ 3 s (**TBC after benchmark**) | Timestamps logged in code, 50+ runs, report mean/95th percentile |
| R2 | Model accuracy on held-out boards | ≥ 85 % overall (hard floor 80 %), plus per-class precision/recall | Test set from boards never used in training; confusion matrix |
| R3 | Defect recall for critical class (solder bridge) | ≥ 90 % | Same test set |
| R4 | Cloud sync latency | ≤ 5 s, ≥ 99 % delivery | Timestamps at Pi and dashboard, 100 messages |
| R5 | Presence-sensor trigger reliability | ≥ 98 % over 100 passes | Repeated conveyor runs |
| R6 | Camera scale calibration error | ≤ 0.1 mm/px scale error or ≤ 2 % length error | Ruler/calibration target before vs after |
| R7 | Portable runtime on battery | ≥ 2 h (**TBC**) | Timed run |

Pick the final list of at least six (the two mandatory ones plus application-specific ones), and keep every number backed by a logged test.

---

## 5. Source of the idea and what to trust

The starting point is the Hackster project **"Multi-Class PCB Solder Defect Inspector"** (by Rahul Khanna): https://www.hackster.io/rahulkhanna/multi-class-pcb-solder-defect-inspector-e3fdc5

It uses a Raspberry Pi Zero, the Raspberry Pi **AI Camera** (Sony IMX500), Sony's Brain Builder for training, a 6-class detector, an LED ring light and a small aluminium rig holding the camera ~80 mm above the board. We are **not** copying it. We are using it as a blueprint and changing hardware (Pi 4 + regular camera v1.3), which moves inference from the camera sensor to the Pi CPU.

**Known problems with that page — do not copy blindly:**
- Its `inference.py` runs a TFLite model on the CPU via OpenCV, even though the text says inference runs on the AI Camera.
- The output-decoding function does not match a standard YOLOv5 TFLite export (grid/anchor decoding resembles older YOLO versions, and its own comments disagree with the code).
- Setup commands reference packages from older OS releases and plain `pip install` that current OS versions block.
- Results are inconsistent (expected mAP 0.85–0.92 vs. a claimed 98.4 % accuracy; a sample log has a class not in the class list) and no dataset or trained model is published.
- Brain Builder targets the IMX500, so it is not suitable for our camera.

Treat its rig layout, class list and logging format as inspiration only. **Any numbers in our documents must come from our own tests.**

---

## 6. Dataset and evaluation plan

**Dataset must be ours** (the rules require showing datasets at the exhibition).

1. Build or buy **10–20+ simple practice boards** and assemble them in the lab. Create deliberate defects safely (e.g., excess solder to bridge pads, too little solder, lifted/cold joints, rotated/misplaced components, missing parts). Photograph every board before and after.
2. Capture images under **fixed lighting, fixed distance, fixed camera settings**, with natural variation (small position/rotation shifts, slight lighting changes) so the model generalizes.
3. Target roughly **150–200 labeled examples per class**, then augment (small rotation, flip where valid, brightness jitter). Keep a record of how many images come from each physical board.
4. **Split by board, not by image.** Frames from the same board must never be in both train and test, or accuracy will be inflated. Keep a fixed **test set that is touched only for final evaluation**.
5. Label with a consistent guide (written definitions and example photos per class). Have a second person re-label a sample to check agreement.
6. Metrics: overall accuracy, per-class precision/recall/F1, confusion matrix; for detectors also mAP@0.5. Report **results on real held-out boards**, even if lower than validation results.
7. **Rubric for the exhibition:** a one-page scoring sheet describing classes, labeling rules, test procedure, pass criteria (≥ 80 %), and results — plus the dataset folder, split files, training notebook and the confusion matrix.

Optional background datasets (e.g., PKU-Market-PCB or DeepPCB) cover **bare-board** defects, not solder joints. If used at all, check the licence and be clear in the logbook about what they were used for.

---

## 7. Calibration and testing (graded evidence)

Document each of these with a table, plot or photo, the method, and the before/after error:

- **Camera scale (px/mm):** photograph a ruler or checkerboard at the working distance; compute scale and error; repeat after remounting.
- **Camera focus/sharpness:** record the focus procedure and a sharpness metric on a test target.
- **Lighting:** measure with a lux meter or reference app; log the ambient light sensor against it; fit a correction.
- **Presence sensor:** distance/threshold calibration against a ruler; reliability test over repeated passes.
- **Timing:** instrument code to log latencies per stage (capture, inference, post-process, upload).
- **System tests:** show outputs changing with inputs (good board → PASS, bridged board → FAIL + reject, covered sensor → warning, Wi-Fi off → queued then synced).
- **Reliability:** run for a fixed period (e.g., 1–2 hours) without a crash; record temperature/throttling.

---

## 8. Deliverables checklist

- [ ] Working portable prototype (hardware + software), testable on demand
- [ ] ≥ 4 measurable design requirements with test evidence (response time and accuracy included)
- [ ] Model accuracy ≥ 80 % shown on held-out data
- [ ] Datasets, split files, labeling guide and **rubric** ready to show
- [ ] Sensor calibration records
- [ ] Cloud dashboard live (with offline fallback)
- [ ] Printed **logbook for each team member**
- [ ] Portfolio/report and poster (format per school instructions — **TBC**)
- [ ] Demo script and a backup plan (hotspot, spare SD card with image, spare cables, spare boards)

---

## 9. Suggested phases (adjust to the real exhibition date — **TBC**)

1. **Pilot (1–2 weeks):** mount camera and light, check focus, capture ~100 images of a few boards, benchmark a small model on the Pi 4, decide Option A vs B.
2. **Data (2–3 weeks):** build defect boards in the lab, capture the full dataset, label, split by board.
3. **Model (2 weeks):** train, evaluate, tune, export, benchmark latency on the Pi.
4. **Integration (2 weeks):** sensors, trigger, conveyor/reject, local UI, MQTT + cloud dashboard, offline queue.
5. **Calibration and testing (1–2 weeks):** all of §7.
6. **Polish (1 week):** enclosure, documentation, poster, rehearsals including failure drills (Wi-Fi off, camera unplugged).

---

## 10. Repository layout (proposed)

```
pcb-inspector/
  agent.md                  # this file
  README.md
  docs/                     # requirements, calibration tables, test reports, logbook scans
  hardware/                 # BOM, wiring diagram, pin map, CAD/enclosure files
  data/
    raw/                    # original images (never edited)
    labels/                 # annotation files
    splits/                 # train/val/test lists (split by board)
    DATASET.md              # how data was collected, class definitions, counts
  training/                 # notebooks/scripts, configs, exported models
  src/
    capture.py              # Picamera2 capture with locked settings
    sensors.py              # presence + light sensors
    inference.py            # model loading, preprocessing, decoding, NMS
    pipeline.py             # trigger -> capture -> infer -> decide -> log
    actuators.py            # conveyor + reject servo
    cloud.py                # MQTT publish, offline queue, retry
    ui/                     # local Flask/FastAPI dashboard
  tests/                    # unit tests (decoding, thresholds, queue) and benchmark scripts
  results/                  # metrics, confusion matrices, latency logs
```

---

## 11. Working rules for the agent

**Do**
- Ask before big changes to scope, hardware or architecture, and when a requirement is unclear. Prefer one short question over assumptions.
- Keep everything **reproducible**: pinned dependencies (`requirements.txt`), scripts instead of one-off manual steps, fixed random seeds, saved configs.
- Run and **measure on the real Pi 4** (latency, memory, temperature); never guess performance.
- Log decisions, test results and iterations in `docs/` in a form we can paste into the logbook and portfolio. Include dates, what changed, why, and the evidence.
- Write clear, commented code that every team member can understand and explain to judges.
- Handle failure gracefully (camera missing, Wi-Fi down, sensor timeout) with clear status messages.
- Keep secrets (API keys, broker passwords) in environment variables or an untracked config file, never in the repo.
- Note licences and credit every library, dataset, tutorial or repository we reuse.

**Do not**
- Use Arduino boards or Arduino-branded controllers.
- Invent or "estimate" accuracy/latency numbers or datasets; do not report validation numbers as test numbers; never leak test boards into training.
- Copy large amounts of code from other projects without attribution — judges will ask us to explain it.
- Remove or plan to remove lab equipment from the labs.
- Add features that make the system hard to demonstrate or test on exhibition day.
- Over-engineer: a simple, reliable, well-tested system beats a complicated, flaky one.

**Coding conventions**
- Python 3 with a virtual environment; format with `black`/`ruff`; type hints where helpful.
- Small modules, one responsibility each; configuration in a single file (thresholds, pins, MQTT topics).
- Every module that touches hardware should have a mock/simulation mode so code can be tested on a laptop.

---

## 12. Risks and open questions

| Risk / question | Mitigation / next step |
|---|---|
| Camera v1.3 fixed focus may not focus at the planned distance | Test on day one; adjust lens or raise the mount; plan B macro lens |
| Not enough real defect samples for some classes | Start with 3–4 classes; build more boards; per-class counts in DATASET.md |
| Accuracy drops on new boards (overfitting to the boards we trained on) | Split by board; reserve unseen boards for the final test |
| Lighting changes between lab and exhibition hall | Enclosed, self-lit rig; lock camera settings; recalibrate on site |
| Wi-Fi unreliable at the exhibition | Local UI + offline queue; phone hotspot; test failure drill |
| Pi 4 overheating or throttling during demo | Heatsink/fan; log temperature; duty-cycle inference |
| Conveyor mechanics eat the schedule | Build the vision/AI part first; use a simple slider/feeder as fallback |
| Scope not clearly covered by the challenge examples | It matches "product quality inspection and control" in Pathway A, but confirm with the teacher/school board |

**TBC list:** exhibition date; team size and roles; budget; which cloud service; final class list; conveyor vs static rig; final poster/portfolio formats; capstone grade weights.

---

## 13. References
- Official challenge document: *Capstone Challenge 2026–2027, Grade 12 Semester 1, Information Technology and AI Applications* (Central Unit for Gifted and Talented Students).
- Inspiration (use with caution, see §5): Hackster, *Multi-Class PCB Solder Defect Inspector*.
- Related academic example for edge inspection on Raspberry Pi hardware: Okano et al., *Edge AI for Industrial Visual Inspection: YOLOv8-Based Visual Conformity Detection Using Raspberry Pi*, Algorithms 18(8), 510 (2025).
- Background on Egypt's STEM schools and capstone model: Egypt STEM Schools Project (USAID/Ministry of Education) materials, and school capstone guides published by individual STEM schools.