# AI PCB Defect Inspector

A portable, hardware-based automated optical inspection (AOI) station for printed circuit boards.
A board enters the inspection zone, a Raspberry Pi 4 camera captures it under locked lighting, an
on-device AI model decides **PASS / FAIL** and names the defect, the result is shown locally, and it
is pushed over Wi-Fi to a cloud dashboard. A reject actuator sorts failed boards.

Capstone project — Grade 12, Semester 1, 2026/2027 · *Information Technology and AI Applications* ·
Pathway A (Industrial Base: product quality inspection and control).

---

## Status

| Area | State |
|---|---|
| Repository structure | ✅ scaffolded |
| Architecture decision (which AI pipeline) | 🟡 proposed in [`docs/approaches.md`](docs/approaches.md) — **awaiting team decision** |
| Dataset strategy | 🟡 researched in [`docs/datasets.md`](docs/datasets.md) — **public-data pilot executed 2026-11-24** |
| Hardware rig | ⬜ not built |
| Our dataset | 🟡 public-data pilot captured in `data/external/`; our physical captures still pending |
| Trained model | 🟢 public-data pilot model in [`training/runs/option_b_public_pilot/run_20261124_224758/`](training/runs/option_b_public_pilot/run_20261124_224758/option_b_mobilenetv3_small.pt) (CPU pilot, public data only — not the project accuracy claim) |
| Pipeline / cloud / UI code | ⬜ not written |

---

## The system at a glance

```
board arrives → presence sensor triggers → camera capture (locked exposure/white balance)
   → AI inference on the Pi 4 (INT8 TFLite / NCNN) → PASS/FAIL + defects + boxes + confidence
   → local UI + LEDs + reject servo → JSON log + saved image
   → MQTT → cloud broker → database → dashboard (with an offline queue when Wi-Fi drops)
```

Hardware: Raspberry Pi 4 Model B (4 GB), Camera Module v1.3 (OV5647, 5 MP, fixed focus), diffuse
LED ring in a light-tight shroud, IR/ToF presence sensor, BH1750 ambient-light sensor, small feeder,
micro-servo reject diverter. **No Arduino** (competition rule).

---

## Repository layout

```
data/
  raw/          original captures — never edited (git-ignored)
  labels/       annotations
  splits/       train/val/test lists — split by physical board, not by image
  external/     downloaded public datasets (git-ignored)
docs/           requirements, calibration records, test reports, decision records
  approaches.md architecture options, comparison, and the one-week pilot plan
  datasets.md   survey of public PCB defect datasets and the gap analysis
hardware/       BOM, wiring diagram, pin map, enclosure/CAD
training/       notebooks, configs, exported models
src/            capture, sensors, inference, pipeline, actuators, cloud, local UI
tests/          unit tests and benchmark scripts (hardware mocked)
results/        metrics, confusion matrices, latency logs
agent.md        the project brief (git-ignored by choice — keep it local)
```

---

## Documents to read, in order

2. [`docs/approaches.md`](docs/approaches.md) — four ways to build the AI inspector, scored against
   the grading criteria, with a recommendation (two-stage CV + CNN, upgradable to a hybrid) and a
   one-week pilot that decides it with measurements instead of opinions.
3. [`docs/datasets.md`](docs/datasets.md) — every public PCB/solder dataset we found, what it is
   actually good for, and why none can be our dataset.

---

## Nothing here is measured yet (except the public-data pilot)

No accuracy or latency figure in this repository is a measurement of *our* hardware or boards.
Every number marked *benchmark* or TBC must be produced on our own hardware and logged in
`docs/` before it appears in the portfolio, poster or logbook. The only committed result is the
public-data Option B pilot (`training/runs/option_b_public_pilot/run_20261124_224758/`); its
`test_accuracy` is a feasibility number on public images, never the ≥80% project claim.

---

## Conventions

- Python 3 in a virtual environment; `black` / `ruff`; type hints where useful.
- One responsibility per module; all thresholds, pins and MQTT topics in a single config file.
- Every hardware-touching module has a mock mode so the system runs on a laptop.
- Secrets in `.env`, never in the repo.
- Fixed random seeds and saved configs, so any result can be reproduced.

## Licence

TBC — decision for the team. Third-party datasets and libraries are credited individually in
[`docs/datasets.md`](docs/datasets.md) and must be credited in the portfolio.
