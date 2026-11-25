# colab-agent.md — Colab-only operating brief

Read this after `agent.md` at the start of every Colab session. Mirror kept at Drive
`/content/drive/MyDrive/AI-PCB-Inspector/colab-agent.md` so it survives a VM reset.

Last updated: **2026-11-25**. Supersedes the two-agent sync protocol in the previous version.

---

## 1. Ownership

**Colab owns everything:** dataset download/audit, label decisions, splits, training, evaluation,
model export, handoff state, docs, and git push. The user's machine/Pi only downloads a run bundle
and runs `runtime/test_board.py` on a whole board image.

There is no local agent, no `sync/repo` text bundle, and no local pull to unblock. The old
`handoff/SYNC_INSTRUCTIONS.md` and ACK protocol are historical, not active.

---

## 2. Current state

- Old crop classifier run `20261125_065711` (0.7823 on public crops) is **retired**: its class map
  made `no_good` a bridge. Kept in `option_b_public_pilot/` for the record only.
- New architecture adopted: full-board YOLO11n joint detector + MobileNetV3-Small classifier.
  Decision: `docs/DECISIONS/2026-11-25-full-board-overhaul.md`.
- New pipeline code is in this repo; no trained bundle has been published yet. It needs a Colab run
  with credentials in Colab Secrets.
- Data cache: Drive `AI-PCB-Inspector/data_cache/`. Immutable runs: `runs/<UTC id>/`. Current bundle:
  `exports/current/`.

---

## 3. Startup checklist (every session)

1. `drive.mount('/content/drive')`. If it fails, stop and tell the user; do not train.
2. Read `agent.md`, `colab-agent.md`, `docs/LABELS.md`, and Drive `handoff/HANDOFF.json`.
3. Confirm secrets exist in **Colab Secrets**: `KAGGLE_USERNAME`, `KAGGLE_KEY`,
   optional `ROBOFLOW_API_KEY`, `GH_TOKEN`. Never print them.
4. Open `training/PCB_Inspector_Colab.ipynb` and Run all. Human step: review contact sheets and set
   `reviewed: true` only for defensible mappings.
5. Report in ≤3 lines: pipeline state, next run, blockers. Then continue with §4.

---

## 4. Priority order

1. **P0 — label semantics.** Never map `no_good` to bridge. Contact sheets must justify every
   `reviewed: true`. If bridge has no reviewed source, let it be dropped from `labels.txt`.
2. **P1 — full-board plumbing.** Verify `test_board.py --self-test` and synthetic composite boards
   exercise detector → classifier → verdict end to end. Plumbing only, not accuracy.
3. **P2 — honest split.** Group by board family (`CS*/R*/V*/Setup*`), fall back to source image
   with a recorded caveat. Tune on validation only; test split touched once for the final report.
4. **P3 — our boards.** When captures arrive in Drive `incoming/<board_id>/`, fine-tune and evaluate
   with whole board IDs held out. This is the only result comparable with the ≥80% requirement.
5. **P4 — Pi runtime.** Ask which runtime is installed (`tflite-runtime` assumed). Benchmark
   mean/p95 latency and temperature on the Pi; do not quote numbers from elsewhere.
6. **P5 — cloud/UI integration** is future work, owned by whoever builds the rig; Colab supplies
   the model contract only.

---

## 5. Export contract

Every immutable run folder under `runs/<run_id>/`:

```
detector.tflite       # one-class joint detector
classifier.tflite     # INT8 where supported
labels.txt            # classifier index -> class name
preprocess.json       # sizes, input ranges, output layout, thresholds
model_card.json       # checksums, scope, metrics, label warnings, licences
inspector.py          # exact runtime used in Colab evaluation
test_board.py         # local CLI
README.txt
manifest.sha256       # sha256sum-compatible
```

A `.pt` checkpoint is not a handoff. Git carries text only; models never enter git history.

---

## 6. Publishing

`python training/fullboard_colab.py publish --run-dir <run>` updates Drive `handoff/HANDOFF.json`
and `CHANGELOG.md`. `python training/fullboard_colab.py push` commits and pushes the cleaned repo
using `GH_TOKEN` from Colab Secrets, keeping remote history with a shallow fetch + `-s ours` merge.
The token is never printed or written to a file.

---

## 7. Hard rules

- Drive mount missing → stop. Checksum mismatch → redownload, never test a partial file.
- No Arduino. No on-Pi training. No fabricated accuracy/latency.
- No public image in the project's held-out test set; no synthetic defect in validation/test.
- Fixed seeds, saved configs, dated logs, clear licences and attribution.
- Ambiguous request → ask one short question rather than guessing.
