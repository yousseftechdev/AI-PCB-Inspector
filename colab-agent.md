# colab-agent.md — Google Colab training agent brief

Read this **after** `agent.md` at the start of every Colab session. It is the persistent instruction
file for the agent running inside Colab: it defines the job, the boundary with the local agent, the
current state, and the sync protocol for large artifacts.

**If this file is missing, stale, or the repo looks unfamiliar after a VM reset**, read the mirror at
`/content/drive/MyDrive/AI-PCB-Inspector/colab-agent.md` first and copy it back into the repo. Google
Drive survives a Colab reset; `/content` does not.

Last updated: **2026-11-25** (Colab session). Items marked **TBC** need a human decision.

---

## 1. Division of labour

| | Colab agent (this agent) | Local agent (PC / Pi) |
|---|---|---|
| **Owns** | Training algorithm, dataset preparation, splits, experiments, metrics, model selection, model export, handoff artifacts | Pi runtime: capture, ROI locator, inference wrapper, pipeline, actuators, cloud/UI, on-device benchmarks, hardware calibration |
| **Writes** | `training/`, `data/splits/`, `data/labels/`, experiment configs, model artifacts in Drive | `src/`, `tests/`, `hardware/`, `results/pi_*`, deployment configs |
| **Does not do** | Pi deployment code, live camera/sensor code, hardware claims | Training, changing loss/splits/architecture without agreeing here |
| **Shared** | `docs/` (dated decision records), `handoff state in Drive` | same |

Rule of thumb: **if it runs on the Pi, the local agent owns it. If it needs a GPU/Colab runtime to
train, this agent owns it.** The seam between us is the exported model contract in §5: the local
agent should never need to open a `.pt` file or know PyTorch.

If asked to do local-agent work, do not silently take over. Produce the model-side artifact/advice and
hand it off instead, unless the user explicitly reassigns the job.

---

## 2. Verified current state (2026-11-25)

The notebook `training/notebooks/option_b_colab.ipynb` **ran to completion again today** on Colab CPU.
This is a rerun/refinement of the 2026-11-24 pilot, not a new architecture.

- Colab run directory: `/content/option_b_run_20261125_065502/`
- Drive copy: `/content/drive/MyDrive/AI-PCB-Inspector/option_b_public_pilot/run_20261125_065711/`
- Model: MobileNetV3-Small (ImageNet init), 128×128 crops, CPU
- 10 epochs, AdamW lr `2e-4`, weight decay `1e-4`, class-weighted cross-entropy, seed `42`
- Splits grouped by **source image** only: train 414 / val 133 / test 124 crops
- Crop counts: `good` 368, `solder_bridge` 246, `insufficient_solder` 57; test support 59 / 52 / **13**
- **Test accuracy 0.7823** (previous pilot 0.7903). Bridge recall **0.731**, insufficient-solder
  recall **0.692**; final train accuracy 0.908 vs val 0.782 → overfitting.
- Only SolDef_AI was used. Roboflow contributed nothing (licence still unconfirmed).
- **Public data only.** This is feasibility/plumbing evidence, **never** the ≥80% project claim.

### Data-integrity warning — read before optimizing anything

The current class map in `training_config.json` is:

```
good -> good, exc_solder -> good, poor_solder -> insufficient_solder,
no_good -> solder_bridge, spike -> solder_bridge
```

`docs/datasets.md` §3 says SolDef's `no_good` is **component misplacement**, not a solder bridge, and
that `exc_solder` is not automatically a bridge (mapping it to `good` is equally unproven). The
0.78 number may therefore be measuring the wrong task. **P0 below is to resolve this with visual
evidence before chasing accuracy.** Do not tune hyperparameters against an indefensible label map.

Also note: the repo already carries ~1.2 GB of public images under `data/external/soldef/`. Never add
more binaries to git.

---

## 3. On every fresh Colab session — startup checklist

1. `drive.mount('/content/drive')`. If the mount fails, stop and tell the user; do not retrain.
2. Read `agent.md`, this file, and `drive/MyDrive/AI-PCB-Inspector/handoff/HANDOFF.json`.
3. Read `handoff/REQUESTS.md` (local agent → Colab) and any `handoff/ACK_*.md` receipts.
4. Get the latest code: `git pull` if a remote exists (this VM currently has **no remote configured**),
   otherwise copy from Drive `sync/repo/` if the local agent published there. Never overwrite
   uncommitted local work blindly.
5. Load Kaggle / Roboflow credentials from **Colab Secrets only**. Never print, echo, commit, or paste
   a key into chat or a notebook cell output.
6. Report to the user in ≤3 lines: latest published run, next planned experiment, and blockers.
   Then continue with the highest-priority item in §4 without waiting to be re-prompted.

---

## 4. Training / optimization backlog (priority order)

Work one item at a time. Every experiment gets a new immutable run id, a saved config, a fixed seed,
and a `metrics.json`. **Tune on validation only; the test split is touched once for the final report.**

1. **P0 — label semantics.** Inspect `data/external/audit_previews/` and the COCO polygons for
   `no_good`, `exc_solder`, `spike`, `poor_solder`. Write a dated decision in `docs/datasets.md`
   with example images. If `no_good` really is misplacement, drop it or map to a new class — do not
   call it `solder_bridge`. If only two honest classes survive, train a narrower pilot and say so.
2. **P1 — honest split.** Replace the source-image group key with the path-derived board family
   `CS*/R*/V*/Setup*` (27 groups in SolDef). Note that `no_good/good` live mostly in `V1/Setup1`
   while `spike/exc_solder/poor_solder` live in `V2/V2.1`, so the current split can leak family
   clues. Re-run with the stricter group key even if accuracy drops; the lower number is the honest one.
3. **P2 — rare class and recall.** `insufficient_solder` has 57 crops / 13 test samples. Try
   weighted sampling or focal loss, and tune a decision threshold specifically to raise solder-bridge
   recall toward the draft **R3 ≥ 90%** target. Report the precision cost explicitly.
4. **P3 — capacity / regularization.** GPU if available, 160–192 px crops, more epochs with early
   stopping on validation, cosine LR schedule, stronger augmentation, dropout/label smoothing.
   Compare against the frozen baseline on the *same* split; no cherry-picked runs.
5. **P4 — export contract.** Publish `model.tflite` (INT8 with a representative set drawn only from
   train), `labels.txt`, `preprocess.json` (input size, normalization, crop convention), and
   `model_card.json` (class map, split policy, metrics, checksum, licence). Ask the local agent which
   runtime they target (`tflite-runtime` vs NCNN) before choosing the final export.
6. **P5 — our boards.** When the local agent uploads captures to Drive `incoming/<board_id>/`,
   fine-tune and evaluate only on board IDs held out from both training and augmentation. This is the
   only result that may be compared with the ≥80% requirement.

Never present public-data numbers as project accuracy, and never let synthetic defect edits into
validation/test.

---

## 5. Model contract handed to the local agent

The local agent implements `Inspector.load(path)` / `Inspector.inspect(image)` from
`docs/approaches.md` §5. Our export must therefore be self-describing:

```
runs/<run_id>/
  model.tflite        # INT8, input [1, H, W, 3], uint8 or float as declared in preprocess.json
  labels.txt          # index -> class name, same order as training
  preprocess.json     # input size, resize/crop rule, normalization, score threshold
  model_card.json     # sha256 of model.tflite, metrics, class map, licences, date
  metrics.json, training_config.json, split_manifest.csv, manifest.sha256
```

A `.pt` checkpoint alone is **not** a handoff: it requires PyTorch and does not declare preprocessing.
Keep it in the run folder for reproducibility, but the Pi consumes the TFLite bundle.

---

## 6. Sync plan — how the two agents exchange work

The internet is too slow/unstable to pull weights or datasets through git, and a failed mid-download
must not corrupt a run. **Git carries text/code only. Google Drive carries every binary.** Drive is
already mounted read-write in Colab at `/content/drive`, which is why it is the transport.

### 6.1 Channels

| Content | Channel | Reason |
|---|---|---|
| Source code, docs, configs, small text reports | Git repo (commit + push) | diffable, versioned, small |
| Weights, TFLite exports, crops, raw datasets, large logs | Google Drive **only** | resumable download, no git history bloat |
| Current state / pointers / requests | Drive `handoff/` (plus a tiny receipt in repo `results/`) | one source of truth across resets |

Do **not** `git add` `*.pt`, `*.tflite`, `data/external/`, or crop folders for new work. Do not rewrite
existing git history to remove the already-committed 1.2 GB without explicit user approval.

### 6.2 Drive layout

```
MyDrive/AI-PCB-Inspector/
  colab-agent.md                 # mirror of this file; survives VM reset
  handoff/
    HANDOFF.json                 # latest published run, status, checksums, next action
    REQUESTS.md                  # local agent -> Colab: experiments/data requests
    CHANGELOG.md                 # dated entries from both agents
    ACK_<run_id>.md              # local agent confirms it received/tested a run
  runs/<run_id>/                 # immutable experiment bundles (see §5)
  incoming/<board_id>/           # local agent -> Colab: our-board images + labels CSV
  exports/current/               # stable copy of the accepted best bundle for the Pi
  sync/repo/                     # fallback code transport if no git remote is available
```

Run ids are UTC timestamps `YYYYMMDD_HHMMSS`. **Never overwrite a run folder.** Keep each run folder
to ≤8 files (gdown's folder mode has a 50-file cap).

### 6.3 Publishing from Colab (my side)

1. Finish training → write the complete run folder under `runs/<run_id>/`.
2. Write `manifest.sha256` (`sha256sum`-compatible lines, relative filenames).
3. Update `handoff/HANDOFF.json`: `run_id`, `status` (`ready_for_local_test` / `superseded` /
   `blocked`), file paths, sha256 of `model.tflite`, classes, input size, test accuracy (labelled
   `public_pilot` or `held_out_boards`), licence, and `requested_action`.
4. Append one dated line to `handoff/CHANGELOG.md`. Do not mark the previous run superseded until
   the new bundle's checksums verify.

### 6.4 Pulling on the local PC (no git for weights)

```bash
pip install -U gdown
mkdir -p runs/<run_id>
gdown --continue -O runs/<run_id>/model.tflite '<Drive share link for model.tflite>'
gdown --continue -O runs/<run_id>/labels.txt '<...>'
gdown --continue -O runs/<run_id>/preprocess.json '<...>'
gdown --continue -O runs/<run_id>/model_card.json '<...>'
gdown --continue -O runs/<run_id>/manifest.sha256 '<...>'
cd runs/<run_id> && sha256sum -c manifest.sha256
```

`--continue` resumes an interrupted transfer on the next attempt; rerunning the exact same command is
safe. For many small files use
`gdown --folder --continue --remaining-ok '<folder share link>'`. Alternatives if gdown is awkward:
Google Drive desktop client with selective sync, or
`rclone copy --transfers 1 --retries 10 --checksum remote:AI-PCB-Inspector/runs/<run_id> ./runs/<run_id>`.

**A file with a failed checksum is never evaluated or deployed.** Delete the partial file and resume.
Never open a half-downloaded model "just to test quickly".

### 6.5 Local → Colab and acknowledgement

- Code/docs: commit + push if a remote exists; otherwise upload the changed text files to
  `sync/repo/` in Drive. (This VM has no remote, so Drive is currently the reliable path.)
- Our-board images: `incoming/<board_id>/` plus a CSV with `board_id, joint_id, label, camera_settings,
  defect_method`. Hold out whole board IDs; a joint from a test board must never appear in training.
- After receiving a run, the local agent writes `handoff/ACK_<run_id>.md` with the exact sha256
  tested, Pi latency (mean/p95), memory, and any inference mismatch. That receipt is the signal for
  me to publish the next experiment.
- If two consecutive runs get no ACK, stop publishing new runs and ask what is blocking the Pi side.

### 6.6 Failure rules

- Drive mount missing → stop, report, do not retrain.
- Checksum mismatch → treat the artifact as corrupt; resume the download.
- Secrets → Colab Secrets / local `.env` only; never in Drive handoff files, repo, or chat.
- Ambiguous request → ask one short question rather than guessing (per `agent.md` §11).

---

## 7. TBC — questions to resolve with the user / local agent

1. Confirm the true meanings of SolDef `no_good`, `exc_solder`, `spike` before any more training.
2. Can the local agent share a Drive folder link (or configure a git remote) so both machines can
   reach the same handoff directory?
3. Which Pi runtime is chosen: `tflite-runtime` (INT8) or NCNN? This fixes the final export format.
4. Any scrap/expendable boards available for the held-out board test? One perfboard is not enough.
5. Roboflow licence and class definitions inside the logged-in account — still unverified.

---

## 8. Hard rules inherited from `agent.md`

- No Arduino-branded controllers. No on-Pi training. No fabricated accuracy or latency numbers.
- No public-data image in the project's held-out test set; no synthetic edit in validation/test.
- Reproducible: pinned deps, fixed seeds, saved config, dated logs, clear attribution/licences.
- Small, explained changes; ask before scope/hardware/architecture changes.
