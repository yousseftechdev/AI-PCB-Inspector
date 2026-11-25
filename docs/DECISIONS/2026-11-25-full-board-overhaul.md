# Decision record — 2026-11-25: Colab-only, full-board inspector

## Context

The 2026-11-25 public-data pilot (`20261125_065711`) scored 0.7823 on 124 crop-level test samples
but behaved incorrectly on the user's local tests:

- a tilted component was reported as a solder bridge;
- two pads that were not connected were also reported as a bridge;
- the user had to supply crops, so a full board could not be inspected.

Root cause: the class map made `no_good` (component misplacement, per `docs/datasets.md`) a
ground-truth `solder_bridge`, and `spike`/`exc_solder` were also folded into bridge/good without
evidence. A secondary cause was that the model architecture required human-supplied crops.

The old workflow also split work between a "local agent" and a "Colab agent", which required
fetching a repo with a 1.2 GB parent commit on an unstable local link.

## Decision

1. **Architecture:** adopt a two-stage full-board pipeline — YOLO11n one-class joint detector,
   then MobileNetV3-Small classifier on detector boxes. The caller supplies a whole board image.
2. **Label semantics:** `no_good → other`, never bridge. All mappings require a `reviewed` flag
   backed by a contact sheet. Unreviewed mappings need an explicit `--allow-provisional` flag and
   are recorded as warnings. `solder_bridge` is omitted from `labels.txt` unless a reviewed source
   exists, making false bridge calls impossible.
3. **Workflow:** all development, training, export, handoff and git push happen in Colab. The
   local machine/Pi only downloads a bundle and runs `test_board.py`. `sync/repo/` and the
   local-agent handoff protocol are deprecated.
4. **Export:** TFLite detector + classifier, `labels.txt`, `preprocess.json`, `model_card.json`,
   `manifest.sha256`, shared runtime, local test CLI.
5. **Evidence:** public-data results remain feasibility-only. The ≥80% claim still needs our own
   physically held-out boards.

## Alternatives considered

- **Whole-board YOLO with defect classes:** rejected for now. ~671 annotated joints across 428
  close-up images would make tiny-object recall fragile, and defect classes would reintroduce the
  exact label-semantics bug into the detector.
- **Keep crop classifier and add a UI crop selector:** rejected. It does not satisfy "analyze the
  full board" and keeps the human in the loop for every joint.
- **Anomaly detection:** kept as a future second opinion; it cannot name defect classes, which the
  rubric requires.

## Consequences

- Two TFLite models to load instead of one; runtime stays small and shared.
- Harder to claim high accuracy until we have real full-board images: synthetic composite boards
  test the plumbing only.
- Git history can be cleaned without anybody fetching the 1.2 GB commit from a local PC.
