# Label semantics — the rulebook (2026-11-25)

This document exists because the 2026-11-25 pilot learned the wrong task. It is the authority for
what each public category is allowed to mean in our model.

---

## 1. The bug we are fixing

Old run `20261125_065711` used:

```
good -> good
exc_solder -> good
poor_solder -> insufficient_solder
no_good -> solder_bridge     <-- wrong
spike -> solder_bridge       <-- unproven
```

`docs/datasets.md` §3 states that SolDef `no_good` is **component misplacement**, not a solder
bridge. Therefore:

- a **tilted / misplaced component** was a ground-truth `solder_bridge`;
- two pads that are simply **not connected** carried bridge or good labels depending on which
  public category the annotator chose;
- the model was rewarded for predicting "bridge" from rotation and from absence of solder.

No amount of hyperparameter tuning fixes a wrong target. The mapping had to change first.

---

## 2. Our canonical classes

| Class | Definition (must be visible in an example image) | Board effect |
|---|---|---|
| `good` | solder joint with continuous, concave fillet; no short to a neighbouring pad | PASS |
| `insufficient_solder` | visibly too little solder / poor wetting; pad or lead exposed where a fillet should be | FAIL |
| `solder_bridge` | solder physically connects two pads/leads that must be electrically separate | FAIL |
| `other` | a genuine annotation outside our taxonomy (misplacement, spike, excess solder) — reported for review, never collapsed into a bridge | REVIEW |

`other` is a deliberate "honest bin". It prevents an unknown defect from being forced into one of
the three project classes.

---

## 3. Source category policy

Every mapping in `training/configs/fullboard_config.json` has three fields:

- `label` — target class, or `null` = do not use for classification;
- `reviewed` — `true` only after a human looked at `runs/audit/<source>/<category>_contact.jpg`;
- `reason` — one sentence explaining the decision.

### SolDef_AI (Kaggle, GPL-3.0 per API metadata)

| Source category | Target | Reviewed default | Rationale |
|---|---|---|---|
| `good` | `good` | yes | Explicit class name, consistent with the paper. |
| `poor_solder` | `insufficient_solder` | no | Plausible; verify on contact sheet before using. |
| `exc_solder` | `other` | no | Excess solder is not a bridge. Never map to `solder_bridge`. |
| `no_good` | `other` | yes | Component misplacement per `docs/datasets.md`. **Never** a bridge. |
| `spike` | `other` | no | A spike is not automatically a short to another pad. |

### Roboflow `work-6qkmv/pcb-solder-joint` (verify licence in the logged-in project)

| Source category | Target | Reviewed default | Rationale |
|---|---|---|---|
| `GOOD` | `good` | no | Verify visually. |
| `COLDSOLDER` | `insufficient_solder` | no | Cold solder and insufficient solder are not synonyms by assumption. |
| `STICKSOLDER` | `solder_bridge` | no | Only after seeing actual pad-to-pad bridging in the contact sheet. |

All SolDef/Roboflow annotations, whatever their category, are used to train the **one-class joint
detector**. Category semantics only affect the classifier.

---

## 4. Enforcement in code

`training/fullboard_colab.py`:

- `check_label_policy()` raises if a used mapping has `reviewed: false`, unless the operator runs
  with `--allow-provisional`. Provisional runs are stamped with warnings in `metrics.json` and
  `model_card.json`.
- `solder_bridge` is removed from `labels.txt` when no reviewed source supplies it. The runtime
  cannot output a class that is not in `labels.txt`.
- There is no code path from `no_good` to `solder_bridge`. A regression test in `tests/` guards it.

---

## 5. What still needs evidence

1. Look at `runs/audit/soldef/poor_solder_contact.jpg` and confirm/deny `insufficient_solder`.
2. Look at `runs/audit/roboflow/STICKSOLDER_contact.jpg` and confirm/deny `solder_bridge`.
3. When our own boards exist, label them with this same rulebook. Any synthetic bridge used for
   training must be disclosed and kept out of validation/test.
4. Do not report the public-data accuracy as the project's ≥80% number.
