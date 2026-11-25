# Colab-only workflow (2026-11-25)

The old two-agent setup is retired. There is no local training step, no `sync/repo` bundle, no
local agent to acknowledge runs, and no 1.2 GB git pull problem.

---

## 1. Where things live

| Thing | Location | Notes |
|---|---|---|
| Source code, docs, configs | GitHub `yousseftechdev/AI-PCB-Inspector` | Pushed from Colab. Source of truth. |
| Working copy in Colab | Drive `AI-PCB-Inspector/repo/` | Edited and committed by the Colab agent. |
| Raw datasets (1.2 GB) | Drive `AI-PCB-Inspector/data_cache/` | Never in git. Cached so reruns are cheap. |
| Immutable runs | Drive `AI-PCB-Inspector/runs/<UTC id>/` | Models, card, checksums, metrics. |
| Current accepted bundle | Drive `AI-PCB-Inspector/exports/current/` | What you download. |
| Handoff state | Drive `AI-PCB-Inspector/handoff/HANDOFF.json` | Points at the current run. |
| `colab-agent.md` mirror | Drive `AI-PCB-Inspector/colab-agent.md` | Survives VM resets. |

Git carries text only. Models, crops, raw datasets and logs never go into git.

---

## 2. Secrets (Colab Secrets only)

Add in the Colab sidebar, never in chat, a notebook output, a Drive text file or git:

- `KAGGLE_USERNAME`, `KAGGLE_KEY` — SolDef_AI download.
- `ROBOFLOW_API_KEY` — optional; needed only if we use Roboflow as a bridge source.
- `GH_TOKEN` — GitHub PAT with write access to the remote repo. Used only by the push cell.

The notebook injects them into the environment without printing them.

---

## 3. Run sequence

Open `training/PCB_Inspector_Colab.ipynb` and **Run all**:

1. Mount Drive, install pinned-ish dependencies.
2. Download/cache data.
3. Generate audit contact sheets and display them.
4. **Human step**: review contact sheets, set `reviewed: true` only for defensible mappings in
   `training/configs/fullboard_config.json`.
5. Prepare data, train detector + classifier, export TFLite bundle, publish to Drive.
6. Push the cleaned repo to GitHub.
7. Verify `exports/current/manifest.sha256` in the last cell.

Every run gets a new UTC id. Nothing is overwritten except `exports/current`.

---

## 4. What you download locally

From Drive `AI-PCB-Inspector/exports/current/` download the folder (or zip it in Drive first):

```
detector.tflite
classifier.tflite
labels.txt
preprocess.json
model_card.json
inspector.py
test_board.py
README.txt
manifest.sha256
```

On the PC or Pi:

```bash
pip install numpy pillow tflite-runtime   # on a PC: pip install numpy pillow tensorflow
cd bundle
sha256sum -c manifest.sha256              # PowerShell: see deploy/README.md
python test_board.py --bundle . --image /path/to/whole_board.jpg
```

The CLI writes `<board>_annotated.jpg` and `<board>_report.json`. It accepts the whole board;
`--self-test` generates a synthetic board to check the install.

Checksums must pass before a model is evaluated. A partially downloaded bundle is never tested.

---

## 5. Failure rules

- Drive not mounted → stop, remount.
- Missing `GH_TOKEN` → push cell fails with a clear message; code remains committed locally in the
  Drive working copy, so no work is lost.
- Checksum mismatch → delete the partial file and redownload it.
- Ambiguous label → leave `reviewed: false`; the pipeline either stops or marks the run provisional.
- No reviewed bridge source → bridge class is dropped; the model cannot call a bridge.
