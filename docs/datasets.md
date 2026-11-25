# Public PCB / solder-defect datasets — survey and verdict

> **2026-11-25 update:** the crop-only pilot and its class map are retired. The active label
> policy is [`LABELS.md`](LABELS.md); the active pipeline is [`ARCHITECTURE.md`](ARCHITECTURE.md).
> `no_good` is component misplacement and is **never** a solder bridge. Public data remain
> feasibility-only.

Status: **Colab pilot executed 2026-11-24; artifacts in `data/external/`** · Purpose: find out
whether we can train the inspector on existing public images instead of building our own dataset.

**Short answer: no public set replaces project-specific captures or the final evaluation.** There
are useful pilot/pretraining sources, but their optics, board populations, and labels do not match
our planned inspection rig and classes. Details and the proposed approach are below.

---

## 1. What we are looking for (the acceptance criteria)

A candidate has to satisfy all six to be usable as our training data:

1. **Assembled boards with solder joints** — not bare copper boards, not X-ray, not IC-lead-only.
2. **Captured like our rig** — top-down, diffuse light, a whole small board in frame, ~5 MP, no
   line-scan/telecentric distortion. Otherwise the model learns their optics, not our defects.
3. **Our class vocabulary** — good / insufficient solder / bridge / missing component, with
   annotations in boxes or per-crop labels.
4. **Open licence** that permits academic use *and* augmentation/distribution of a derived model.
5. **No login wall** where we can't get it, and small enough to move around (≤ a few GB).
6. **Enough volume and board diversity** — ≥ 50 distinct physical boards, otherwise the
   train/test split is not honest and generalisation can't be measured.

Spoiler: criteria 2, 3 and 6 are where every single public dataset fails.

### Colab pilot candidates

The official [Kaggle SolDef_AI dataset](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) is 1,178,779,597 bytes and the Kaggle API reports its dataset licence as **GPL-3.0**. It contains solder-joint images with `good`, `exc_solder`, `no_good`, `poor_solder`, and `spike` labels. Download requires a Kaggle account. Use it as a Colab-only pilot/pretraining source, attribute it, and keep it separate from project data. The paper's CC BY 4.0 article licence does not replace the dataset's listed GPL-3.0 terms.

The [Roboflow PCB solder-joint project](https://universe.roboflow.com/work-6qkmv/pcb-solder-joint) is the second Colab candidate. The existing survey records its export as CC BY 4.0, with `GOOD`, `COLDSOLDER`, and `STICKSOLDER`; the user reports having a Roboflow account. Verify the licence in the logged-in project/export page and visually inspect representative examples before mapping class semantics. Do not assume `STICKSOLDER` means a bridge or that `COLDSOLDER` means insufficient solder without checking its labeling guide/examples.

The GitHub repository [`hshgogogo/pcb-solder-defect-yolov8-experiment`](https://github.com/hshgogogo/pcb-solder-defect-yolov8-experiment) also contains a five-class solder-joint dataset and prepared splits, but has no repository licence and no demonstrated board-independent split. Do not use it unless upstream provenance and terms are confirmed.

---

## 2. The datasets

"Access" and "licence" were checked on 2026-11-24. Where a page is login-walled, the licence was
**not** verifiable — that is marked ⚠ and must be checked by a human with an account before use.

### 2.1 Solder-joint / assembled-board defect datasets (the relevant family)

| Dataset | Content | Size | Classes | Access | Licence | Fit |
|---|---|---|---|---|---|---|
| **SolDef_AI** (STIIMA-CNR, 2024, JMMP 8(3):117) | SMT caps/resistors (0603–1206) on 6 board types, 3 viewpoints, polygon annotations, Mask R-CNN baseline | 1,150 images / 460 joints; Kaggle archive ~1.18 GB | `good`, `exc_solder`, `no_good`, `poor_solder`, `spike` | Kaggle account + API key; download from Colab | **GPL-3.0** per Kaggle API metadata | Best candidate for good/poor-solder pilot; `exc_solder` is not automatically a bridge |
| **Roboflow "PCB solder joint"** (by *Work*) | Solder joints, multi-label boxes | 1,626 images | `GOOD`, `COLDSOLDER`, `STICKSOLDER` | Roboflow account/API key; download from Colab | CC BY 4.0 per existing survey; verify on project/export page | Candidate for all three pilot labels only after visual/definition audit |
| **Roboflow "SolDef_AI for defect detection"** (mirror) | Same content as SolDef_AI, converted to instance-seg | 429 images | 5 SolDef classes | Roboflow free account | **CC BY 4.0** (verified) | Convenient re-upload; prefer the original |
| **IEEE DataPort "Defect Detection Dataset"** (2026) | Real production line, 2,451 boxes, every image has ≥ 2 defect types | 654 raw (+aug) / 1.84 GB | `cold_joint`, `component_shift`, `missing_part`, `insufficient_solder`, `short_circuit` | IEEE DataPort — **login required** | ⚠ not verifiable | Taxonomy is *exactly* ours, but small and login-walled. Worth one request |
| **PCBA-DET** (Sci. Reports 14:19287, 2024) | PCBA photos with motherboard/fan screws, wiring and scratches; not solder-joint defects | 4,000 images + 2,384 augmented images (per repository README) | 8 screw, wiring and scratch classes | GitHub repo + Google Drive links | ⚠ repository has no dataset licence; its README points to an upstream dataset | Not suitable for our solder-joint classes; prior description of its classes and fit was incorrect |
| **PCB-AoI** (KubeEdge-IANVS, 2022) | Solder-paste inspection on IC leads, 231 real boards upscaled to 1,200+ | ~1,211 images | 2: `bad_solder_slope`, `lifted_lead` | Kaggle — free account | ⚠ check page | Narrow (IC leads only), single defect family. Low value |
| **Mobiusi "PCB Solder Joint Defect Detection"** | High-res controlled lighting, AOI-style | 15,000 records / 3.2 GB | multiple solder defect types | HuggingFace copy is **access-walled** (API returned auth error); vendor site requires signup | **CC BY-NC-SA 4.0** — non-commercial, share-alike | NC + ShareAlike is awkward for a published derived model. De-prioritise |
| **Roboflow "AI PCB solder DEFECT INSPECTION"** (FYP1) | Student project | 107 images | **20 inconsistent duplicates** (`Insufficient`, `Insufficient_solder`, `insufficient solder`, …) | Roboflow free account | CC BY 4.0 | Too small, labels unusable without a full re-mapping. Skip |
| **`solder-joint-dataset`** (GitHub, zhouyongbing125-rgb) | Flat images, YOLO labels | 50 images | 5 defect types | GitHub — direct | ⚠ none stated | Demo-sized only |
| **`AndyLiu0104/Soldering-Data-V3`** (HuggingFace) | Blue PCBA photos with **LLM-written captions**, not structured labels | 1,542 images | free-text (`bridge`, `excess solder`, …) | HuggingFace — public, direct download ✅ **verified downloadable** | ⚠ none stated, provenance unclear | Would need caption parsing → weak labels. Only as a last resort |

### 2.2 Bare-board / component datasets (mostly *not* what we need)

| Dataset | Content | Size | Why it doesn't fit |
|---|---|---|---|
| **DeepPCB** | Line-scan bare boards, template/test pairs | 1,500 pairs | Etching defects (`open`, `short`, `mousebite`, `spur`, `copper`, `pinhole`) — no solder, no components |
| **HRIPCB / PKU-Market-PCB** | Bare board photos | 693 (+aug 10k+) | Same. Useful only for a detector pretrain |
| **DsPCBSD+** (Sci. Data 11:811, 2024) | Bare-board surface defects incl. scratches, foreign objects, hole breakout | 10,259 images | Right *idea*, wrong defects (surface, not solder) |
| **FICS-PCB / FPIC** (trust-hub, PhysicalDB) | 93–261 boards, 71k+ annotations: components, traces, vias, text | 9,912 images | **Registration required**; annotation is component/text localisation, not solder quality. Good for a *component-detection* stage |
| **FPIC-Component** (DatasetNinja mirror) | Instance-seg subset | 6,260 images / 5.1 GB | **CC BY-NC-ND 4.0 — "NoDerivatives"**, so augmentation is arguably forbidden. Avoid |
| **FPIC roboflow mirror**, **PCB-Bank**, **VisA (PCB)**, **Dataset-PCB**, **Micro-PCB Images** | Various | 51 – 4,416 images | Catalogued by third parties (UniPCB `pcb-dataset-atlas`), **not independently verified**. Component/anomaly oriented |

---

## 3. Verdict: the gap is real, and it is structural

None of the above can serve as *our* dataset. Four independent reasons, in order of severity:

1. **Domain shift.** Candidate solder datasets use different optics, working distances and lighting
   from our OV5647 and proposed diffuse-light enclosure. Solder appearance depends on both the joint
   and image-capture conditions, so public images cannot establish performance on our rig. Local
   capture and fine-tuning may reduce the shift, but the result must be evaluated on held-out images
   from our setup.
2. **Class semantics don't transfer.** `exc_solder` in SolDef_AI means "too much solder, maybe
   bridging"; their `spike` has no equivalent in our list; their `no_good` is *component
   misplacement*. Bare-board `open`/`short` are copper etching faults, not solder faults. Building a
   mapping by hand means inventing a taxonomy and then presenting it as ground truth — the exact
   thing agent.md §11 forbids.
3. **Board diversity is too low to prove generalisation.** SolDef_AI is 6 physical boards. Even if we
   trained on it perfectly, we would have no way to demonstrate generalisation to *unseen* boards,
   and the exhibition rubric requires exactly that.
4. **The rules require our own dataset.** Challenge requirement #4: provide "datasets and a rubric"
   at the exhibition. A downloaded dataset fails that requirement even if it worked.

### Add to that: access friction

Of the original candidates, **six** need an account or subscription (Kaggle ×3, Roboflow ×3,
IEEE ×2, trust-hub, Mobiusi), **two** have no stated licence, and **one** forbids derivatives.
The approved pilot path uses the official Kaggle source (GPL-3.0 metadata) and the account-accessible
Roboflow project; the Colab notebook downloads into `/content`, not `data/external/` on the PC.
The unlicensed GitHub mirror remains excluded.

---

## 4. What the public data *is* genuinely good for

Three uses that are honest, documentable, and save us weeks:

1. **Latency pilot, week 1 (highest value).** R1 (response time) can be measured by running 100
   inferences on the Pi 4 with accessible PCB images. This tells us whether the Pi 4 can hit the time
   budget **before** we build boards. Do not assume an exported INT8 model or benchmark images are
   available until they have been located and checked.
2. **Warm-start / pretraining.** Fine-tune a classifier on public crops, then on our board images
   when available. Keep public images out of the project's physical-board test set and record each
   source and licence.
3. **Baseline comparison table** for the portfolio: "public-benchmark mAP vs our-board accuracy" makes
   the domain-shift argument with numbers we generated, which is a strong slide.

**Never:** report accuracy measured on a public dataset as ours, or include a public image in our
held-out test set.

---

## 5. The plan I recommend (hybrid dataset)

```
training/
  stage 1  pretrain   SolDef_AI (1,150) + Roboflow PCB-solder-joint (1,626)   -> generic solder appearance
  stage 2  fine-tune   our boards: ~15-20 physical boards, ~800-1,500 crops    -> our rig, our defects
  stage 3  evaluate   held-out boards ONLY (never seen in stage 1 or 2)       -> the >= 80 % number
```

Deliberate, disclosed use of both sources. This is the standard industrial answer to "not enough
defects" and it is easy to defend in front of judges.

### Where the real bottleneck is, and the ways around it

Our own images are the blocker, specifically *defective* ones. Options, cheapest first:

| Approach | Effort | Honesty / risk | Verdict |
|---|---|---|---|
| **A. Fabricate defect boards in the lab** (excess solder bridge, too little solder, missing component, rotated part) on 15–20 practice boards | 2–3 lab sessions, ~EGP 500–1000 | Fully legitimate, best evidence, real variation | **Do this. It is the plan.** |
| **B. Digitally synthesised defects** (warp/blur/erase solder to fake a cold joint; paste a bridge patch) applied to photos of *our own* good boards | 1 day of scripting | Legitimate **only if disclosed as augmentation** and kept out of the test set. Cannot be used to claim accuracy | Use to **top up** rare classes |
| **C. One good board + many variations** (angle/position/lighting jitter, 30 shots each) | 1 session | Inflates numbers if split by image. Must split by board | Combine with A, never alone |
| **D. Get real scrap boards** from a local electronics repair shop / school lab / e-waste recycler — real defects for free | 1 afternoon of asking | Excellent: real defect statistics | **Try this first — it's free and fast** |
| **E. Ask another STEM school / previous cohort** for their boards or dataset | 1 message | Fine, with attribution | Worth one email |
| **F. Buy a pre-made "solder practice kit" with intentional faults** | ~EGP 300 online | Legitimate | Fallback for A |

So the answer to "are there suitable datasets online" is: **there are promising pretraining/pilot
candidates, but no verified drop-in dataset for our final accuracy claim.** Our own physical captures
remain necessary. Synthetic defect edits can be disclosed as training-only augmentation; they cannot
replace physically defective examples in validation/test or substantiate real-world accuracy.

---

## 6. Current decisions and unresolved blockers

- The user selected **Option B** (`good`, `solder_bridge`, `insufficient_solder`) and plans to train
  in **Google Colab Free** with Kaggle and Roboflow account access.
- The user has no lab access and has one perfboard, a personal soldering iron, and solder; the
  budget is very limited. Use that single perfboard for a small proof-of-process only: record joint
  IDs and take fixed-condition images before/after carefully making known bridge and low-solder
  examples if safe. Multiple joints/images on that same board are not independent board samples and
  cannot support a board-generalization test or final accuracy claim. Do not risk sensors or
  microcontrollers as defect samples.
- Target by Sunday 2026-11-29: complete the Colab download/audit workflow and produce a reproducible
  public-data Option B baseline if the audited public labels support all three classes.
  **Status 2026-11-24: done** — the download/audit workflow and the public-data pilot ran to
  completion on Colab (see the executed block below).
### Colab pilot executed 2026-11-24

The Colab notebook `training/notebooks/option_b_colab.ipynb` + companion
`training/notebooks/option_b_colab.py` ran to completion in hosted Colab (Free tier, CPU). The
audited artifacts are committed in the repo:

- Datasets: `data/external/soldef/` (Kaggle SolDef_AI `_annotations.coco.json`, 428 images, 671
  annotations) and `data/external/audit_summary.json` with `data/external/audit_previews/`.
  No Roboflow data was available this run.
- Run: `training/runs/option_b_public_pilot/run_20261124_224758/` — `option_b_mobilenetv3_small.pt`,
  `metrics.json`, `split_manifest.csv`, `training_config.json`, `crops/`. Pilot `test_accuracy`
  0.7903 (CPU, 10 epochs); see `metrics.json` for per-class precision/recall and the confusion
  matrix.

Reproducing requires Kaggle username/key and Roboflow API key in Colab Secrets (not chat, notebook,
  source code or repository); visually audit labels and fill the explicit class mapping in
  `training/notebooks/option_b_colab.ipynb` — the defaults map nothing, and training stops unless
  all three classes are supported by reviewed labels.
- Public dataset splits do not prove physical-board independence. Report this pilot as public-data
  feasibility only, never as the final ≥80% project accuracy claim.

Remaining ask: can a teammate, school, or repair shop provide additional inexpensive/scrap boards?
Who can physically verify Roboflow's current licence and class definitions in the logged-in UI?
Credentials should be added directly to Colab Secrets.

---

## 7. Sources

- SolDef_AI — MDPI *JMMP* 8(3):117 (2024), doi:10.3390/jmmp8030117 · [official Kaggle dataset](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) (GPL-3.0 per Kaggle API metadata)
- Roboflow Universe: `work-6qkmv/pcb-solder-joint` (1,626, CC BY 4.0) · `fyp1-mtj8f/ai-pcb-solder-defect-inspection` · `defectdetection-e5sqy/soldef_ai-for-defect-detection`
- PCBA-DET — *Scientific Reports* 14:19287 (2024), doi:10.1038/s41598-024-70176-1 · github.com/ismh16/PCBA-Dataset
- Fast-access experiment mirror (dataset provenance/licence unresolved): https://github.com/hshgogogo/pcb-solder-defect-yolov8-experiment
- PCB-AoI — KubeEdge-IANVS benchmark, kaggle.com/datasets/kubeedgeianvs/pcb-aoi
- IEEE DataPort — "Defect Detection Dataset" (2026) and "part of multiscale defects PCBA dataset"
- FICS-PCB (2020, eprint 2020/366) / FPIC (ACM JETC 19(2):17, 2023) — trust-hub.org, physicaldb.ece.ufl.edu
- DsPCBSD+ — *Scientific Data* 11:811 (2024), doi:10.1038/s41597-024-03656-8
- Dataset catalogues used to find these: github.com/asfx0412/pcb-dataset-atlas, github.com/asfx0412/UniPCB
- Roboflow PCB-solder-suite examples of the two-stage architecture: github.com/yuva-karthik/solder-defect-detection
