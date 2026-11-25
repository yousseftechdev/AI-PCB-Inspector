# Sync changelog

- 2026-11-25 (colab-agent): Option B notebook reran to completion on CPU; run `20261125_065711` published, public-data test accuracy 0.7823. Created this handoff folder and `colab-agent.md` mirror.
- 2026-11-25 (colab-agent): local filtered git fetch/merge failed at ~70% on the user's connection. Declared git unusable as the transport; published text bundle `sync/repo/` (315 KB + MANIFEST.sha256) and `handoff/SYNC_INSTRUCTIONS.md`. Colab will not git push. User will gdown the bundle into the local repo root.
- 2026-11-25 (colab-agent): CORRECTION - origin is configured; Colab git push is fast and recommended. Local plain pull remains blocked by the 1.2 GB parent commit. Rewrote SYNC_INSTRUCTIONS.md with blob-filtered fetch (sparse-checkout first) plus the gdown bundle fallback. Pushing text commits from Colab.
- 2026-11-25 (colab-agent): local agent is on Windows; added PowerShell Get-FileHash / robocopy equivalents to handoff/SYNC_INSTRUCTIONS.md.

- 2026-11-25: Workflow overhauled to Colab-only. Crop-only pilot retired; full-board detector+classifier pipeline, label rulebook, local test script and git push path added. No new run published yet.
