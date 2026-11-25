# Current requests (2026-11-25)

The local-agent request list is retired. The Colab agent owns all code, training and export.

## From the user (needed to publish the first full-board bundle)

- [ ] Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to **Colab Secrets**.
- [ ] Add `ROBOFLOW_API_KEY` to Colab Secrets if we want a reviewed `solder_bridge` source.
- [ ] Add `GH_TOKEN` (GitHub PAT with write access) to Colab Secrets so the notebook can push.
- [ ] Open the audit contact sheets in the notebook and set `reviewed: true` only for mappings you can defend.
- [ ] After the bundle is published, run `test_board.py` on a **whole board** image and report confusing cases.

## From the Colab agent (later, once a bundle exists)

- [ ] Which Pi runtime is installed: `tflite-runtime` (assumed) or something else?
- [ ] Capture full-board images under fixed lighting and share them via Drive `incoming/<board_id>/`.
- [ ] Obtain expendable boards for a physically held-out test set; one perfboard is not enough.
- [ ] Confirm Roboflow licence and `STICKSOLDER` definition in the logged-in project UI.
