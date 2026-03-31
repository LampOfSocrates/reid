# Add Comparison-Friendly `comparison_*` Outputs For Coursework Runs

## What This Feature Does

Add lightweight comparison-friendly outputs to the coursework experiment flow so multiple runs can be compared easily from a notebook.

The goal is that after running experiments for models like:

- `mobilenet_v3_small`
- `resnet50`
- `clip_senet`

each run folder will contain additional files starting with `comparison_` that are easy to read from a notebook without changing the existing log file structure or format.

This should make it easy to paste a small notebook cell that scans run folders and compares:

- model name
- key hyperparameters
- final/best retrieval metrics
- total elapsed time
- average epoch time

## Files That Will Be Changed Or Created

- `coursework/main.py`
  Add comparison-oriented summary outputs written alongside the existing logs
- Possibly a new utility under `coursework/src/utils/`
  If a small helper for writing comparison summaries keeps `main.py` cleaner
- Possibly a notebook helper snippet file or small utility module
  Only if it helps keep the notebook paste-in experience simple
- `docs/plans/coursework-comparison-outputs.md`
  Feature plan document

## Implementation Steps

1. Inspect what run-time information is already available in `coursework/main.py`, especially:
   - architecture
   - save directory
   - elapsed total training time
   - per-epoch timing
   - evaluation metrics such as Rank-1 and mAP
2. Keep the existing logs untouched in structure and format.
3. Add new comparison-oriented artifacts in each run folder, with filenames beginning with `comparison_`, for example:
   - `comparison_summary.json`
   - `comparison_summary.csv`
4. Ensure those files include enough information for notebook-level comparison across runs.
5. If needed, add timestamped run directory support or align with the current save-dir behavior so repeated runs do not overwrite each other.
6. Provide a small notebook-friendly example for scanning those `comparison_*` files and building a table across runs.

## Risks Or Open Questions

- If some metrics are only visible in printed text and not retained in variables, `main.py` may need small refactoring to capture them cleanly.
- We should decide whether the comparison summary stores final metrics, best metrics, or both. Best practice is to store both when available.
- If repeated runs still reuse the same `--save-dir`, comparison outputs could still be overwritten unless the save-dir path is timestamped or otherwise made unique.
