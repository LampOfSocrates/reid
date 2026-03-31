# Add Run Comparison Support For Coursework Logs

## What This Feature Does

Add a way to compare completed coursework experiment runs across separate log folders, including both model-quality metrics and time taken.

The comparison feature should make it easy to inspect differences such as:

- architecture
- learning rate
- batch size
- data fraction
- best retrieval metrics such as Rank-1 and mAP
- total training time
- per-epoch time where available

Because repeated runs currently risk overwriting the same `--save-dir` path such as `logs/resnet50_fc512-veri`, this feature should also ensure that each run writes into a timestamped run directory.

Existing log file formats and structure should not be changed. Instead, any new structured comparison outputs should be written as additional files whose names begin with `comparison_`.

## Files That Will Be Changed Or Created

- `coursework/main.py`
  Needs small output-path changes so each run writes to a timestamped subdirectory derived from `--save-dir`
- `coursework/src/utils/`
  Likely a new utility module for aggregating and comparing run folders
- Possible new file:
  - `coursework/src/utils/compare_runs.py`
    or similar helper for scanning timestamped run directories and building summaries
- Possible new notebook:
  - `compare_runs.ipynb`
    if a notebook view is preferred for side-by-side inspection
- `docs/plans/coursework-run-comparison.md`
  Feature plan document

## Implementation Steps

1. Update the run output path logic so `--save-dir logs/resnet50_fc512-veri` becomes a timestamped run folder, for example under that base path or with a timestamp suffix/prefix, to avoid overwriting prior runs.
2. Inspect what each run currently writes into its log folder, including text logs, checkpoints, and any structured outputs.
3. Decide the minimum structured data needed for reliable comparison, especially:
   - final or best Rank-1 / mAP
   - total elapsed training time
   - per-epoch time when available
   - key hyperparameters
4. Keep all existing log files unchanged in structure and format; add only new comparison-oriented files such as:
   - `comparison_summary.csv`
   - `comparison_summary.json`
   - other `comparison_*` artifacts if needed
5. Add a comparison utility that:
   - scans multiple run folders
   - extracts metrics and timing
   - outputs a ranked comparison table
6. Optionally add a notebook for easier visual inspection of runs and plots.
7. Verify that the comparison output is useful for selecting the best run by both quality and runtime cost.

## Risks Or Open Questions

- Current run folders may not yet store all timing information in a structured format, so comparison code may need to parse the existing logs or rely on new `comparison_*` outputs generated alongside them.
- If the existing logs are mostly free-form text, parsing them robustly could be brittle, which is why preserving them and adding separate `comparison_*` artifacts is the safer path.
- We should decide whether “time taken” means total wall-clock time, average epoch time, or both. My recommendation is to capture both when practical.
