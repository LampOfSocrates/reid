# Coursework Experiment Sweeps

## What This Feature Does

Adds a repeatable way to run the coursework experiments for:

1. learning-rate exploration
2. batch-size exploration

The implementation will derive an experiment-specific log folder name from the active parameters and use that value in `save_dir`, so each run lands in a predictable directory without manual renaming. It will also add a simple loop-based entry point for launching the requested command variants.

The planned experiment values are:

- Learning rates: default `0.0003` plus `0.0001`, `0.001`, `0.005`, `0.01`
- Batch sizes: default `64` plus `8`, `16`, `32`, `128`

## Files To Change Or Create

- `coursework/main.py`
  Add or adjust save-dir naming so the final run directory reflects the experiment parameters.
- `coursework/args.py`
  Add any minimal CLI surface needed to support experiment naming or grouped experiment execution.
- `coursework/` runner script or helper module
  Add a small experiment launcher that builds the requested LR and batch-size sweeps and invokes `coursework/main.py` in a loop.
- `docs/plans/coursework-experiment-sweeps.md`
  This plan document.

## Implementation Steps

1. Define the experiment matrix for the two coursework questions:
   - 4 learning-rate values plus the default: `0.0001`, `0.0003`, `0.001`, `0.005`, `0.01`
   - 4 batch sizes plus the default: `8`, `16`, `32`, `64`, `128`, using the best LR from step 1
2. Add a compact parameter-to-folder-name formatter that converts the active experiment settings into a clean suffix such as model, optimizer, LR, and batch size.
3. Update save-dir handling so the requested base path can include the derived experiment identifier without the user manually composing it each time.
4. Add a runner that loops over the experiment configurations and launches `python coursework/main.py ... --save-dir logs/{runtime}/Qx/...` with the correct arguments for each run.
5. Keep the runner aligned with the exact coursework command shape you provided, including dataset, model, blur augmentation, and student environment variables.
6. Document any assumptions inline so it is easy to adjust the candidate LR or batch-size lists later.

## Risks Or Open Questions

- "Best LR" and "best batch size" are chosen after observing results, so the runner may need either:
  - fixed placeholder values that you edit after reviewing results, or
  - separate commands/stages rather than one single fully automatic sweep.
- `coursework/main.py` currently appends a timestamp to `save_dir`; we should preserve uniqueness while making sure the parameter-derived folder naming stays readable and predictable.
