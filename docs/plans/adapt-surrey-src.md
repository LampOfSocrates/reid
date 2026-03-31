# Adapt Surrey Coursework Src Into `reid`

## What This Feature Does

Adapt the reusable parts of `D:\S\Code\2026\vehicle-identification\Surrey_EEEM071_Coursework\src` into this `reid` repository so the local project gains a stronger training and evaluation structure without replacing its current custom model set.

The goal is to bring over the useful framework pieces, not to copy the source tree wholesale. The adapted result should preserve the current `reid` models (`bot`, `agw`, `transreid`, `pcb`, `clip_senet`) while improving:

- dataset handling for VeRi
- training/evaluation organization
- experiment artifact output
- reusable metrics and logging helpers
- notebook compatibility

## Files That Will Be Changed Or Created

- `data.py`
  Likely refactor to support cleaner train/query/gallery loading and better reuse
- `eval.py`
  Likely expand to support richer CMC/mAP style outputs similar to the Surrey metrics flow
- `train.py`
  Likely adapt to use cleaner helpers while keeping the existing model choices
- `plot_metrics.py`
  May be extended to support richer metrics emitted by the adapted training flow
- `notebook_runner.py`
  Likely updated to work with the adapted experiment outputs
- `experiment_utils.py`
  May be extended for config and artifact bookkeeping
- `tests/test_experiment_utils.py`
  May need updates if run metadata changes
- `tests/test_plot_metrics.py`
  May need updates if plot inputs expand
- `tests/test_notebook_runner.py`
  May need updates if orchestration changes
- `tests/` new files
  Additional unit tests for new dataset, metrics, and logging helpers
- New modules, likely:
  - `datasets/` for VeRi dataset abstractions
  - `utils/` for experiment logging / average meters / torch helpers
  - `metrics/` or expanded evaluation helpers
  - sampler or transform helpers if we adopt them

## Implementation Steps

1. Identify the minimal Surrey components worth adapting into `reid`:
   - data manager patterns
   - eval metric utilities
   - experiment logging/artifact structure
   - optional transforms and samplers
2. Write failing tests first for the specific behaviors we want to add, likely around:
   - VeRi dataset split loading
   - richer evaluation outputs
   - experiment artifact/logging behavior
   - compatibility with notebook-driven runs
3. Extract or recreate the Surrey-style reusable utilities in a form that fits this repo’s current structure and naming.
4. Refactor data loading so training and evaluation use clearer dataset abstractions instead of only raw folder loaders.
5. Refactor evaluation so it can compute and expose richer retrieval metrics while preserving current simple use cases.
6. Add experiment logging/output helpers inspired by the Surrey utilities, but shaped to this repo’s current workflow.
7. Update training orchestration to use the new helpers without replacing the existing custom models.
8. Update the Colab/notebook path if needed so it still runs the adapted flow cleanly.
9. Run the full relevant test suite and verify that the refactor preserves existing behavior where expected.

## Risks Or Open Questions

- “Adapt the code” could mean anything from selective helper reuse to a broad framework migration. My working assumption is selective adaptation of the reusable infrastructure, not a full source-tree replacement.
- The Surrey source uses relative package layout and abstractions that do not match this repo one-to-one, so some translation will be needed rather than direct copying.
- Some Surrey components assume multiple datasets and train samplers; we should only bring in what materially helps this VeRi-focused repo.
- A large migration could destabilize the current Colab notebook flow unless we keep compatibility in mind throughout.
- The source and destination repos may have different dependency expectations, so some imports may need rewriting or trimming.
