# Improve Coursework Training Loop Progress Display

## What This Feature Does

Improve the training loop in `coursework/main.py` so training is easier to monitor by:

- showing a `tqdm` progress bar during training batches
- printing the time taken per epoch
- preserving the existing training metrics and logging behavior as much as possible

The goal is to make training progress more readable without changing the model or optimization behavior.

## Files That Will Be Changed Or Created

- `coursework/main.py`
  Update the training loop to use `tqdm` and add per-epoch timing output
- `docs/plans/coursework-training-progress.md`
  Feature plan document

## Implementation Steps

1. Inspect the current training loop in `coursework/main.py` and identify the cleanest place to wrap the dataloader with `tqdm`.
2. Add epoch timing around the training loop so elapsed time is measured and printed clearly for each epoch.
3. Update the batch-level display to work cleanly with `tqdm`, avoiding noisy duplicated console output.
4. Keep the existing summary metrics intact while improving readability of training progress.
5. Verify the updated script structure and output flow.

## Risks Or Open Questions

- The existing logger redirects stdout, so `tqdm` output may need careful configuration to render cleanly.
- If batch logging and progress bars both print aggressively, the console output could become noisy unless the two are coordinated.
- This should remain a display-only improvement and not affect training results.
