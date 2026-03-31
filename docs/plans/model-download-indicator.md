# Model Download Indicator

## What This Feature Does

Adds a visible signal in `coursework/main.py` when a model may be downloading pretrained weights, so the training script does not appear stuck during model initialization.

## Files To Change Or Create

- `coursework/main.py`
  Add the user-facing download indication around model initialization.
- `docs/plans/model-download-indicator.md`
  This plan document.

## Implementation Steps

1. Inspect the current model initialization flow in `coursework/main.py` to identify the point where pretrained weights may be fetched.
2. Add a clear console message before the potentially slow download step so the user sees that pretrained model weights may be downloading.
3. Add a completion message after model initialization finishes so the user can tell download/loading is done.
4. Keep the change lightweight and avoid altering model-loading behavior itself.

## Risks Or Open Questions

- `timm` and related libraries do not always expose download progress directly, so the practical solution may be a textual “downloading/loading pretrained weights” status message rather than a true progress bar.
- The message should avoid implying a download when `--no-pretrained` is used.
