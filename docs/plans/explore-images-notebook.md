# Add `explore_images.ipynb`

## What This Feature Does

Create a notebook named `explore_images.ipynb` that uses `coursework/src/utils/explore.py` to inspect VeRi images and display the selected query image together with its good and junk matches.

The notebook should:

- detect whether it is running in Google Colab via an `is_colab()` helper
- use `/content/drive/MyDrive/Veri` when running in Colab
- use `G:\My Drive\VeRi` when running locally
- expose a simple configuration cell for choosing the query index and display limits
- call the new `show_veri_good_and_junk(...)` helper from `coursework/src/utils/explore.py`

## Files That Will Be Changed Or Created

- `explore_images.ipynb`
  New notebook for image exploration using the coursework utility
- `docs/plans/explore-images-notebook.md`
  Feature plan document

## Implementation Steps

1. Inspect the existing notebook and utility layout so the new notebook imports the coursework explorer cleanly.
2. Build `explore_images.ipynb` with cells for:
   - repo/path setup
   - `is_colab()` detection
   - environment-specific dataset root selection
   - configurable query/display settings
   - invocation of `show_veri_good_and_junk(...)`
3. Keep the notebook lightweight and notebook-friendly, with clear output for the chosen paths and query index.
4. Validate the notebook JSON after creation.

## Risks Or Open Questions

- The local path uses Windows-style `G:\My Drive\VeRi`, which is fine for a local Jupyter environment but not for Colab; the notebook will need to branch cleanly.
- If the notebook is run from a different working directory, it may need to insert the repo root into `sys.path` before importing coursework utilities.
- Notebook behavior is best validated structurally here; actual image display still depends on the dataset existing at the expected location.
