# New Session Handoff

## What This Project Is

This repository is a vehicle re-identification project centered on VeRi-style data and training workflows.

It currently contains two main strands of work:

- the original `reid` codebase with custom models and Colab-oriented notebooks
- an imported coursework codebase under `coursework/`, based on `Surrey_EEEM071_Coursework`, which is being adapted and extended inside this repo

Key areas in the repo:

- `coursework/main.py`
  Main coursework training/evaluation entry point
- `coursework/args.py`
  Coursework CLI argument definitions and help formatting
- `coursework/src/`
  Coursework dataset, model, loss, sampler, and utility modules
- `coursework/src/utils/explore.py`
  Utility to inspect a query image together with good and junk gallery matches
- `explore_images.ipynb`
  Notebook for image exploration with Colab/local path detection
- `colab_reid.ipynb`
  Notebook for running the non-coursework ReID flow in Colab

## Current Working Conventions

- Do not create or run tests unless the user explicitly asks for it.
- Use `NewSession.md` as a quick way to recover project context in future sessions.

## Most Recent Changes

1. Added coursework image exploration tools.
   - Added `coursework/src/utils/explore.py`
   - Exported the explorer from `coursework/src/utils/__init__.py`
   - Added `explore_images.ipynb` for Colab/local dataset browsing

2. Added `data_fraction` support to the coursework pipeline.
   - Added `--data-fraction` in `coursework/args.py`
   - Applied train/query/gallery subsampling in `coursework/src/data_manager.py`

3. Improved coursework CLI help formatting.
   - `coursework/args.py` now prints a multiline usage block with one argument entry per line
   - Running `coursework/main.py` without required args now prints full help before the error

4. Improved the coursework training loop display.
   - `coursework/main.py` now uses `tqdm`
   - It also prints elapsed time per epoch

5. Imported the Surrey coursework repository into this repo.
   - It lives under `coursework/`
   - The nested upstream git metadata was removed so it is tracked as part of this repo

## Useful Paths

- Repo root: `d:\S\Code\2026\reid`
- Coursework entry point: `d:\S\Code\2026\reid\coursework\main.py`
- Coursework utils: `d:\S\Code\2026\reid\coursework\src\utils`
- Local VeRi path often assumed in notebooks: `G:\My Drive\VeRi`
- Colab VeRi path often assumed in notebooks: `/content/drive/MyDrive/Veri`
