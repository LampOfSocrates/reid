# Google Colab Notebook For VeRi Experiments

## What This Feature Does

Create a Jupyter notebook that runs this ReID project in Google Colab with Google Drive already mounted. The notebook will:

- Assume the VeRi dataset lives under `G:\My Drive\Colab Notebooks\data\VeRI` in the mounted Drive context described by the user
- Provide a single configuration cell for selecting the experiment to run, including model, training hyperparameters, output root, and optional evaluation settings
- Run the existing training flow for supported models (`bot`, `agw`, `transreid`, `pcb`, `clip_senet`)
- Save each run into an output directory whose name starts with a timestamp and includes the model name plus key parameters
- Write run artifacts such as metrics CSVs, plots, checkpoints, and notebook-friendly summaries into that run directory
- Be structured so a user can re-run a different experiment by editing only the configuration cell

## Files That Will Be Changed Or Created

- `colab_reid.ipynb`
  Google Colab notebook with setup, configuration, training, plotting, and optional evaluation cells
- `train.py`
  Likely needs small changes so training can accept an output directory, save artifacts there, and optionally emit checkpoints in a notebook-friendly way
- `plot_metrics.py`
  May need a small update so plots can be saved to an explicit destination reliably from both scripts and notebooks
- `eval.py`
  May need small adjustments if we wire notebook-driven evaluation into the same run directory structure
- `docs/plans/colab-notebook.md`
  Feature plan document

## Implementation Steps

1. Review the current training, plotting, and evaluation entry points and identify the smallest changes needed to support notebook-driven runs.
2. Add failing tests first for any extracted helpers or behavior changes, especially:
   - run directory naming
   - explicit artifact output paths
   - notebook/config-driven experiment parameter handling where practical
3. Refactor training support code as needed so notebook cells can call reusable Python functions instead of shelling out in fragile ways.
4. Update training to save outputs into a caller-provided run directory named like:
   `YYYYMMDD_HHMMSS_<model>_e<epochs>_bs<batch_size>...`
5. Update plotting and any evaluation helpers so they write into that same run directory.
6. Create `colab_reid.ipynb` with clear cells for:
   - environment/setup imports
   - configuration
   - path validation for the Drive-mounted VeRi dataset
   - optional dependency installation guidance if required
   - experiment execution
   - results display and artifact links
7. Verify the notebook logic locally as far as possible without an actual Colab runtime, and confirm the produced paths and naming scheme.

## Risks Or Open Questions

- The user-specified dataset path uses Windows-style `G:\...`, while Google Colab normally exposes Drive under `/content/drive/MyDrive/...`. We should confirm whether the notebook should literally target the provided `G:\...` path convention or translate it to standard Colab paths.
- Current code does not appear to save model checkpoints, so checkpoint output may require an explicit new behavior.
- Some models may have extra runtime or dependency requirements in Colab that are not yet captured in the repository.
- The current training script shells out to `plot_metrics.py`; we may want to replace that with direct function calls for notebook reliability.
- Existing repo has minimal tests, so some notebook-oriented behavior may be best covered with small helper tests rather than full notebook execution tests.
