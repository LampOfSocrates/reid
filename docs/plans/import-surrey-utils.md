# Import Surrey `src/utils` Into `reid`

## What This Feature Does

Bring all code from `D:\S\Code\2026\vehicle-identification\Surrey_EEEM071_Coursework\src\utils` into this repository as a local `utils` package.

This includes:

- `avgmeter.py`
- `compare_viz.py`
- `experiment_logger.py`
- `generaltools.py`
- `iotools.py`
- `loggers.py`
- `mean_and_std.py`
- `metrics_viz.py`
- `torchtools.py`
- `visualtools.py`
- `__init__.py`

The goal is to copy the entire utility package into `reid` and make any minimal path/import adjustments needed so the files live coherently in this repo.

## Files That Will Be Changed Or Created

- `utils/__init__.py`
- `utils/avgmeter.py`
- `utils/compare_viz.py`
- `utils/experiment_logger.py`
- `utils/generaltools.py`
- `utils/iotools.py`
- `utils/loggers.py`
- `utils/mean_and_std.py`
- `utils/metrics_viz.py`
- `utils/torchtools.py`
- `utils/visualtools.py`
- `tests/` new or updated tests as needed for basic import/smoke coverage
- `docs/plans/import-surrey-utils.md`

## Implementation Steps

1. Inspect all files in the source `src/utils` package and identify internal imports or dependencies on modules that do not yet exist in `reid`.
2. Write failing tests first for basic package importability and any utility behavior we can verify cheaply.
3. Create a new `utils/` package in this repo and copy in all source files.
4. Adjust imports only where necessary so the imported utilities can coexist in this repo’s package layout.
5. Add or update lightweight tests to confirm the imported package is usable at least at the import/smoke-test level.
6. Run the relevant test suite and verify the copied utility package is present and importable.

## Risks Or Open Questions

- Some utility files may depend on dataset/model/training modules that do not exist yet in `reid`, so “bring all the code” may still require a few guarded imports or minor adaptation.
- A few files are large visualization/logging helpers and may not be exercised fully by tests without substantial fixtures.
- Copying the whole package may introduce utilities that are not yet used by this repo, but the request here is to bring them over rather than integrate them immediately.
