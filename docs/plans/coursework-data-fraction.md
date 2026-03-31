# Add `data_fraction` Support To Coursework Runner

## What This Feature Does

Add a `data_fraction` parameter to the imported coursework code so trial runs can use a fixed fraction of the dataset, such as `0.1` for 10%.

The behavior should be consistent across the coursework pipeline:

- expose a CLI flag like `--data-fraction`
- pass it through the existing argument-to-dataset plumbing
- apply it in the data manager to training and evaluation splits
- keep `RandomIdentitySampler` usable by ensuring each retained identity still has enough samples for the sampler configuration

Per the user request, only `.py` files should be committed when the work is checked in.

## Files That Will Be Changed Or Created

- `coursework/args.py`
  Add the CLI argument and ensure it is included in dataset kwargs
- `coursework/src/data_manager.py`
  Apply subsampling consistently for train/query/gallery construction
- `tests/` new or updated `.py` tests
  Add focused tests for argument plumbing and subsampling behavior
- `docs/plans/coursework-data-fraction.md`
  Feature plan document

## Implementation Steps

1. Inspect `coursework/args.py` for the existing `dataset_kwargs(args)` helper and identify the cleanest place to add `data_fraction`.
2. Write failing tests first for:
   - argument plumbing into dataset kwargs
   - train split subsampling behavior
   - query/gallery subsampling behavior
   - maintaining sampler viability for small fractions
3. Add `--data-fraction` to the coursework CLI with a default of `1.0`.
4. Update dataset kwargs plumbing so the parsed value reaches `ImageDataManager`.
5. Update `coursework/src/data_manager.py` to:
   - clamp the fraction into a safe range
   - subsample training data in a way that preserves enough instances for `RandomIdentitySampler`
   - subsample query/gallery for faster eval
   - print concise summary information showing the effective fraction
6. Run tests and verify the feature works without changing non-Python files.
7. Commit only the relevant `.py` files, as requested.

## Risks Or Open Questions

- Very small fractions may still conflict with `RandomIdentitySampler` if some identities have too few images; the implementation should preserve at least `num_instances` images where possible.
- Since the coursework code currently has no dedicated test suite in this area, test coverage will likely use small synthetic datasets and lightweight helper tests.
- If there are uncommitted non-Python changes in the repo at commit time, they should remain uncommitted to respect the “commit only `.py` files” requirement.
