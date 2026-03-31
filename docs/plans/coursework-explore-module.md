# Add `coursework/src/utils/explore.py`

## What This Feature Does

Add a new module at `coursework/src/utils/explore.py` that helps inspect the dataset visually.

The module should:

- resolve the query image set and gallery image set from the coursework dataset root
- read the good-match and junk-match index files for a selected query image
- render the actual query image together with its corresponding good and junk gallery images
- return structured metadata about the displayed files and counts

The implementation should adapt the user-provided reference code to the coursework repo layout and utility style.

## Files That Will Be Changed Or Created

- `coursework/src/utils/explore.py`
  New utility module for reading index files, resolving dataset folders, and rendering query/good/junk images
- `tests/` new or updated `.py` tests
  Focused tests for path resolution, index parsing, and result metadata
- `docs/plans/coursework-explore-module.md`
  Feature plan document

## Implementation Steps

1. Inspect the current `coursework/src/utils` package layout and existing visualization helpers to keep the new module stylistically consistent.
2. Write failing tests first for:
   - parsing one line of good/junk index files
   - resolving query/gallery image folders from the dataset root
   - returning the expected metadata structure for a selected query
3. Implement `coursework/src/utils/explore.py` with helpers adapted from the provided code, likely including:
   - index-line reader
   - query/gallery resolver
   - main rendering function for query, good, and junk images
4. Keep the module robust to one-based gallery indices and configurable display limits.
5. Run the tests and verify the new module works without changing unrelated files.
6. If requested after implementation, commit only the relevant `.py` files.

## Risks Or Open Questions

- The request says “under the coursework\\utils folder,” while the current imported utility package lives at `coursework/src/utils`; I am assuming `coursework/src/utils/explore.py` is the intended location.
- The visualization function will likely depend on `matplotlib` and `PIL`, so it should be implemented as a utility for notebook/local use rather than something required by every training run.
- Rendering behavior is hard to unit test directly, so tests should focus mainly on file/index resolution and returned metadata.
