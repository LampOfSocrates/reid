# Add Crop And Blur Augmentation Flags

## What This Feature Does

Add two new command-line boolean flags:

- `--crop-aug`
- `--blur-aug`

When enabled, these flags will extend the training augmentation pipeline built in `coursework/src/transforms.py` so the training transform sequence can optionally include crop augmentation and blur augmentation.

The existing flow already passes augmentation-related options from `coursework/args.py` through `dataset_kwargs(...)` into `coursework/src/data_manager.py`, which then calls `build_transforms(...)`. This feature will follow that same pattern for the two new booleans.

## Files That Will Be Changed Or Created

- `docs/plans/crop-blur-augmentations.md`
- `coursework/args.py`
- `coursework/src/data_manager.py`
- `coursework/src/transforms.py`

## Implementation Steps

1. Add `--crop-aug` and `--blur-aug` to `coursework/args.py` as boolean augmentation flags, aligned with the existing `--random-erase`, `--color-jitter`, and `--color-aug` options.
2. Update `dataset_kwargs(...)` in `coursework/args.py` so the parsed values are forwarded into the data manager configuration.
3. Extend `BaseDataManager` in `coursework/src/data_manager.py` to accept and store the two new augmentation booleans, then pass them into `build_transforms(...)`.
4. Update `build_transforms(...)` in `coursework/src/transforms.py` to accept the new keyword arguments and conditionally append the crop and blur augmentations to `transform_train`.
5. Keep the test transform path unchanged so the new options only affect training-time preprocessing.
6. Verify the CLI-to-transform wiring by checking that the new args flow from parsing through to transform composition, without adding or running tests.

## Risks Or Open Questions

- The request says to “add these transformations,” but it does not specify the exact torchvision transforms or parameters to use for crop and blur. I am assuming this should use standard torchvision choices, likely a crop-based augmentation and `GaussianBlur`, but the exact transform class and strength should be confirmed before implementation.
- No tests should be added or run for this change, per your instruction and the local project workflow.
