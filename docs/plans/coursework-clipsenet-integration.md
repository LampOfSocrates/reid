# Integrate `CLIPSENet` Into Coursework Models

## What This Feature Does

Integrate the `CLIPSENet` model from the main `reid` codebase into the coursework model registry so coursework experiments can be run with:

```bash
-a clip_senet
```

This integration should preserve both:

- the model logic
- the detailed line-by-line explanatory comments from `models/clip_senet.py`

The intent is to make `CLIPSENet` a first-class coursework model without changing the existing coursework training loop contract unless that becomes strictly necessary.

## Compatibility Assessment

### Coursework Training Interface

The current coursework training and evaluation flow is already structurally compatible with `CLIPSENet`.

`coursework/main.py` expects:

- during training: `outputs, features = model(imgs)`
- during evaluation: `features = model(imgs)`

`CLIPSENet` already behaves that way:

- in training mode it returns `logits, features`
- in evaluation mode it returns `features`

So the model interface itself is a good fit.

### Coursework Data Structure

The current coursework data structure should work with `CLIPSENet`.

Reasons:

- `ImageDataManager` yields image tensors, IDs, camera IDs, and paths
- `CLIPSENet` only needs the image tensor input
- extra batch fields do not interfere with it
- `CLIPSENet` internally resizes inputs to `224x224`, so it can tolerate larger incoming coursework images

Conclusion:

- the coursework batch/data structure is compatible enough to run `CLIPSENet`
- no data-structure redesign is needed for a first-pass integration

### Transform Compatibility

The current coursework transforms are operationally compatible, but not necessarily optimal.

- coursework currently uses ImageNet normalization in `coursework/src/transforms.py`
- `CLIPSENet` uses a CLIP-based backbone from `timm`

This means:

- the model should run with the current transforms
- but the model may benefit later from CLIP-specific normalization

Recommended first pass:

- do not change transforms initially
- integrate the model first
- treat CLIP-specific preprocessing as a later follow-up if needed

## Recommended Integration Strategy

Use a coursework-local copy of `CLIPSENet` instead of importing `reid/models/clip_senet.py` directly at runtime.

This is the cleanest approach because:

- the coursework package stays self-contained
- `models/clip_senet.py` currently depends on `reid/models/bot.py`
- coursework already has its own model registry conventions
- it avoids hidden cross-package coupling between `coursework/src/models` and `reid/models`

## Files That Will Be Changed Or Created

- `coursework/src/models/__init__.py`
  Import and register `clip_senet` in the coursework model factory
- `coursework/src/models/clip_senet.py`
  New coursework-local copy/adaptation of `CLIPSENet`, including its detailed comments
- `coursework/src/models/clip_senet.py` internal helper code
  Include the required weight-init helper functions locally unless a tiny shared coursework helper is clearly cleaner
- Possibly `coursework/src/transforms.py`
  Only if CLIP-specific normalization is explicitly included in this same task
- `docs/plans/coursework-clipsenet-integration.md`
  Feature plan document

## Implementation Steps

1. Create `coursework/src/models/clip_senet.py` by copying the logic from `models/clip_senet.py`.
   Preserve the detailed explanatory comments as part of the copy, not just the executable logic.
2. Remove the dependency on `reid/models/bot.py` by bringing the needed helper functions into the coursework-local model file.
   Expected helpers:
   - `weights_init_kaiming`
   - `weights_init_classifier`
3. Add a coursework-style factory function in `coursework/src/models/clip_senet.py`, for example:
   - `def clip_senet(num_classes, loss={"xent"}, pretrained=True, **kwargs): ...`
4. Ensure the coursework-local factory matches the registry contract used by `coursework/main.py`:
   - accepts `num_classes`
   - accepts `loss`
   - accepts `pretrained`
   - returns a model that behaves correctly in both training and evaluation mode
5. Register the model in `coursework/src/models/__init__.py` by:
   - importing `clip_senet`
   - adding `"clip_senet": clip_senet` to `__model_factory`
6. Keep `coursework/main.py` unchanged unless a real interface mismatch is discovered during implementation.
7. Keep `coursework/src/transforms.py` unchanged for the first pass unless CLIP-specific normalization is deliberately included now.
8. Document that `timm` is required in the coursework runtime environment.

## Risks Or Open Questions

- `CLIPSENet` depends on `timm`, so the coursework runtime must have that installed.
- The coursework transform pipeline currently uses ImageNet normalization, which should run but may not be optimal for a CLIP-based backbone.
- The coursework registry expects coursework-style model constructors, so `CLIPSENet` should be wrapped cleanly rather than dropped in raw.
- The model comments are intentionally detailed; preserving them will make the copied file longer, but that is desired here.
- If CLIP-specific preprocessing is wanted later, that should ideally be a separate, clearly scoped follow-up task.
