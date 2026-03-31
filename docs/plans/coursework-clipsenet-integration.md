# Integrate `CLIPSENet` Into Coursework Models

## What This Feature Does

Integrate the `CLIPSENet` model from the main `reid` codebase into the coursework model registry so coursework experiments can be run with:

```bash
-a clip_senet
```

The integration should preserve both:

- the model logic
- the detailed line-by-line explanatory comments from `models/clip_senet.py`

The goal is to make `CLIPSENet` behave like a first-class coursework model while fitting the coursework training/evaluation interface.

## Files That Will Be Changed Or Created

- `coursework/src/models/__init__.py`
  Register `clip_senet` in the coursework model factory
- `coursework/src/models/clip_senet.py`
  New coursework-local copy/adaptation of `CLIPSENet`, including its detailed comments
- Possibly `coursework/src/models/` helper code
  If small shared init helpers are needed locally instead of importing from `reid/models/bot.py`
- Possibly `coursework/src/transforms.py`
  Only if we decide to add CLIP-specific normalization or model-aware preprocessing
- `docs/plans/coursework-clipsenet-integration.md`
  Feature plan document

## Implementation Steps

1. Copy the `CLIPSENet` model logic from `models/clip_senet.py` into a new coursework-local module:
   - preserve the detailed explanatory comments
   - remove the dependency on `reid/models/bot.py`
2. Bring over or recreate the required weight-init helper functions inside the coursework model package.
3. Wrap the coursework-local `clip_senet` entry point so it matches the coursework registry expectations:
   - accepts `num_classes`
   - accepts `loss={"xent"}` or `loss={"xent", "htri"}`
   - accepts `pretrained=True/False` if practical
4. Register the model in `coursework/src/models/__init__.py` so `-a clip_senet` works.
5. Review compatibility with coursework transforms:
   - minimum path: keep current transforms and rely on CLIPSENet’s internal resize to `224x224`
   - optional improvement: add CLIP-friendly normalization if needed
6. Verify that the coursework training loop interface matches the new model:
   - training returns `logits, features`
   - evaluation returns `features`
7. Confirm any dependency note needed for `timm`.

## Risks Or Open Questions

- `CLIPSENet` depends on `timm`, so the coursework runtime must have that installed.
- The coursework transform pipeline currently uses ImageNet normalization, which may work but may not be optimal for a CLIP-based backbone.
- The model comments are intentionally detailed; preserving them will make the copied file longer, but that is desired here.
