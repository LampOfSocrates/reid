# Improve Explorer Layout With Under-Image Metadata

## What This Feature Does

Update the VeRi explorer display so images are shown as large as possible within a grid capped at 4 columns, while the filename, camera, and related metadata are rendered as text underneath each image instead of in the image title area.

This should improve readability by:

- keeping image panels visually larger
- reducing title clutter above images
- placing metadata in a cleaner text block below each image

## Files That Will Be Changed Or Created

- `coursework/src/utils/explore.py`
  Refactor plotting so each image cell uses the title area less and renders metadata underneath the image
- `explore_images.ipynb`
  Update notebook defaults or explanatory text if needed to match the new presentation
- `docs/plans/explore-under-image-metadata.md`
  Feature plan document

## Implementation Steps

1. Refactor the explorer grid so each displayed item has a larger image area.
2. Move filename, camera, and related metadata from the subplot title into text rendered below each image.
3. Keep the grid capped at 4 columns and allow as many rows as needed.
4. Adjust spacing and figure sizing so the under-image text remains readable without shrinking the images too much.
5. Update the notebook text/config if needed so it reflects the improved layout.

## Risks Or Open Questions

- Putting metadata below each image requires more vertical space per cell, so the figure height may need to grow dynamically.
- Long filenames may still need wrapping to avoid overflowing the cell width.
