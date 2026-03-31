# Improve Explorer Grid Layout For Query, Good, and Bad Images

## What This Feature Does

Improve both:

- `coursework/src/utils/explore.py`
- `explore_images.ipynb`

so the query image and its associated good and bad images are displayed clearly in a grid layout with:

- a maximum of 4 columns
- as many rows as needed
- clearer visual grouping for query, good, and bad images

The goal is to make inspection easier when there are many good or bad images, instead of relying on a fixed two-row display that becomes cramped.

## Files That Will Be Changed Or Created

- `coursework/src/utils/explore.py`
  Refactor the plotting logic to support a capped-column grid layout and clearer grouping
- `explore_images.ipynb`
  Update the notebook usage/configuration so it works cleanly with the improved display
- `docs/plans/explore-grid-layout.md`
  Feature plan document

## Implementation Steps

1. Inspect the current `show_veri_good_and_junk` layout and identify how to restructure it into a flexible grid.
2. Update `explore.py` so:
   - the query image is shown clearly
   - good images are displayed in a grid section with at most 4 columns
   - bad/junk images are displayed in a separate grid section with at most 4 columns
   - the number of rows expands as needed
3. Keep captions readable with larger text and wrapped labels where necessary.
4. Adjust figure sizing dynamically so the grid remains readable for larger result sets.
5. Update `explore_images.ipynb` so its defaults and explanatory text match the new layout behavior.
6. Verify the notebook JSON remains valid after the update.

## Risks Or Open Questions

- A single figure with multiple sections can get tall when there are many good and bad images, so figure sizing will likely need to scale with row count.
- We should decide whether the query image should appear once at the top or be repeated per section; my default recommendation is to show it once in its own section.
