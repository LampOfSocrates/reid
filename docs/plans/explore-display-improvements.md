# Improve `show_veri_good_and_junk` Display Layout

## What This Feature Does

Improve the display produced by `coursework/src/utils/explore.py` so the visual output is easier to read when inspecting VeRi images.

The updated display should:

- increase the text size for image names, camera names, and related labels
- allow titles/captions to wrap across multiple lines if needed
- show cleaner section headlines for good and junk images
- keep the actual image-inspection workflow the same

## Files That Will Be Changed Or Created

- `coursework/src/utils/explore.py`
  Update the plotting layout, title formatting, and display headings
- `docs/plans/explore-display-improvements.md`
  Feature plan document

## Implementation Steps

1. Inspect the current `show_veri_good_and_junk` plotting layout and identify the simplest way to improve typography and headings.
2. Update subplot titles so they can display larger, cleaner text across multiple lines.
3. Add clear top-level headings for the good-match row and junk-match row.
4. Adjust spacing so larger text does not overlap the images unnecessarily.
5. Keep the function signature and returned metadata stable unless a small optional formatting parameter is clearly useful.

## Risks Or Open Questions

- Larger text may require more vertical spacing and therefore a slightly taller default figure.
- If filenames are very long, wrapping may still need truncation or manual line splitting for the cleanest result.
