# Improve Coursework CLI Help Formatting

## What This Feature Does

Improve `coursework/args.py` so the command-line help output is easier to read, with each argument rendered more clearly on its own line when:

- invoking help explicitly, such as `python coursework/main.py --help`
- invoking `main.py` without required arguments, which triggers argparse usage/help output

The goal is to make the CLI output more legible without changing the meaning of the arguments.

## Files That Will Be Changed Or Created

- `coursework/args.py`
  Adjust argparse formatter behavior and, if needed, argument definitions/help formatting
- `tests/` new or updated `.py` tests
  Add tests for help text formatting behavior where practical
- `docs/plans/coursework-args-help-format.md`
  Feature plan document

## Implementation Steps

1. Inspect the current `argparse` configuration in `coursework/args.py` and identify the smallest change that improves per-argument readability.
2. Write failing tests first for the expected help-text formatting behavior if it can be verified reliably.
3. Update the parser formatter configuration and any related settings so the help output lists arguments more clearly, one per line.
4. Verify both `--help` output and the error/help path when required arguments are omitted.
5. Run tests and confirm the improved formatting.

## Risks Or Open Questions

- Argparse controls some formatting internally, so “each argument on a separate line” may require choosing the right formatter class and usage settings rather than fully custom rendering.
- The implementation should improve readability without breaking existing argument parsing behavior.
