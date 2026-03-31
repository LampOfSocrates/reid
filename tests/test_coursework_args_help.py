import io
import unittest
from contextlib import redirect_stderr

from coursework.args import argument_parser


class CourseworkArgsHelpTests(unittest.TestCase):
    def test_help_usage_lists_arguments_on_separate_lines(self):
        parser = argument_parser()

        help_text = parser.format_help()

        self.assertIn("usage:\n  ", help_text)
        self.assertIn("\n  [--root ROOT]", help_text)
        self.assertIn("\n  -s, --source-names SOURCE_NAMES [SOURCE_NAMES ...]", help_text)
        self.assertIn("\n  -t, --target-names TARGET_NAMES [TARGET_NAMES ...]", help_text)

    def test_parse_args_without_required_args_prints_full_help(self):
        parser = argument_parser()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit), redirect_stderr(stderr):
            parser.parse_args([])

        output = stderr.getvalue()
        self.assertIn("usage:\n  ", output)
        self.assertIn("options:\n", output)
        self.assertIn("--root ROOT", output)
        self.assertIn("--train-sampler TRAIN_SAMPLER", output)


if __name__ == "__main__":
    unittest.main()
