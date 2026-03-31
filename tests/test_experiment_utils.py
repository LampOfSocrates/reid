import re
import tempfile
import unittest
from pathlib import Path

from experiment_utils import create_run_dir, make_run_slug


class ExperimentUtilsTests(unittest.TestCase):
    def test_make_run_slug_includes_timestamp_model_and_params(self):
        slug = make_run_slug(
            model_name="transreid",
            epochs=24,
            batch_size=32,
            extra_params={"lr": 0.00035, "tag": "baseline"},
            timestamp="20260331_101530",
        )

        self.assertEqual(
            slug,
            "20260331_101530_transreid_e24_bs32_lr0p00035_tagbaseline",
        )

    def test_make_run_slug_sanitizes_extra_param_values(self):
        slug = make_run_slug(
            model_name="clip_senet",
            epochs=12,
            batch_size=16,
            extra_params={"notes": "aug v2/test"},
            timestamp="20260331_101530",
        )

        self.assertRegex(
            slug,
            r"^20260331_101530_clip_senet_e12_bs16_notesaug-v2-test$",
        )

    def test_create_run_dir_creates_timestamped_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = create_run_dir(
                output_root=tmpdir,
                model_name="agw",
                epochs=8,
                batch_size=64,
                extra_params={"lr": 0.00035},
                timestamp="20260331_101530",
            )

            self.assertTrue(run_dir.exists())
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(
                run_dir.name,
                "20260331_101530_agw_e8_bs64_lr0p00035",
            )
            self.assertEqual(run_dir.parent, Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
