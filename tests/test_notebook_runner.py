import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from notebook_runner import run_experiment


class NotebookRunnerTests(unittest.TestCase):
    def test_run_experiment_creates_timestamped_run_dir_and_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"

            def fake_train(model_name, epochs, batch_size, data_dir, use_gpu=True):
                metrics_path = Path(f"metrics_{model_name}.csv")
                with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Epoch", "Loss"])
                    writer.writerow([1, 1.23])

            with patch("notebook_runner.train", side_effect=fake_train) as train_mock:
                result = run_experiment(
                    model_name="bot",
                    epochs=4,
                    batch_size=16,
                    data_dir="/tmp/VeRI/image_train",
                    output_root=output_root,
                    learning_rate=0.00035,
                    run_tag="baseline",
                    timestamp="20260331_101530",
                )

            train_mock.assert_called_once_with("bot", 4, 16, "/tmp/VeRI/image_train", use_gpu=True)
            self.assertEqual(
                result["run_dir"].name,
                "20260331_101530_bot_e4_bs16_lr0p00035_tagbaseline",
            )
            self.assertTrue(result["metrics_csv"].exists())
            self.assertTrue(result["plot_path"].exists())
            self.assertTrue(result["config_path"].exists())
            self.assertTrue(result["summary_path"].exists())


if __name__ == "__main__":
    unittest.main()
