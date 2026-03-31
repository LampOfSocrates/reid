import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_metrics import plot_metrics


class PlotMetricsTests(unittest.TestCase):
    def test_plot_metrics_saves_png_to_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics.csv"
            output_path = Path(tmpdir) / "plots" / "loss.png"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Epoch": [1, 2], "Loss": [1.5, 1.2]}).to_csv(csv_path, index=False)

            result = plot_metrics(csv_path, output_path)

            self.assertEqual(Path(result), output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
