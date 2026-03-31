import argparse
import unittest

from coursework.args import argument_parser, dataset_kwargs
from coursework.src.data_manager import subsample_records


class CourseworkDataFractionTests(unittest.TestCase):
    def test_dataset_kwargs_includes_data_fraction(self):
        parser = argument_parser()
        args = parser.parse_args(
            [
                "-s",
                "veri",
                "-t",
                "veri",
                "--data-fraction",
                "0.1",
            ]
        )

        kwargs = dataset_kwargs(args)

        self.assertEqual(kwargs["data_fraction"], 0.1)

    def test_subsample_records_keeps_fraction_for_eval_data(self):
        records = [("img1.jpg", 1, 1), ("img2.jpg", 2, 1), ("img3.jpg", 3, 1), ("img4.jpg", 4, 1), ("img5.jpg", 5, 1)]

        subset = subsample_records(records, data_fraction=0.4)

        self.assertEqual(len(subset), 2)

    def test_subsample_records_preserves_min_instances_per_identity_for_training(self):
        records = [
            ("a1.jpg", 1, 1),
            ("a2.jpg", 1, 2),
            ("a3.jpg", 1, 3),
            ("b1.jpg", 2, 1),
            ("b2.jpg", 2, 2),
            ("b3.jpg", 2, 3),
        ]

        subset = subsample_records(records, data_fraction=0.1, num_instances=2)

        pid_counts = {}
        for _, pid, _ in subset:
            pid_counts[pid] = pid_counts.get(pid, 0) + 1

        self.assertEqual(pid_counts[1], 2)
        self.assertEqual(pid_counts[2], 2)


if __name__ == "__main__":
    unittest.main()
