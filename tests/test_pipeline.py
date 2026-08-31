import unittest

import pandas as pd

from stda_classification.pipeline import (
    classify_patients_supervised,
    cluster_patients_unsupervised,
    quantify_recurrence_by_group,
)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "fractionation_index": [0.2, 0.3, 0.25, 0.9, 1.0, 0.95, 0.4, 0.85],
                "cycle_length_ms": [140, 142, 138, 95, 98, 92, 130, 100],
                "signal_entropy": [0.5, 0.55, 0.52, 0.9, 0.92, 0.88, 0.6, 0.84],
                "recurrence": [0, 0, 0, 1, 1, 1, 0, 1],
            }
        )
        self.features = ["fractionation_index", "cycle_length_ms", "signal_entropy"]

    def test_supervised_metrics_include_accuracy(self):
        metrics = classify_patients_supervised(self.df, self.features)

        self.assertIn("accuracy", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_unsupervised_groups_and_recurrence_summary(self):
        grouped = cluster_patients_unsupervised(self.df, self.features, n_clusters=2)
        self.assertIn("patient_group", grouped.columns)

        summary = quantify_recurrence_by_group(grouped)
        self.assertIn("recurrence_rate", summary.columns)
        self.assertEqual(summary["total_patients"].sum(), len(self.df))


if __name__ == "__main__":
    unittest.main()
