import unittest

import pandas as pd

from sismo_model import classify, train_and_evaluate


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_and_evaluate()

    def test_dataset_has_at_least_500_synthetic_rows(self):
        data = self.result["data"]
        self.assertGreaterEqual(len(data), 500)
        self.assertEqual(set(data["procedencia"]), {"SINTETICO_EDUCATIVO"})

    def test_event_groups_do_not_leak_into_test(self):
        self.assertFalse(
            set(self.result["train"]["evento_id"])
            & set(self.result["test"]["evento_id"])
        )
        self.assertEqual(
            self.result["test"]["evento_id"].nunique(),
            round(self.result["data"]["evento_id"].nunique() * 0.2 + 0.4),
        )

    def test_classifier_returns_supported_label_and_k_votes(self):
        estimate = classify(self.result, 5.5, 40, 35, 5)
        self.assertIn(estimate["prediccion"], {"I-IV", "V-VI", "VII-XII"})
        self.assertEqual(sum(estimate["votos"].values()), 5)

    def test_classifier_rejects_out_of_range_input(self):
        with self.assertRaises(ValueError):
            classify(self.result, 12, 40, 35, 5)

    def test_missing_required_column_fails_clearly(self):
        with self.assertRaises(ValueError):
            train_and_evaluate(pd.DataFrame({"evento_id": ["a"]}))


if __name__ == "__main__":
    unittest.main()
