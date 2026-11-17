"""Focused checks for leakage controls, sparse groups, and valid PD predictions."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from historical_pd import (
    PortfolioHistoricalPD,
    SegmentedHistoricalPD,
    binary_target,
    check_predictions,
    load_model,
    save_model,
)
from run_step_04 import evaluate, probability_groups, run


class HistoricalPDTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            "PAY_0": [0, 0, 0, 0, 2, 2, 4, 4],
            "LIMIT_BAL": [10, 20, 30, 40, 10, 20, 90, 100],
        })
        self.target = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    def test_portfolio_uses_only_training_outcomes_and_constant_scores(self):
        model = PortfolioHistoricalPD().fit(self.features, self.target)
        new_rows = pd.DataFrame(index=range(13))
        np.testing.assert_array_equal(model.predict(new_rows), np.full(13, 0.5))
        self.assertEqual(evaluate(self.target, model.predict(self.features))["roc_auc"], 0.5)
        self.assertEqual(model.training_defaults_, 4)

    def test_status_smoothing_matches_hand_calculation(self):
        model = SegmentedHistoricalPD(smoothing=4).fit(self.features, self.target)
        self.assertAlmostEqual(model.predict(self.features.iloc[[0]])[0], (1 + 4 * 0.5) / (4 + 4))
        self.assertAlmostEqual(model.predict(self.features.iloc[[4]])[0], (2 + 4 * 0.5) / (2 + 4))

    def test_joint_smoothing_uses_smoothed_parent(self):
        model = SegmentedHistoricalPD(True, n_limit_bands=2, smoothing=4).fit(self.features, self.target)
        parent = (1 + 4 * 0.5) / (4 + 4)
        # Status 0, low-limit segment: two observations, zero defaults.
        self.assertAlmostEqual(model.predict(self.features.iloc[[0]])[0], (0 + 4 * parent) / (2 + 4))

    def test_sparse_group_shrinks_away_from_zero_or_one(self):
        model = SegmentedHistoricalPD(True, n_limit_bands=2, smoothing=10).fit(
            self.features.iloc[:7], self.target[:7]
        )
        rare = model.segment_table_.loc[model.segment_table_["PAY_0"] == 4].iloc[0]
        self.assertEqual(rare.customers, 1)
        self.assertEqual(rare.raw_pd, 1)
        self.assertGreater(rare.estimated_pd, model.portfolio_pd_)
        self.assertLess(rare.estimated_pd, 1)

    def test_unseen_joint_status_and_missing_values_have_hierarchical_fallbacks(self):
        model = SegmentedHistoricalPD(True, n_limit_bands=2, smoothing=4).fit(self.features, self.target)
        unseen = pd.DataFrame({
            "PAY_0": [2, 99, np.nan, 2, 2, 2],
            "LIMIT_BAL": [100, 100, 10, np.nan, -5, np.inf],
        }, index=[8, 6, 4, 2, 1, 0])
        details = model.prediction_details(unseen)
        np.testing.assert_allclose(details.predicted_pd, [2 / 3, 0.5, 0.5, 2 / 3, 2 / 3, 2 / 3])
        self.assertEqual(details.prediction_source.tolist(), ["PAY_0", "portfolio", "portfolio", "PAY_0", "PAY_0", "PAY_0"])
        self.assertEqual(details.index.tolist(), unseen.index.tolist())

    def test_boundaries_are_frozen_and_extremes_enter_outer_bands(self):
        model = SegmentedHistoricalPD(True, n_limit_bands=2).fit(self.features, self.target)
        original = model.credit_limit_cutoffs_.copy()
        np.testing.assert_array_equal(original, [25.0])
        new_rows = pd.DataFrame({"PAY_0": [0, 0, 0, 0], "LIMIT_BAL": [1, 25, 25.001, 1e9]})
        self.assertEqual(model.prediction_details(new_rows).limit_band.tolist(), [0, 0, 1, 1])
        np.testing.assert_array_equal(original, model.credit_limit_cutoffs_)

    def test_tied_training_limits_collapse_bands(self):
        features = self.features.assign(LIMIT_BAL=100)
        model = SegmentedHistoricalPD(True).fit(features, self.target)
        self.assertEqual(len(model.credit_limit_cutoffs_), 0)
        self.assertEqual(model.prediction_details(features).limit_band.unique().tolist(), [0])
        check_predictions(model.predict(features), len(features))

    def test_extra_target_and_demographic_columns_cannot_change_predictions(self):
        model = SegmentedHistoricalPD(True).fit(self.features, self.target)
        altered = self.features.assign(default_next_month=1 - self.target, SEX=2, AGE=99)
        np.testing.assert_array_equal(model.predict(self.features), model.predict(altered))
        other = SegmentedHistoricalPD(True).fit(altered, self.target)
        np.testing.assert_array_equal(model.predict(self.features), other.predict(self.features))

    def test_predictions_valid_on_training_and_new_customers(self):
        for model in [PortfolioHistoricalPD(), SegmentedHistoricalPD(), SegmentedHistoricalPD(True)]:
            model.fit(self.features, self.target)
            predictions = model.predict(self.features)
            self.assertEqual(len(predictions), len(self.features))
            self.assertTrue(np.isfinite(predictions).all())
            self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())
            self.assertEqual(len(model.predict(self.features.iloc[:0])), 0)

    def test_json_round_trip_preserves_predictions_and_fallbacks(self):
        new_rows = pd.DataFrame({"PAY_0": [0, 2, 99], "LIMIT_BAL": [10, 100, 100]})
        with tempfile.TemporaryDirectory() as directory:
            for model in [PortfolioHistoricalPD(), SegmentedHistoricalPD(), SegmentedHistoricalPD(True)]:
                model.fit(self.features, self.target)
                path = Path(directory) / "model.json"
                save_model(model, path)
                np.testing.assert_array_equal(model.predict(new_rows), load_model(path).predict(new_rows))

    def test_prediction_validation_rejects_missing_out_of_range_and_wrong_length(self):
        for predictions in [[np.nan], [np.inf], [-0.01], [1.01], [0.1, 0.2]]:
            with self.assertRaises(ValueError):
                check_predictions(predictions, 1)

    def test_invalid_targets_and_inputs_are_rejected(self):
        for target in [[], [0, np.nan], [0, 2], [[0], [1]]]:
            with self.assertRaises(ValueError):
                binary_target(target)
        with self.assertRaises(ValueError):
            PortfolioHistoricalPD().fit(self.features, [0])
        with self.assertRaises(ValueError):
            SegmentedHistoricalPD(smoothing=0)
        with self.assertRaises(ValueError):
            SegmentedHistoricalPD(n_limit_bands=1)
        with self.assertRaises(ValueError):
            SegmentedHistoricalPD(True).fit(self.features.assign(LIMIT_BAL=np.nan), self.target)
        model = SegmentedHistoricalPD(True).fit(self.features, self.target)
        for invalid in [self.features.drop(columns="PAY_0"), self.features.assign(PAY_0=1.5)]:
            with self.assertRaises(ValueError):
                model.predict(invalid)
        with self.assertRaises(ValueError):
            PortfolioHistoricalPD().predict(self.features)

    def test_one_class_evaluation_has_no_auc_and_finite_losses(self):
        result = evaluate([0, 0], [0, 0])
        self.assertTrue(np.isnan(result["roc_auc"]))
        self.assertEqual(result["brier_score"], 0)
        self.assertTrue(np.isfinite(result["log_loss"]))

    def test_fixed_probability_bands_include_zero_one_and_all_rows(self):
        groups = probability_groups([0, 1, 1], [0, 0.5, 1], [0, 0.5, 1])
        self.assertEqual(groups.customers.sum(), 3)
        self.assertEqual(groups.defaults.sum(), 2)
        self.assertEqual(groups.customers.tolist(), [2, 1])

    def test_workflow_selection_is_independent_of_test_outcomes(self):
        # An end-to-end check: flip every test label, while providing no test
        # predictors at all. Fitted models and validation results must be identical.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "processed"
            data_dir.mkdir()
            self.features.assign(default_next_month=self.target).to_csv(data_dir / "train.csv", index=False)
            self.features.assign(default_next_month=1 - self.target).to_csv(data_dir / "validation.csv", index=False)
            test_path = data_dir / "test.csv"
            pd.DataFrame({"default_next_month": [0, 0, 0]}).to_csv(test_path, index=False)
            first_inputs = {path.name: path.read_bytes() for path in data_dir.glob("*.csv")}
            first = run(data_dir, root / "first")
            self.assertEqual(first_inputs, {path.name: path.read_bytes() for path in data_dir.glob("*.csv")})
            pd.DataFrame({"default_next_month": [1, 1, 1]}).to_csv(test_path, index=False)
            second = run(data_dir, root / "second")
            pd.testing.assert_frame_equal(first, second)
            for path in (root / "first" / "models").glob("*.json"):
                self.assertEqual(path.read_bytes(), (root / "second" / "models" / path.name).read_bytes())
            self.assertEqual(len(list((root / "first" / "figures").glob("*.png"))), 4)
            self.assertTrue((root / "first" / "step_04_historical_pd.md").is_file())
            self.assertFalse(any("test" in path.name and "portfolio" not in path.name
                                 for path in (root / "first" / "tables").glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
