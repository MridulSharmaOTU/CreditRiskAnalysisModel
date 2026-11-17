"""Meaningful Step 5 checks on synthetic rows; production test labels are unused."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate

from historical_pd import TARGET
from logistic_pd import (
    BILL_FEATURES, EXCLUDED_DEMOGRAPHICS, FinancialFeatures, NUMERIC_FEATURES,
    PAYMENT_FEATURES, RAW_FEATURES, STATUS_FEATURES, coefficient_table,
    fit_with_convergence_check, make_pipeline, predict_pd, underlying_pipeline,
)
from run_step_04 import file_hash
from run_step_05 import evaluate_frozen_test, make_prediction_frame, verify_split_inputs
from step_05_reporting import decile_table, probability_metrics


class LogisticPDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.features = pd.DataFrame(index=np.arange(1000, 1240))
        cls.features["LIMIT_BAL"] = rng.integers(20_000, 200_000, 240)
        for column in BILL_FEATURES:
            cls.features[column] = rng.integers(-500, 60_000, 240)
        for column in PAYMENT_FEATURES:
            cls.features[column] = rng.integers(0, 10_000, 240)
        for column in STATUS_FEATURES:
            cls.features[column] = rng.choice([-2, -1, 0, 1, 2], 240)
        cls.features.loc[1000, "PAY_0"] = 8  # Rare training category.
        cls.features.loc[1001, "BILL_AMT2"] = 0
        cls.target = ((cls.features.PAY_0 > 0).to_numpy() | (rng.random(240) < 0.1)).astype(int)
        cls.model = fit_with_convergence_check(
            make_pipeline(engineered=True, solver="liblinear"), cls.features, cls.target
        )

    def test_valid_probabilities_and_exact_customer_alignment(self):
        full = predict_pd(self.model, self.features)
        positions = [20, 3, 20, 99]
        customers = self.features.iloc[positions]
        predictions = predict_pd(self.model, customers)
        self.assertEqual(predictions.index.tolist(), customers.index.tolist())
        np.testing.assert_array_equal(predictions, full.iloc[positions])
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(predictions.between(0, 1).all())
        self.assertEqual(len(predict_pd(self.model, self.features.iloc[:0])), 0)

    def test_target_and_demographic_columns_cannot_affect_pd(self):
        changed = self.features.assign(**{TARGET: 1 - self.target, "SEX": 99,
                                          "AGE": 999, "MARRIAGE": -1, "EDUCATION": 77})
        np.testing.assert_array_equal(predict_pd(self.model, self.features), predict_pd(self.model, changed))
        output = FinancialFeatures(True).fit_transform(changed, self.target)
        self.assertTrue(set(EXCLUDED_DEMOGRAPHICS + [TARGET]).isdisjoint(output.columns))
        self.assertEqual(set(RAW_FEATURES), set(FinancialFeatures().fit_transform(changed).columns))

    def test_zero_and_negative_bills_are_retained_without_invalid_ratios(self):
        customers = self.features.iloc[:2].copy()
        customers.loc[:, "BILL_AMT1"] = [-100, 100]
        customers.loc[:, "BILL_AMT2"] = [0, -200]
        customers.loc[:, "PAY_AMT1"] = 500
        transformed = FinancialFeatures(True).fit_transform(customers)
        self.assertEqual(transformed.BILL_AMT1.iloc[0], -100)
        self.assertEqual(transformed.payment_to_previous_bill.tolist(), [0, 0])
        self.assertEqual(transformed.no_positive_previous_bill.tolist(), [1, 1])
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertLess(transformed.utilization_recent.iloc[0], 0)

    def test_invalid_financial_records_fail_explicitly(self):
        invalid = [
            self.features.drop(columns="BILL_AMT1"), self.features.assign(LIMIT_BAL=0),
            self.features.assign(PAY_AMT1=-1), self.features.assign(BILL_AMT1=np.nan),
            self.features.assign(BILL_AMT1=np.inf), self.features.assign(PAY_0=0.5),
        ]
        for frame in invalid:
            with self.assertRaises(ValueError):
                predict_pd(self.model, frame)

    def test_unseen_status_uses_learned_rare_pool(self):
        # Compare the original-feature model: engineering also uses the numeric
        # delay magnitude, so two different positive codes need not have equal PD.
        model = fit_with_convergence_check(make_pipeline(solver="liblinear"), self.features, self.target)
        customer = self.features.iloc[[0]].copy()
        rare_pd = predict_pd(model, customer)
        customer.loc[:, "PAY_0"] = 99
        with self.assertWarnsRegex(UserWarning, "unknown categories"):
            unknown_pd = predict_pd(model, customer)
        np.testing.assert_array_equal(unknown_pd, rare_pd)

    def test_inference_does_not_update_learned_scaling(self):
        scaler = self.model.named_steps["preprocessor"].named_transformers_["numeric"]
        before_mean, before_var = scaler.mean_.copy(), scaler.var_.copy()
        shifted = self.features.copy()
        shifted.loc[:, NUMERIC_FEATURES] = shifted[NUMERIC_FEATURES] + 1_000_000
        predict_pd(self.model, shifted)
        np.testing.assert_array_equal(scaler.mean_, before_mean)
        np.testing.assert_array_equal(scaler.var_, before_var)
        self.assertEqual(scaler.n_samples_seen_, len(self.features))

    def test_cv_fits_scalers_and_categories_on_each_training_fold_only(self):
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(self.features, self.target))
        results = cross_validate(make_pipeline(solver="liblinear"), self.features, self.target,
                                 cv=folds, scoring="neg_log_loss", return_estimator=True, error_score="raise")
        for pipeline, (fit_rows, held_out_rows) in zip(results["estimator"], folds):
            self.assertFalse(np.intersect1d(fit_rows, held_out_rows).size)
            fitted_features = self.features.iloc[fit_rows]
            scaler = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
            np.testing.assert_allclose(scaler.mean_, fitted_features[NUMERIC_FEATURES].mean(), rtol=1e-12)
            self.assertEqual(scaler.n_samples_seen_, len(fit_rows))
            encoder = pipeline.named_steps["preprocessor"].named_transformers_["status"]
            for column, categories in zip(STATUS_FEATURES, encoder.categories_):
                np.testing.assert_array_equal(categories, np.sort(fitted_features[column].unique()))

    def test_pipeline_and_calibrated_model_serialize_exactly(self):
        calibrated = fit_with_convergence_check(
            CalibratedClassifierCV(make_pipeline(solver="liblinear"), cv=3, method="sigmoid", ensemble=False),
            self.features, self.target, parameter_prefix="estimator__",
        )
        self.assertEqual(underlying_pipeline(calibrated).named_steps["features"].n_samples_seen_, len(self.features))
        with tempfile.TemporaryDirectory() as directory:
            for model in [self.model, calibrated]:
                path = Path(directory) / "pipeline.joblib"
                joblib.dump(model, path)
                pd.testing.assert_series_equal(predict_pd(model, self.features), predict_pd(joblib.load(path), self.features))

    def test_l1_and_l2_fit_with_supported_solver(self):
        for ratio in [0, 1]:
            model = fit_with_convergence_check(make_pipeline(solver="liblinear", l1_ratio=ratio), self.features, self.target)
            self.assertEqual(len(predict_pd(model, self.features)), len(self.features))

    def test_coefficient_odds_ratios_and_reference_metadata(self):
        table = coefficient_table(self.model)
        np.testing.assert_allclose(table.odds_ratio, np.exp(table.coefficient))
        self.assertTrue(table.comparison.str.contains("training standard deviation|encoded category").all())
        self.assertEqual(len(table), self.model.named_steps["classifier"].coef_.shape[1])

    def test_all_compared_models_share_the_same_ordered_observations(self):
        values = predict_pd(self.model, self.features)
        frame = make_prediction_frame(self.target, {"first": values, "second": values.to_numpy()}, "a" * 64)
        self.assertEqual(len(frame), len(self.features))
        self.assertTrue(frame.row_key.is_unique)
        np.testing.assert_array_equal(frame[TARGET], self.target)
        np.testing.assert_array_equal(frame["first"], frame["second"])
        with self.assertRaises(ValueError):
            make_prediction_frame(self.target, {"short": values.iloc[:-1]}, "a" * 64)

    def test_split_hashes_reject_swapped_or_changed_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for i, split in enumerate(["train", "validation", "test"]):
                path = root / f"{split}.csv"
                path.write_text(f"sample\n{i}\n", encoding="utf-8")
                hashes[split] = {"sha256": file_hash(path)}
            (root / "step_04_run_manifest.json").write_text(json.dumps({"inputs": hashes}), encoding="utf-8")
            verify_split_inputs(root, root)
            (root / "test.csv").write_bytes((root / "train.csv").read_bytes())
            with self.assertRaises(ValueError):
                verify_split_inputs(root, root)

    def test_test_data_cannot_be_loaded_before_selection_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("run_step_05.pd.read_csv") as read_csv:
                with self.assertRaises(FileNotFoundError):
                    evaluate_frozen_test({}, root, root / "missing_selection.json")
                read_csv.assert_not_called()

    def test_recorded_test_evaluation_does_not_repeat_inference_or_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}", encoding="utf-8")
            (root / "step_05_test_evaluation.json").write_text(json.dumps({
                "selection_sha256": file_hash(selection), "saved_evaluation_hashes": {},
            }), encoding="utf-8")
            with patch("run_step_05.pd.read_csv") as read_csv:
                evaluate_frozen_test({}, root, selection)
                read_csv.assert_not_called()

    def test_constant_pd_and_tied_deciles_keep_all_rows(self):
        target, predictions = [0, 0, 0, 1], [0.25] * 4
        metrics = probability_metrics(target, predictions)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["average_precision"], 0.25)
        deciles = decile_table(target, predictions)
        self.assertEqual(deciles.customers.sum(), 4)
        self.assertEqual(len(deciles), 1)


if __name__ == "__main__":
    unittest.main()
