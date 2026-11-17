"""Behavioral/financial logistic PD pipelines and reusable, indexed predictions.

No target or demographic column is used. Feature formulas are fixed and row-wise;
all learned encoding, rare-category pooling, and scaling live inside the pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from historical_pd import check_predictions

STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERIC_FEATURES = ["LIMIT_BAL", *BILL_FEATURES, *PAYMENT_FEATURES]
RAW_FEATURES = [*NUMERIC_FEATURES, *STATUS_FEATURES]
EXCLUDED_DEMOGRAPHICS = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
ENGINEERED_FEATURES = [
    "utilization_recent",
    "utilization_mean",
    "payment_to_previous_bill",
    "payment_to_previous_bills_5m",
    "no_positive_previous_bill",
    "delinquent_months",
    "max_delay",
    "balance_change_to_limit",
]


class FinancialFeatures(TransformerMixin, BaseEstimator):
    """Select 19 original predictors and optionally add eight fixed features.

    Negative bills remain signed. Payment coverage uses positive previous-month
    bills only, with zero coverage when no positive denominator is available and
    a separate recent-bill indicator. Fixed caps limit ratio sensitivity without
    learning cutoffs from other customers. Inputs must be finite numeric records,
    with positive credit limits and integer repayment codes; missing required
    fields/values are rejected explicitly rather than silently inventing records.
    """

    def __init__(self, engineered: bool = False):
        self.engineered = engineered

    def fit(self, features: pd.DataFrame, target=None):
        self.transform(features)
        self.n_samples_seen_ = len(features)
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = len(features.columns)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Pass customer records as a pandas DataFrame with named columns.")
        if not features.columns.is_unique:
            raise ValueError("Customer column names must be unique.")
        missing = set(RAW_FEATURES).difference(features.columns)
        if missing:
            raise ValueError(f"Missing required financial features: {sorted(missing)}")
        result = features[RAW_FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Financial inputs contain missing or infinite values.")
        if (result["LIMIT_BAL"] <= 0).any():
            raise ValueError("Credit limits must be positive.")
        statuses = result[STATUS_FEATURES].to_numpy()
        if (statuses != np.floor(statuses)).any():
            raise ValueError("Repayment status codes must be integers.")
        if (result[PAYMENT_FEATURES] < 0).any().any():
            raise ValueError("Payment amounts must be nonnegative.")
        if self.engineered:
            limit = result["LIMIT_BAL"].to_numpy()
            bills = result[BILL_FEATURES].to_numpy()
            payments = result[PAYMENT_FEATURES].to_numpy()

            def coverage(numerator, denominator):
                ratio = np.divide(
                    numerator, denominator,
                    out=np.zeros(len(result), dtype=float), where=denominator > 0,
                )
                return np.clip(ratio, 0, 5)

            # PAY_AMT1 is September payment; BILL_AMT2 is the August statement.
            # This is an approximate coverage proxy, not a contractual repayment rate.
            result["utilization_recent"] = np.clip(bills[:, 0] / limit, -5, 10)
            result["utilization_mean"] = np.clip(bills.mean(axis=1) / limit, -5, 10)
            result["payment_to_previous_bill"] = coverage(payments[:, 0], bills[:, 1])
            result["payment_to_previous_bills_5m"] = coverage(
                payments[:, :5].sum(axis=1), np.maximum(bills[:, 1:], 0).sum(axis=1)
            )
            result["no_positive_previous_bill"] = (bills[:, 1] <= 0).astype(int)
            result["delinquent_months"] = (statuses > 0).sum(axis=1)
            result["max_delay"] = np.maximum(statuses.max(axis=1), 0)
            result["balance_change_to_limit"] = np.clip((bills[:, 0] - bills[:, 1]) / limit, -10, 10)
            if not np.isfinite(result.to_numpy()).all():
                raise ValueError("Engineered features overflowed; check financial inputs.")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(RAW_FEATURES + (ENGINEERED_FEATURES if self.engineered else []), dtype=object)


def make_pipeline(
    engineered: bool = False,
    solver: str = "lbfgs",
    C: float = 1.0,
    l1_ratio: float = 0.0,
    max_iter: int = 2000,
) -> Pipeline:
    """Build an unfitted pipeline for scikit-learn 1.9 (no deprecated penalty API)."""
    numeric = NUMERIC_FEATURES + (ENGINEERED_FEATURES if engineered else [])
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("status", OneHotEncoder(
            min_frequency=20,
            drop="first",
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        ), STATUS_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
    classifier = LogisticRegression(
        solver=solver, C=C, l1_ratio=l1_ratio, max_iter=max_iter,
        random_state=42, class_weight=None, intercept_scaling=10,
    )
    return Pipeline([
        ("features", FinancialFeatures(engineered=engineered)),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def fit_with_convergence_check(estimator, features, target, parameter_prefix=""):
    """Treat non-convergence as a failure; retry once with a higher iteration cap.

    Works for pipelines, GridSearchCV, and CalibratedClassifierCV by specifying
    the prefix of their contained pipeline. Other warnings are left visible.
    """
    parameter = parameter_prefix + "classifier__max_iter"
    for limit in [2000, 8000]:
        estimator.set_params(**{parameter: limit})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return estimator.fit(features, target)
        except ConvergenceWarning:
            if limit == 8000:
                raise
            print("Convergence limit reached; retrying with max_iter=8000.", flush=True)


def predict_pd(model, customers: pd.DataFrame) -> pd.Series:
    """Return default probabilities with exactly the input index and row order.

    ``model`` can be a fitted pipeline or a cross-validated calibrated wrapper.
    An empty input returns an empty indexed Series. Extra target/demographic
    columns are ignored by the pipeline's FinancialFeatures transformer.
    """
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Expected a fitted binary PD model with classes [0, 1].")
    if not len(customers):
        return pd.Series([], index=customers.index, name="predicted_pd", dtype=float)
    values = model.predict_proba(customers)[:, 1]
    return pd.Series(check_predictions(values, len(customers)), index=customers.index, name="predicted_pd")


def underlying_pipeline(model) -> Pipeline:
    """Get the all-training logistic pipeline, including for ensemble=False calibration."""
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV) and len(model.calibrated_classifiers_) == 1:
        return model.calibrated_classifiers_[0].estimator
    raise TypeError("Expected a pipeline or single-estimator calibrated pipeline.")


def coefficient_table(model) -> pd.DataFrame:
    """Coefficients/odds ratios of the underlying logistic score, not a calibrator."""
    pipeline = underlying_pipeline(model)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    numeric_names = NUMERIC_FEATURES + (
        ENGINEERED_FEATURES if pipeline.named_steps["features"].engineered else []
    )
    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["status"]
    references = {
        feature: encoder.categories_[i][encoder.drop_idx_[i]]
        for i, feature in enumerate(STATUS_FEATURES)
    }
    rows = []
    for name, coefficient in zip(names, coefficients):
        feature = name.split("__", 1)[1]
        row = {
            "feature": feature,
            "coefficient": float(coefficient),
            "odds_ratio": float(np.exp(coefficient)),
            "direction": "higher odds" if coefficient > 1e-12 else "lower odds" if coefficient < -1e-12 else "zero coefficient",
            "comparison": "one training standard deviation increase",
            "training_mean": np.nan,
            "training_standard_deviation": np.nan,
        }
        if name.startswith("numeric__"):
            index = numeric_names.index(feature)
            row["training_mean"] = float(scaler.mean_[index])
            row["training_standard_deviation"] = float(np.sqrt(scaler.var_[index]))
        else:
            status = next(value for value in STATUS_FEATURES if feature.startswith(value + "_"))
            row["comparison"] = f"encoded category versus {status}={references[status]:g}"
        rows.append(row)
    return pd.DataFrame(rows).assign(
        absolute_coefficient=lambda frame: frame.coefficient.abs()
    ).sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
