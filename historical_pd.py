"""Training-only historical probability-of-default estimators for Step 4.

``predict`` returns one PD per input row, preserving row order. Only PAY_0 and,
optionally, LIMIT_BAL are read; extra columns (including a target) are ignored.
The JSON helpers preserve fitted estimates for later comparisons without refitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default_next_month"


def binary_target(values, expected_length: int | None = None) -> np.ndarray:
    """Validate a nonempty, one-dimensional binary target without coercing labels."""
    target = np.asarray(values, dtype=float)
    if target.ndim != 1 or not len(target):
        raise ValueError("The target must be a nonempty one-dimensional array.")
    if expected_length is not None and len(target) != expected_length:
        raise ValueError("Features and target have different lengths.")
    if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("The target must contain only 0 and 1, with no missing values.")
    return target


def check_predictions(values, expected_length: int) -> np.ndarray:
    """Check length, missing/infinite values, and the inclusive probability range."""
    predictions = np.asarray(values, dtype=float)
    if predictions.shape != (expected_length,):
        raise ValueError("Expected exactly one PD per customer.")
    if not np.isfinite(predictions).all():
        raise ValueError("PD predictions contain missing or infinite values.")
    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("PD predictions must lie between 0 and 1.")
    return predictions


class PortfolioHistoricalPD:
    """Assign the observed training default frequency to every customer."""

    def fit(self, features: pd.DataFrame, target) -> PortfolioHistoricalPD:
        target = binary_target(target, len(features))
        self.training_size_ = len(target)
        self.training_defaults_ = int(target.sum())
        self.portfolio_pd_ = float(target.mean())
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "portfolio_pd_"):
            raise ValueError("Fit the historical PD estimator before predicting.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return check_predictions(
            np.full(len(features), self.portfolio_pd_), len(features)
        )


class SegmentedHistoricalPD(PortfolioHistoricalPD):
    """Estimate PD by PAY_0, optionally crossed with training credit-limit bands.

    ``smoothing`` is the number of pseudo-observations assigned to the parent
    rate. PAY_0 rates shrink to the portfolio rate; joint segment rates shrink
    to their already-smoothed PAY_0 rate. An unseen/missing repayment status
    falls back to the portfolio. An unseen joint segment or a missing/nonpositive
    credit limit falls back to its PAY_0 rate. Out-of-range positive credit limits
    enter the first/last training band; boundaries are never recalculated at predict.
    """

    def __init__(
        self,
        use_credit_limit: bool = False,
        n_limit_bands: int = 5,
        smoothing: float = 20.0,
    ) -> None:
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be a positive finite number.")
        if not isinstance(n_limit_bands, int) or n_limit_bands < 2:
            raise ValueError("n_limit_bands must be an integer of at least 2.")
        self.use_credit_limit = use_credit_limit
        self.n_limit_bands = n_limit_bands
        self.smoothing = float(smoothing)

    def _features(self, features: pd.DataFrame) -> pd.DataFrame:
        required = ["PAY_0"] + (["LIMIT_BAL"] if self.use_credit_limit else [])
        missing = set(required).difference(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        result = pd.DataFrame(index=range(len(features)))
        status = pd.to_numeric(features["PAY_0"], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        finite = np.isfinite(status)
        if np.isinf(status).any() or (status[finite] != np.floor(status[finite])).any():
            raise ValueError("PAY_0 must contain integer categories or missing values.")
        result["PAY_0"] = status
        if self.use_credit_limit:
            limits = pd.to_numeric(features["LIMIT_BAL"], errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
            result["LIMIT_BAL"] = np.where(
                np.isfinite(limits) & (limits > 0), limits, np.nan
            )
        return result

    def _bands(self, features: pd.DataFrame) -> np.ndarray:
        limits = features["LIMIT_BAL"].to_numpy()
        # Left insertion puts values equal to a cutoff in the lower band: (a, b].
        return np.where(
            np.isfinite(limits),
            np.searchsorted(self.credit_limit_cutoffs_, limits, side="left"),
            -1,
        )

    def fit(self, features: pd.DataFrame, target) -> SegmentedHistoricalPD:
        frame = self._features(features)
        super().fit(features, target)
        frame[TARGET] = binary_target(target, len(frame))
        self.status_table_ = (
            frame.groupby("PAY_0", observed=True)[TARGET]
            .agg(customers="size", defaults="sum")
            .reset_index()
        )
        self.status_table_["raw_pd"] = (
            self.status_table_["defaults"] / self.status_table_["customers"]
        )
        self.status_table_["parent_pd"] = self.portfolio_pd_
        self.status_table_["estimated_pd"] = (
            self.status_table_["defaults"] + self.smoothing * self.portfolio_pd_
        ) / (self.status_table_["customers"] + self.smoothing)
        self.status_rates_ = dict(
            zip(self.status_table_["PAY_0"], self.status_table_["estimated_pd"])
        )
        self.credit_limit_cutoffs_ = np.array([], dtype=float)
        self.segment_table_ = pd.DataFrame()
        self.segment_rates_ = {}
        if self.use_credit_limit:
            limits = frame["LIMIT_BAL"].dropna().to_numpy()
            if not len(limits):
                raise ValueError("Training requires at least one positive finite credit limit.")
            # Drop repeated quantiles before discarding outer boundaries. All-equal
            # limits therefore produce one band instead of a failing qcut operation.
            edges = np.unique(np.quantile(limits, np.linspace(0, 1, self.n_limit_bands + 1)))
            self.credit_limit_cutoffs_ = edges[1:-1]
            frame["limit_band"] = self._bands(frame)
            self.segment_table_ = (
                frame.loc[frame["limit_band"] >= 0]
                .groupby(["PAY_0", "limit_band"], observed=True)[TARGET]
                .agg(customers="size", defaults="sum")
                .reset_index()
            )
            table = self.segment_table_
            table["raw_pd"] = table["defaults"] / table["customers"]
            table["parent_pd"] = table["PAY_0"].map(self.status_rates_)
            table["estimated_pd"] = (
                table["defaults"] + self.smoothing * table["parent_pd"]
            ) / (table["customers"] + self.smoothing)
            self.segment_rates_ = {
                (row.PAY_0, row.limit_band): row.estimated_pd
                for row in table.itertuples(index=False)
            }
        return self

    def prediction_details(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions and their fallback source, in input row order."""
        self._check_fitted()
        frame = self._features(features)
        status_pd = frame["PAY_0"].map(self.status_rates_)
        # pandas 3 uses read-only NumPy views; joint updates need an owned array.
        predictions = status_pd.fillna(self.portfolio_pd_).to_numpy(copy=True)
        source = np.where(status_pd.notna(), "PAY_0", "portfolio").astype(object)
        if self.use_credit_limit:
            frame["limit_band"] = self._bands(frame)
            for index, (status, band) in enumerate(
                zip(frame["PAY_0"], frame["limit_band"])
            ):
                joint_pd = self.segment_rates_.get((status, band))
                if joint_pd is not None:
                    predictions[index] = joint_pd
                    source[index] = "PAY_0 + LIMIT_BAL"
        frame["predicted_pd"] = check_predictions(predictions, len(features))
        frame["prediction_source"] = source
        frame.index = features.index
        return frame.drop(columns=["LIMIT_BAL"], errors="ignore")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.prediction_details(features)["predicted_pd"].to_numpy()


def save_model(model: PortfolioHistoricalPD, path: str | Path) -> None:
    """Save only fitted parameters and training counts; no customer-level data."""
    model._check_fitted()
    state = {
        "format_version": 1,
        "model_type": "segmented" if isinstance(model, SegmentedHistoricalPD) else "portfolio",
        "training_size": model.training_size_,
        "training_defaults": model.training_defaults_,
        "portfolio_pd": model.portfolio_pd_,
    }
    if isinstance(model, SegmentedHistoricalPD):
        state.update(
            use_credit_limit=model.use_credit_limit,
            n_limit_bands=model.n_limit_bands,
            smoothing=model.smoothing,
            credit_limit_cutoffs=model.credit_limit_cutoffs_.tolist(),
            status_table=model.status_table_.to_dict(orient="records"),
            segment_table=model.segment_table_.to_dict(orient="records"),
        )
    Path(path).write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(path: str | Path) -> PortfolioHistoricalPD:
    """Restore a model produced by save_model, without training or tuning again."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state["format_version"] != 1 or state["model_type"] not in {"portfolio", "segmented"}:
        raise ValueError("Unsupported historical PD model format.")
    model = PortfolioHistoricalPD()
    if state["model_type"] == "segmented":
        model = SegmentedHistoricalPD(
            state["use_credit_limit"], state["n_limit_bands"], state["smoothing"]
        )
        model.status_table_ = pd.DataFrame(state["status_table"])
        model.segment_table_ = pd.DataFrame(state["segment_table"])
        model.credit_limit_cutoffs_ = np.asarray(state["credit_limit_cutoffs"], dtype=float)
        model.status_rates_ = {
            row["PAY_0"]: row["estimated_pd"] for row in state["status_table"]
        }
        model.segment_rates_ = {
            (row["PAY_0"], row["limit_band"]): row["estimated_pd"]
            for row in state["segment_table"]
        }
    model.training_size_ = state["training_size"]
    model.training_defaults_ = state["training_defaults"]
    model.portfolio_pd_ = state["portfolio_pd"]
    return model
