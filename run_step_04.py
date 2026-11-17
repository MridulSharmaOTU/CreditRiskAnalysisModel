r"""Run historical PD benchmarks against existing Step 3 CSVs, without network access.

Usage: .venv\Scripts\python.exe run_step_04.py
Selection uses validation log loss (then Brier score for a tie). Test targets are
read only AFTER selection, solely for the requested overall descriptive rate.
No test metrics, segmented test predictions, or Step 5 models are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from historical_pd import (
    TARGET,
    PortfolioHistoricalPD,
    SegmentedHistoricalPD,
    binary_target,
    check_predictions,
    load_model,
    save_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SMOOTHING_GRID = (5, 20, 50, 100, 200)
CALIBRATION_EDGES = np.linspace(0, 1, 11)
RISK_EDGES = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
COLORS = ("#526879", "#228275", "#c47b26")


def probability_groups(target, predictions, edges) -> pd.DataFrame:
    """Summarize fixed probability bands; labels describe right-closed bins."""
    target = binary_target(target)
    predictions = check_predictions(predictions, len(target))
    labels = [
        f"{'[' if i == 0 else '('}{lower:.0%}, {upper:.0%}]"
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    frame = pd.DataFrame({"target": target, "predicted_pd": predictions})
    frame["pd_band"] = pd.cut(
        predictions, bins=edges, labels=labels, include_lowest=True, right=True
    )
    groups = (
        frame.groupby("pd_band", observed=True)
        .agg(
            customers=("target", "size"),
            defaults=("target", "sum"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("target", "mean"),
        )
        .reset_index()
    )
    groups["defaults"] = groups["defaults"].astype(int)
    groups["calibration_gap"] = groups["mean_predicted_pd"] - groups["observed_default_rate"]
    return groups


def evaluate(target, predictions) -> dict:
    """Validation metrics; clipping applies only to numerical log-loss scoring."""
    target = binary_target(target)
    predictions = check_predictions(predictions, len(target))
    calibration = probability_groups(target, predictions, CALIBRATION_EDGES)
    return {
        "brier_score": float(brier_score_loss(target, predictions)),
        "log_loss": float(log_loss(target, np.clip(predictions, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, predictions)) if len(np.unique(target)) == 2 else np.nan,
        "mean_predicted_pd": float(predictions.mean()),
        "observed_default_rate": float(target.mean()),
        "calibration_bias": float(predictions.mean() - target.mean()),
        "ece_10_bins": float(
            np.average(abs(calibration["calibration_gap"]), weights=calibration["customers"])
        ),
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    """Avoid adding tabulate as a dependency just for report formatting."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}" if np.isfinite(value) else "N/A"
        return str(value)

    rows = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    rows.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def save_figures(figures_dir, comparison, models, selected_id, calibration, risk_groups):
    labels = comparison["approach"].tolist()
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}):
        status_model = models[comparison.loc[comparison["approach"] == "PAY_0", "model_id"].iloc[0]]
        table = status_model.status_table_
        positions = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(positions, table["raw_pd"], color=COLORS[0], label="Observed training rate")
        ax.plot(positions, table["estimated_pd"], "o-", color=COLORS[1], label="Smoothed training PD")
        for x, row in zip(positions, table.itertuples(index=False)):
            ax.annotate(f"n={row.customers:,}", (x, max(row.raw_pd, row.estimated_pd)),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(positions, [str(int(value)) for value in table["PAY_0"]])
        ax.set(xlabel="Most recent repayment status (PAY_0 code)", ylabel="Training default rate / estimated PD",
               title="Historical PD by repayment status (training only)", ylim=(0, 1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "step_04_default_by_pay_0.png", dpi=160)
        plt.close(fig)

        groups = risk_groups.loc[risk_groups["model_id"] == selected_id]
        positions = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(positions - 0.18, groups["mean_predicted_pd"], width=0.36,
               color=COLORS[1], label="Mean predicted PD")
        ax.bar(positions + 0.18, groups["observed_default_rate"], width=0.36,
               color=COLORS[2], label="Observed validation default rate")
        for x, row in zip(positions, groups.itertuples(index=False)):
            ax.annotate(f"n={row.customers:,}", (x, max(row.mean_predicted_pd, row.observed_default_rate)),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(positions, groups["pd_band"].astype(str))
        ax.set(xlabel="Fixed bands of predicted PD (empty bands omitted)", ylabel="Probability of default",
               title=f"Selected segmented baseline: validation risk groups\n{selected_id}")
        ax.set_ylim(0, min(1.1, max(groups[["mean_predicted_pd", "observed_default_rate"]].max()) + 0.12))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "step_04_risk_group_calibration.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", label="Perfect calibration")
        for color, row in zip(COLORS, comparison.itertuples(index=False)):
            groups = calibration.loc[calibration["model_id"] == row.model_id]
            ax.plot(groups["mean_predicted_pd"], groups["observed_default_rate"], "o-",
                    color=color, label=row.approach)
        ax.set(xlabel="Mean predicted PD", ylabel="Observed validation default rate",
               title="Validation calibration (10 fixed probability bins)", xlim=(0, 1), ylim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "step_04_calibration.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        for ax, metric, title in zip(axes, ["brier_score", "log_loss"], ["Brier score", "Log loss"]):
            bars = ax.bar(labels, comparison[metric], color=COLORS)
            ax.bar_label(bars, fmt="%.4f", padding=4)
            ax.set(ylabel=title, title=f"{title} (lower is better)", xlabel="Historical PD approach")
            ax.set_ylim(0, comparison[metric].max() * 1.2)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Validation probability accuracy")
        fig.tight_layout()
        fig.savefig(figures_dir / "step_04_validation_scores.png", dpi=160)
        plt.close(fig)


def write_report(report_dir, train, validation, test_summary, comparison, candidates,
                 models, selected_id, recommended_id, risk_groups, audit, sources):
    portfolio = models["portfolio"]
    selected = comparison.loc[comparison["model_id"] == selected_id].iloc[0]
    baseline = comparison.loc[comparison["model_id"] == "portfolio"].iloc[0]
    status_result = comparison.loc[comparison["approach"] == "PAY_0"].iloc[0]
    selected_risk = risk_groups.loc[risk_groups["model_id"] == selected_id].drop(columns="model_id")
    joint_id = comparison.loc[comparison["approach"] == "PAY_0 + LIMIT_BAL", "model_id"].iloc[0]
    joint = models[joint_id]
    cutoffs = ", ".join(f"{value:,.0f}" for value in joint.credit_limit_cutoffs_)
    best_per_family = comparison[["approach", "smoothing", "brier_score", "log_loss", "roc_auc",
                                  "mean_predicted_pd", "calibration_bias", "ece_10_bins"]]
    top = candidates.sort_values(["log_loss", "brier_score", "n_segmentation_features", "model_id"]).iloc[0]
    brier_change = 100 * (baseline.brier_score - selected.brier_score) / baseline.brier_score
    log_change = 100 * (baseline.log_loss - selected.log_loss) / baseline.log_loss
    status_table = models[comparison.loc[comparison["approach"] == "PAY_0", "model_id"].iloc[0]].status_table_
    report = f"""# Step 4: Historical probability of default

## Objective

Build reproducible historical PD benchmarks for the later logistic-regression comparison.
This step fits a portfolio rate, PAY_0 rates, and PAY_0 crossed with credit-limit bands.
Step 5, LGD, and ECL are outside this implementation.

## Existing dataset and split audit

The existing `main.py` implements the UCI ID 350 download, descriptive renaming,
EDUCATION 5/6-to-0 cleaning, and stratified 70/15/15 split with random_state=42.
All three processed CSVs already existed. They were reused without fetching data,
rerunning Step 3, deduplicating, or rewriting any input. No notebook was present.
Training and validation each have {train.shape[1] - 1} predictors plus `{TARGET}`,
zero missing values, binary targets, and cleaned EDUCATION codes 0 through 4.

| Split | Customers | Defaults | Observed default rate |
| --- | ---: | ---: | ---: |
| Training | {len(train):,} | {portfolio.training_defaults_:,} | {portfolio.portfolio_pd_:.6%} |
| Validation | {len(validation):,} | {int(validation[TARGET].sum()):,} | {validation[TARGET].mean():.6%} |
| Test (descriptive only) | {test_summary['customers']:,} | {test_summary['defaults']:,} | {test_summary['default_rate']:.6%} |

There are {audit['train_duplicate_rows']} duplicate full rows within training and
{audit['validation_duplicate_rows']} within validation. {audit['validation_feature_matches_training']}
validation rows have an exact match across all predictor columns in training.
The CSVs lack customer IDs, so independent customers cannot be verified. These
pre-existing repeats can make validation optimistic; preserve the splits for the
Step 5 comparison and investigate them before any later resplitting.

## Historical PD calculation

The portfolio estimate is {portfolio.training_defaults_:,} / {portfolio.training_size_:,}
= **{portfolio.portfolio_pd_:.8f} ({portfolio.portfolio_pd_:.4%})**.
Every validation customer receives this value. The requested constant assignment
for every test row is saved in `tables/step_04_test_portfolio_predictions.csv`;
it uses no test predictors or individual outcomes. A constant score cannot
distinguish low-risk and high-risk borrowers: every pair is tied and validation
ROC-AUC is {baseline.roc_auc:.6f} when both target classes are present.

## Segmentation and sparse groups

Only PAY_0 and LIMIT_BAL enter the segmented models. Demographics and additional
financial ratios are excluded to keep these benchmarks simple and interpretable.
PAY_0 uses the recorded integer categories, without imposing undocumented meanings
on the nonpositive codes. Both segmented models are fitted using training outcomes only.

The joint model requests five training quantile bands. Its internal cutoffs are
**{cutoffs} NTD**, producing {len(joint.credit_limit_cutoffs_) + 1} actual bands.
Intervals are right-closed; e.g. a value equal to the first cutoff belongs to the
first band. The outer bands extend to negative/positive infinity mathematically;
only positive finite limits are accepted into them. Positive values outside the
training range enter the outer bands. Repeated quantile edges are removed; constant
training limits give one band. Validation/test values never set the boundaries.

For alpha pseudo-observations, training estimates are:

```text
portfolio_pd = training_defaults / training_customers
status_pd = (status_defaults + alpha * portfolio_pd) / (status_customers + alpha)
joint_pd = (joint_defaults + alpha * status_pd) / (joint_customers + alpha)
```

All observed groups are smoothed; small groups receive more shrinkage. This is a
hierarchical heuristic, not an independently specified Bayesian posterior, since
the parent rates also contain the subgroup observations. Unseen joint combinations
and missing/nonpositive/nonfinite credit limits fall back to the smoothed status PD.
Unseen or missing PAY_0 falls back to portfolio PD. Missing required columns,
nonnumeric inputs, and noninteger repayment codes raise a clear error.

Training repayment-status estimates for the selected PAY_0-only configuration:

{markdown_table(status_table)}

## Validation selection and results

Each segmentation family tries alpha in {list(SMOOTHING_GRID)} with five requested
credit-limit bands fixed in advance. The selection rule is lowest validation log
loss, then lowest Brier score; exact remaining ties prefer fewer segmentation
features and then model ID. No validation target enters training rates or cutoffs.
Validation labels are used for scoring and configuration selection only.
All {len(candidates)} candidate scores are in `tables/step_04_tuning_results.csv`.
The table below retains the best alpha within each segmentation family; smoothing
is N/A for the portfolio estimator. Brier, log loss, and ECE are lower-is-better;
ROC-AUC is higher-is-better. Calibration bias is predicted minus observed PD.

{markdown_table(best_per_family)}

The selected segmented configuration is **`{selected_id}`**. The lowest observed
validation log loss across all baselines belongs to **`{recommended_id}`**
({top.log_loss:.6f}). The selected segmented model changes Brier score by
{brier_change:.2f}% and log loss by {log_change:.2f}% relative to the portfolio
(positive percentages denote reductions). Its ROC-AUC is {selected.roc_auc:.6f},
versus {baseline.roc_auc:.6f} for the portfolio. These are point estimates on the
same validation set used for selection, not evidence of statistical significance
or out-of-sample superiority. Probability accuracy, ranking, and calibration
must be considered separately.

Calibration uses 10 fixed equal-width probability bins. ECE is the customer-weighted
absolute predicted-versus-observed gap across nonempty bins, so its value depends
on the chosen bins. Signed aggregate calibration bias can hide offsetting errors.
The calibration CSV includes counts and signed gaps; no post-hoc calibrator is fitted.

Selected segmented model risk analysis (fixed PD bands; empty bands omitted):

{markdown_table(selected_risk)}

The selected model's ECE is {selected.ece_10_bins:.6f}, compared with
{status_result.ece_10_bins:.6f} for PAY_0 alone and {baseline.ece_10_bins:.6f} for
the constant portfolio. Lower loss and better ranking therefore do not imply
better calibration by this binned measure. The portfolio's nearly exact aggregate
calibration is expected in these stratified splits and does not measure ranking.
The risk-band table shows local calibration with customer counts; thin bands have
noisier observed rates. Tables for actual PAY_0 and joint segments, including
fallback sources, are also saved separately.

## Figures

![Training default rates by repayment status](figures/step_04_default_by_pay_0.png)

![Predicted and observed validation PD by risk band](figures/step_04_risk_group_calibration.png)

![Validation calibration](figures/step_04_calibration.png)

![Validation Brier score and log loss](figures/step_04_validation_scores.png)

## Limitations and interpretation

The portfolio baseline can be well calibrated in aggregate while having no ranking
ability. The segmented baselines can distinguish financial risk groups, but assume
future customers resemble the training cohort. Rare groups and coarse credit-limit
bands limit precision. Smoothed probabilities depend on alpha and parent-rate quality.
No confidence intervals, significance tests, demographic fairness analysis, or
temporal validation are included. This snapshot predicts default next month; it
does not establish annual/lifetime PD, causal effects, LGD, or ECL.

The random split does not test changes in default behavior over time. Validation
was used repeatedly for a small predefined parameter grid and may be optimistic.
Identical feature records across splits are another limitation. Test outcomes did
not influence fitting or selection: only the explicitly requested overall test
count/default rate was read, after choices were frozen. There are no test loss,
AUC, calibration results, or segmented test predictions in Step 4.

## Reproduction and checks

From the project root, run:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe run_step_04.py
```

The runner is independent of the current working directory and accepts `--data-dir`
and `--report-dir`. It uses the installed pandas, NumPy, scikit-learn, and matplotlib
environment, requires no new dependency installation, and performs no random sampling.
Every prediction array is checked for length, finite values, and PD in [0, 1].
Saved JSON models are reloaded and checked for exact prediction agreement. Automated
tests also cover sparse/unseen groups, training-only cutoffs, missing inputs,
one-class evaluation, and ignored target/demographic columns at prediction time.
The runner verifies SHA-256 hashes of all three processed CSVs are unchanged.
`step_04_run_manifest.json` records these hashes, dependency versions, selection,
split counts, and fallback counts. Original `main.py` is preserved.

## Recommendations for Step 5

Keep these same processed splits and their hashes. Fit logistic regression using
training data only, including any scaling/encoding, and choose its hyperparameters
on validation with the same metrics. Freeze all choices before the single final
test comparison. Compare the portfolio, both frozen segmented configurations, and
logistic regression on exactly the same test rows. Retain calibration and risk-band
analysis alongside discrimination; report uncertainty before claiming superiority.
Investigate duplicate/customer identity and temporal representativeness before
using the estimates for decisions.

Load a frozen historical model for new customer rows without refitting:

```python
from historical_pd import load_model

model = load_model("reports/models/{selected_id}.json")
pd_values = model.predict(new_customers)  # DataFrame with PAY_0 and, if needed, LIMIT_BAL
```

Source files: `{sources['train']}`, `{sources['validation']}`, `{sources['test']}`.
"""
    (report_dir / "step_04_historical_pd.md").write_text(report, encoding="utf-8")


def run(data_dir: Path, report_dir: Path) -> pd.DataFrame:
    sources = {name: data_dir / f"{name}.csv" for name in ["train", "validation", "test"]}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing existing Step 3 artifact: {path}")
    hashes_before = {name: file_hash(path) for name, path in sources.items()}
    train = pd.read_csv(sources["train"])
    validation = pd.read_csv(sources["validation"])
    train_y = binary_target(train[TARGET], len(train))
    validation_y = binary_target(validation[TARGET], len(validation))
    # Explicit feature selection prevents accidental use of targets or demographics.
    train_x = train[["PAY_0", "LIMIT_BAL"]]
    validation_x = validation[["PAY_0", "LIMIT_BAL"]]
    if train.isna().any().any() or validation.isna().any().any():
        raise ValueError("Existing processed train/validation data unexpectedly contain missing values.")
    if set(train.columns) != set(validation.columns):
        raise ValueError("Training and validation columns differ.")

    models = {"portfolio": PortfolioHistoricalPD().fit(train_x, train_y)}
    predictions = {"portfolio": models["portfolio"].predict(validation_x)}
    rows = [{"model_id": "portfolio", "approach": "Portfolio", "smoothing": np.nan,
             "n_segmentation_features": 0, **evaluate(validation_y, predictions["portfolio"])}]
    for use_limit in [False, True]:
        for alpha in SMOOTHING_GRID:
            model_id = f"pay_0{'_limit' if use_limit else ''}_alpha_{alpha}"
            model = SegmentedHistoricalPD(use_credit_limit=use_limit, smoothing=alpha)
            models[model_id] = model.fit(train_x, train_y)
            predictions[model_id] = model.predict(validation_x)
            rows.append({"model_id": model_id,
                         "approach": "PAY_0 + LIMIT_BAL" if use_limit else "PAY_0",
                         "smoothing": alpha, "n_segmentation_features": 2 if use_limit else 1,
                         **evaluate(validation_y, predictions[model_id])})
    candidates = pd.DataFrame(rows)
    order = ["log_loss", "brier_score", "n_segmentation_features", "model_id"]
    ranked = candidates.sort_values(order)
    best_status = ranked.loc[ranked["approach"] == "PAY_0"].iloc[0]["model_id"]
    best_joint = ranked.loc[ranked["approach"] == "PAY_0 + LIMIT_BAL"].iloc[0]["model_id"]
    selected_id = ranked.loc[ranked["n_segmentation_features"] > 0].iloc[0]["model_id"]
    recommended_id = ranked.iloc[0]["model_id"]
    retained_ids = ["portfolio", best_status, best_joint]
    comparison = candidates.set_index("model_id").loc[retained_ids].reset_index()
    comparison["selected_segmented"] = comparison["model_id"] == selected_id
    comparison["recommended_baseline"] = comparison["model_id"] == recommended_id

    # Model and hyperparameter choices are now frozen. Read ONLY the test target
    # for the requested descriptive summary; do not score any model on it.
    test_target = binary_target(pd.read_csv(sources["test"], usecols=[TARGET])[TARGET])
    test_summary = {"customers": len(test_target), "defaults": int(test_target.sum()),
                    "default_rate": float(test_target.mean())}
    test_portfolio_pd = models["portfolio"].predict(pd.DataFrame(index=range(len(test_target))))
    del test_target

    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    models_dir = report_dir / "models"
    for directory in [figures_dir, tables_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(tables_dir / "step_04_tuning_results.csv", index=False)
    comparison.to_csv(tables_dir / "step_04_validation_metrics.csv", index=False)
    validation_predictions = pd.DataFrame({"row_number": np.arange(len(validation)), TARGET: validation_y.astype(int)})
    calibration_tables, risk_tables, fallback_counts = [], [], {}
    for model_id in retained_ids:
        validation_predictions[model_id] = predictions[model_id]
        calibration_tables.append(probability_groups(validation_y, predictions[model_id], CALIBRATION_EDGES).assign(model_id=model_id))
        risk_tables.append(probability_groups(validation_y, predictions[model_id], RISK_EDGES).assign(model_id=model_id))
        model = models[model_id]
        model_path = models_dir / f"{model_id}.json"
        save_model(model, model_path)
        if not np.array_equal(load_model(model_path).predict(validation_x), predictions[model_id]):
            raise RuntimeError(f"Saved model predictions changed for {model_id}.")
        if isinstance(model, SegmentedHistoricalPD):
            model.status_table_.to_csv(tables_dir / f"{model_id}_training_status.csv", index=False)
            details = model.prediction_details(validation_x)
            fallback_counts[model_id] = {str(key): int(value) for key, value in details["prediction_source"].value_counts().items()}
            details[TARGET] = validation_y
            keys = ["PAY_0"] + (["limit_band"] if model.use_credit_limit else []) + ["prediction_source"]
            segment_validation = details.groupby(keys, dropna=False, observed=True).agg(
                customers=(TARGET, "size"), defaults=(TARGET, "sum"),
                mean_predicted_pd=("predicted_pd", "mean"), observed_default_rate=(TARGET, "mean")
            ).reset_index()
            segment_validation["defaults"] = segment_validation["defaults"].astype(int)
            segment_validation.to_csv(tables_dir / f"{model_id}_validation_segments.csv", index=False)
            if model.use_credit_limit:
                model.segment_table_.to_csv(tables_dir / f"{model_id}_training_segments.csv", index=False)
    calibration = pd.concat(calibration_tables, ignore_index=True)
    risk_groups = pd.concat(risk_tables, ignore_index=True)
    validation_predictions.to_csv(tables_dir / "step_04_validation_predictions.csv", index=False)
    pd.DataFrame({"row_number": np.arange(test_summary["customers"]), "portfolio_pd": test_portfolio_pd}).to_csv(
        tables_dir / "step_04_test_portfolio_predictions.csv", index=False
    )
    calibration.to_csv(tables_dir / "step_04_calibration.csv", index=False)
    risk_groups.to_csv(tables_dir / "step_04_risk_groups.csv", index=False)
    save_figures(figures_dir, comparison, models, selected_id, calibration, risk_groups)

    feature_columns = [column for column in train.columns if column != TARGET]
    audit = {
        "train_duplicate_rows": int(train.duplicated().sum()),
        "validation_duplicate_rows": int(validation.duplicated().sum()),
        "validation_feature_matches_training": int(pd.MultiIndex.from_frame(validation[feature_columns]).isin(
            pd.MultiIndex.from_frame(train[feature_columns])).sum()),
    }
    hashes_after = {name: file_hash(path) for name, path in sources.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A processed input CSV changed during the Step 4 run.")
    manifest = {
        "step": 4,
        "inputs": {name: {"path": str(sources[name].resolve()), "sha256": hashes_before[name]} for name in sources},
        "input_files_unchanged": True,
        "split_summary": {
            "train": {"customers": len(train), "defaults": int(train_y.sum()), "default_rate": float(train_y.mean())},
            "validation": {"customers": len(validation), "defaults": int(validation_y.sum()), "default_rate": float(validation_y.mean())},
            "test_descriptive_only": test_summary,
        },
        "smoothing_grid": list(SMOOTHING_GRID),
        "selection_order": order,
        "selected_segmented_model": selected_id,
        "recommended_baseline": recommended_id,
        "retained_models": retained_ids,
        "test_model_metrics_computed": False,
        "test_segmented_predictions_computed": False,
        "validation_prediction_sources": fallback_counts,
        "duplicate_audit": audit,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
    }
    (report_dir / "step_04_run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (models_dir / "selection.json").write_text(json.dumps({key: manifest[key] for key in [
        "selected_segmented_model", "recommended_baseline", "retained_models", "selection_order", "smoothing_grid"
    ]}, indent=2) + "\n", encoding="utf-8")
    write_report(report_dir, train, validation, test_summary, comparison, candidates, models,
                 selected_id, recommended_id, risk_groups, audit, sources)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    arguments = parser.parse_args()
    comparison = run(arguments.data_dir.resolve(), arguments.report_dir.resolve())
    print(comparison[["model_id", "brier_score", "log_loss", "roc_auc", "selected_segmented"]].to_string(index=False))
    print(f"\nReport: {arguments.report_dir.resolve() / 'step_04_historical_pd.md'}")
    print("Processed CSVs unchanged. No test scoring or Step 5 models were run.")


if __name__ == "__main__":
    main()
