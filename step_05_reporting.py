"""Metrics, figures, and the Step 5 report, using saved, aligned predictions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_curve,
)

from historical_pd import TARGET, binary_target, check_predictions
from run_step_04 import CALIBRATION_EDGES, evaluate, markdown_table, probability_groups


def classification_metrics(target, predictions, threshold=0.5):
    target = binary_target(target)
    predictions = check_predictions(predictions, len(target))
    if not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must lie in [0, 1].")
    labels = predictions >= threshold
    tn, fp, fn, tp = confusion_matrix(target, labels, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(target, labels)),
        "precision": float(precision_score(target, labels, zero_division=0)),
        "recall": float(recall_score(target, labels, zero_division=0)),
        "f1": float(f1_score(target, labels, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def probability_metrics(target, predictions):
    """Reuse Step 4's probability metrics and add AP and threshold-0.5 results."""
    target = binary_target(target)
    predictions = check_predictions(predictions, len(target))
    return {
        **evaluate(target, predictions),
        "average_precision": float(average_precision_score(target, predictions)) if target.sum() else 0.0,
        **classification_metrics(target, predictions),
        "observations": len(target), "defaults": int(target.sum()),
    }


def decile_table(target, predictions):
    """Prediction-quantile diagnostics; ties can yield fewer than ten groups."""
    target = binary_target(target)
    predictions = check_predictions(predictions, len(target))
    groups = pd.qcut(predictions, q=10, labels=False, duplicates="drop")
    if np.isnan(groups).all():
        groups = np.zeros(len(predictions))
    frame = pd.DataFrame({"decile": np.asarray(groups, dtype=int) + 1,
                          TARGET: target, "predicted_pd": predictions})
    result = frame.groupby("decile").agg(
        customers=(TARGET, "size"), defaults=(TARGET, "sum"),
        minimum_pd=("predicted_pd", "min"), maximum_pd=("predicted_pd", "max"),
        mean_predicted_pd=("predicted_pd", "mean"), observed_default_rate=(TARGET, "mean"),
    ).reset_index()
    result["defaults"] = result["defaults"].astype(int)
    result["calibration_gap"] = result.mean_predicted_pd - result.observed_default_rate
    return result


def plot_results(figures_dir, context, validation_predictions, test_predictions,
                 validation_metrics, test_metrics, coefficients):
    """Draw publication-size PNG and SVG figures, explicitly identifying each split."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_ids = context["test_model_ids"]
    names = context["model_names"]
    selected = context["selected_model_id"]
    colors = dict(zip(model_ids, ["#64748b", "#10928b", "#cc8422", "#9c77ad", "#3d82b4", "#bd4c48", "#192b55"]))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures_dir / f"step_05_{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / f"step_05_{name}.svg", bbox_inches="tight")
        plt.close(fig)

    style = {"axes.spines.top": False, "axes.spines.right": False,
             "font.size": 10, "axes.titlesize": 13, "legend.fontsize": 8}
    with plt.rc_context(style):
        for split, predictions, metrics in [
            ("validation", validation_predictions, validation_metrics),
            ("test", test_predictions, test_metrics),
        ]:
            target = predictions[TARGET].to_numpy()
            scores = metrics.set_index("model_id")
            fig, ax = plt.subplots(figsize=(8, 6.5))
            ax.plot([0, 1], [0, 1], "--", color="#b0b0b0", label="Random ranking")
            for model_id in model_ids:
                fpr, tpr, _ = roc_curve(target, predictions[model_id])
                ax.plot(fpr, tpr, color=colors[model_id],
                        linewidth=2.5 if model_id == selected else 1.4,
                        label=f"{names[model_id]} ({scores.loc[model_id, 'roc_auc']:.3f})")
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"{split.title()} ROC curves (legend: ROC-AUC)", xlim=(0, 1), ylim=(0, 1.01))
            ax.legend(loc="lower right")
            ax.grid(alpha=0.15)
            save(fig, f"{split}_roc")

            fig, ax = plt.subplots(figsize=(8, 6.5))
            ax.axhline(target.mean(), linestyle="--", color="#b0b0b0", label=f"Prevalence ({target.mean():.3f})")
            for model_id in model_ids:
                precision, recall, _ = precision_recall_curve(target, predictions[model_id])
                ax.step(recall, precision, where="post", color=colors[model_id],
                        linewidth=2.2 if model_id == selected else 1.2,
                        label=f"{names[model_id]} ({scores.loc[model_id, 'average_precision']:.3f})")
            ax.set(xlabel="Recall", ylabel="Precision", title=f"{split.title()} precision-recall curves (legend: AP)",
                   xlim=(0, 1), ylim=(0, 1.01))
            ax.legend(loc="upper right")
            ax.grid(alpha=0.15)
            save(fig, f"{split}_precision_recall")

            fig, ax = plt.subplots(figsize=(8, 6.5))
            ax.plot([0, 1], [0, 1], "--", color="#b0b0b0", label="Perfect calibration")
            for model_id in model_ids:
                table = probability_groups(target, predictions[model_id], CALIBRATION_EDGES)
                ax.plot(table.mean_predicted_pd, table.observed_default_rate, "o-", markersize=4,
                        color=colors[model_id], linewidth=2.2 if model_id == selected else 1.2,
                        label=names[model_id])
            ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate",
                   title=f"{split.title()} calibration (10 fixed probability bins)", xlim=(0, 1), ylim=(0, 1))
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(loc="upper left")
            ax.grid(alpha=0.15)
            save(fig, f"{split}_calibration")

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            labels = [names[model_id] for model_id in model_ids]
            for ax, metric, label in zip(axes, ["brier_score", "log_loss"], ["Brier score", "Log loss"]):
                values = scores.loc[model_ids, metric]
                bars = ax.barh(labels, values, color=[colors[value] for value in model_ids])
                ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
                ax.invert_yaxis()
                ax.set(xlabel=f"{label} (lower is better)", xlim=(0, max(values) * 1.18))
                ax.grid(axis="x", alpha=0.15)
            fig.suptitle(f"{split.title()} probability accuracy")
            save(fig, f"{split}_probability_scores")

        target = test_predictions[TARGET].to_numpy()
        pd_values = test_predictions[selected].to_numpy()
        thresholds = [0.5, context["alternative_threshold"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        for ax, threshold in zip(axes, thresholds):
            matrix = confusion_matrix(target, pd_values >= threshold, labels=[0, 1])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=14,
                        color="white" if value > matrix.max() / 2 else "#172438")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
                   yticklabels=["No default", "Default"], xlabel="Predicted class", ylabel="Observed class",
                   title=f"Test: threshold {threshold:.2f}" + ("\n(validation F1 choice)" if threshold != 0.5 else ""))
        save(fig, "test_confusion_matrix")

        top = coefficients.head(16).sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.barh(top.feature, top.coefficient, color=np.where(top.coefficient >= 0, "#bd4c48", "#228275"))
        ax.axvline(0, color="#64748b", linewidth=0.8)
        ax.set(xlabel="Underlying logistic coefficient (numeric: per training SD)",
               title="Largest absolute logistic coefficients\nCategorical effects are relative to each dropped reference")
        ax.grid(axis="x", alpha=0.15)
        save(fig, "logistic_coefficients")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for value, color, label in [(0, "#228275", "Observed no default"), (1, "#bd4c48", "Observed default")]:
            ax.hist(pd_values[target == value], bins=np.linspace(0, 1, 31), density=True,
                    alpha=0.5, color=color, label=f"{label} (n={(target == value).sum():,})")
        ax.set(xlabel="Selected model predicted PD", ylabel="Density within observed class",
               title="Held-out test PD distributions", xlim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        save(fig, "test_pd_distribution")


def write_report(report_dir, context, validation_metrics, test_metrics, coefficients,
                 secondary_classification, calibration_study, training_audit, endpoints):
    selected_id = context["selected_model_id"]
    selected = test_metrics.set_index("model_id").loc[selected_id]
    historical_id = context["historical_reference_id"]
    historical = test_metrics.set_index("model_id").loc[historical_id]
    validation_selected = validation_metrics.set_index("model_id").loc[selected_id]
    validation_historical = validation_metrics.set_index("model_id").loc[historical_id]
    original = validation_metrics.set_index("model_id").loc["optimized_original"]
    engineered = validation_metrics.set_index("model_id").loc["optimized_engineered"]
    names = context["model_names"]
    selected_endpoints = endpoints.loc[
        (endpoints.split == "test") & (endpoints.model_id == selected_id)
    ].iloc[0]
    uncalibrated = test_metrics.set_index("model_id").loc[context["selected_base_model_id"]]
    coefficient_notes = []
    for feature, description in [
        ("max_delay", "greater worst recorded repayment delay"),
        ("payment_to_previous_bills_5m", "greater payment coverage of prior positive bills"),
        ("utilization_recent", "greater recent credit utilization"),
        ("LIMIT_BAL", "a larger credit limit"),
    ]:
        match = coefficients.loc[coefficients.feature == feature]
        if len(match):
            row = match.iloc[0]
            coefficient_notes.append(
                f"For {description}, the underlying conditional odds ratio is "
                f"{row.odds_ratio:.3f} per training SD ({row.training_standard_deviation:.3f} in feature units)."
            )
    interpretation = (
        f"Relative to {names[historical_id]}, the selected model has "
        f"{'higher' if selected.roc_auc > historical.roc_auc else 'lower'} ROC-AUC, "
        f"{'lower' if selected.brier_score < historical.brier_score else 'higher'} Brier score, and "
        f"{'lower' if selected.log_loss < historical.log_loss else 'higher'} log loss. "
        "The evidence does not establish superiority on every probability-quality measure."
    )

    def comparison(frame):
        columns = ["model_id", "roc_auc", "brier_score", "log_loss", "average_precision", "ece_10_bins"]
        table = frame[columns].copy()
        table["model_id"] = table.model_id.map(names)
        return markdown_table(table.rename(columns={"model_id": "Model"}))

    delta = pd.DataFrame([
        {"Split": split, "ROC-AUC difference": lr.roc_auc - hist.roc_auc,
         "Brier difference": lr.brier_score - hist.brier_score,
         "Log-loss difference": lr.log_loss - hist.log_loss}
        for split, lr, hist in [("Validation", validation_selected, validation_historical), ("Test", selected, historical)]
    ])
    top = coefficients.head(12)[["feature", "coefficient", "odds_ratio", "direction"]]
    cv_rows = pd.DataFrame(context["cv_best"])
    secondary = secondary_classification[["split", "threshold", "accuracy", "precision", "recall", "f1",
                                           "true_negatives", "false_positives", "false_negatives", "true_positives"]]
    report = f"""# Step 5: Logistic regression probability of default

## Executive summary and objective

The selected PD model is **{names[selected_id]}**, with base configuration
`{context['selected_base_model_id']}` and calibration **{context['calibration_method']}**.
On the untouched final comparison rows it has ROC-AUC **{selected.roc_auc:.6f}**,
Brier score **{selected.brier_score:.6f}**, log loss **{selected.log_loss:.6f}**,
and average precision **{selected.average_precision:.6f}**.
The goal is individualized next-month default probabilities from financial and
repayment history, compared with the three frozen Step 4 estimators. Conclusions
below use measured differences rather than assuming logistic regression is better.
{interpretation}

## Data, preprocessing, and feature design

Existing stratified splits were reused: 21,000 training, 4,500 validation, and 4,500
test rows, with default rates 22.1190%, 22.1111%, and 22.1333%. No rows were dropped,
deduplicated, or reassigned. File SHA-256 hashes match Step 4. The original code,
processed CSVs, and all Step 4 reports/models are unchanged. No new data download
was needed. Customer IDs are absent; row keys combine the input CSV hash with its
zero-based row number for an auditable common evaluation order.

The original-feature model uses 19 predictors: LIMIT_BAL, six BILL_AMT variables,
six PAY_AMT variables, and six repayment-status codes. SEX, AGE, MARRIAGE, and
EDUCATION are excluded. This narrows the comparison to financial behavior; it does
not establish demographic fairness or eliminate demographic proxies.

Each pipeline selects financial columns, creates any deterministic engineered
features, standardizes numeric variables, and one-hot encodes repayment statuses.
Categories seen fewer than 20 times in each fitting subset are pooled. The first
category is dropped (the all-training reference is -2 for each status). Unknown
codes use the learned infrequent category when available; otherwise sklearn uses
the all-zero reference encoding and may warn. Missing/nonfinite required values,
nonpositive limits, negative payments, and noninteger codes are rejected explicitly.
No scaler, category list, or rare-category grouping is fitted outside training or
the corresponding training-CV fold.

Eight engineered features add recent and mean signed utilization, recent and
five-month payment coverage, an indicator of no positive previous bill, the count
of positive delay codes, maximum positive delay, and recent balance change divided
by credit limit. Recent payment coverage uses September payment / positive August
bill; five-month coverage pairs September-May payments with August-April bills.
These are approximate coverage proxies, not contractual payment rates. Coverage
is 0 if its denominator is nonpositive, capped at 5; the separate indicator helps
distinguish the recent no-positive-bill case. Utilization is capped at [-5, 10],
and normalized balance change at [-10, 10], using fixed rules. All inputs predate
the next-month target. Average balance and raw balance change are linear combinations
of original bills; their normalized versions provide more useful nonlinear inputs.
The month mapping follows the [UCI dataset description](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Training bill audit:

{markdown_table(training_audit)}

Negative bills are retained in original features and signed utilization. Credit
balances can arise when credits exceed what is owed, as explained by the
[CFPB](https://www.consumerfinance.gov/ask-cfpb/what-is-a-credit-balance-on-my-credit-card-bill-en-42/).
This is a plausible interpretation, not verified account-level provenance for
these UCI records. No negative bill is automatically labeled corrupt or deleted.

## Logistic methodology and optimization

PD is `sigmoid(intercept + sum(coefficient * transformed_feature))`, obtained
through `predict_proba()[:, 1]`. Initial logistic regression uses L2, C=1,
lbfgs, max_iter=2000, random_state=42. Optimization compares L1 and L2 with
liblinear, C in {context['C_grid']}, five stratified training folds shuffled with
seed 42, and refits each family's best mean CV log-loss configuration on all
training rows. Brier, ROC-AUC, and AP are also recorded. The liblinear intercept
uses intercept_scaling=10 and is regularized through its synthetic feature.
Training-CV pipelines refit preprocessing inside every fold. Original and engineered
families use the identical fold assignment, saved in the CV membership table.

{markdown_table(cv_rows)}

The installed scikit-learn uses `l1_ratio=0` for L2 and `l1_ratio=1` for L1;
the deprecated penalty argument is avoided, following the
[official LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
ConvergenceWarning is treated as an error, with one retry at 8,000 iterations;
other warnings stay visible. The CV run emitted expected unknown-category warnings
when singleton repayment codes were absent from a fold's fitting rows; the encoder
mapped them to its learned infrequent group. No convergence failure occurred.
No class weighting or resampling was applied, since
probability estimation under the observed prevalence is the primary objective.

Training CV chooses regularization within each feature family. Validation log loss,
then Brier, chooses the feature family. Calibration is investigated on that chosen
base model using sigmoid and isotonic, each fitted with five-fold out-of-fold
training predictions and ensemble=False, then evaluated on validation. The full
pipeline is inside [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html),
so calibration folds also fit their own preprocessing. The resulting base estimator
is fitted on all training rows, without adding validation rows. Calibration is
adopted only if both validation log loss and Brier improve over the uncalibrated
base. Hyperparameter and calibration choices are not selected from test results.

## Validation results and feature engineering

{comparison(validation_metrics)}

Engineered minus original optimized validation differences: ROC-AUC
{engineered.roc_auc - original.roc_auc:+.6f}, Brier
{engineered.brier_score - original.brier_score:+.6f}, and log loss
{engineered.log_loss - original.log_loss:+.6f}. Positive AUC and negative error
differences favor engineering. These measurements, rather than an assumption about
feature engineering, drive the recorded choice. The initial model remains a
benchmark rather than a member of the optimized-family selection pool.

Calibration study (all fits use training data; scores below use validation):

{markdown_table(calibration_study)}

## Frozen final test comparison

The final configuration, both thresholds, retained comparators, input hashes, and
serialized model hashes were written to `models/step_05_selection.json` **before
loading test rows**. Test models are the three frozen historical estimators,
initial LR, both optimized feature families, and the selected calibrated variant
only if calibration was adopted. Discarded calibrators were not tested. All use
the same 4,500 rows in the same order. Baselines are loaded from Step 4 JSON files,
never refitted. Their validation predictions and metrics were checked against
the saved Step 4 results before proceeding.

{comparison(test_metrics)}

Differences below are selected logistic minus the Step 4 selected segmented
baseline, **{names[historical_id]}**. Positive AUC and negative errors favor LR.

{markdown_table(delta)}

These are point estimates from one held-out comparison, without significance
testing. Relative to the selected historical baseline, test Brier changes by
{100 * (selected.brier_score / historical.brier_score - 1):+.2f}% and log loss by
{100 * (selected.log_loss / historical.log_loss - 1):+.2f}%. Practical value depends
on error reduction, calibration, operating costs, and stability; ranking alone
does not establish economic value. There is no cost/recovery model in Step 5.

## Probability calibration and classification

Isotonic calibration improved both validation errors, but its test log loss is
{selected.log_loss:.6f}, versus {uncalibrated.log_loss:.6f} for the same uncalibrated
base model. The selected model assigns exactly zero PD to
{int(selected_endpoints.zero_pd_customers)} test customers, including
{int(selected_endpoints.defaults_at_zero_pd)} observed defaulters; it assigns exactly
one PD to {int(selected_endpoints.one_pd_customers)} customers, including
{int(selected_endpoints.nondefaults_at_one_pd)} observed nondefaulters.
These endpoint errors show why a small number of overconfident predictions can
worsen log loss while Brier score improves. Log loss uses the unchanged Step 4
numerical clipping rule [1e-15, 1 - 1e-15] for every model; saved predictions retain
their original probabilities. The endpoint diagnostic is descriptive and did not
trigger clipping changes, recalibration, or model replacement after test inspection.
The selected artifact preserves the validation-based choice. The uncalibrated
optimized model remains an explicitly reported comparator, with no retrospective
claim that it was the frozen final selection.

The selected model's validation ECE is {validation_selected.ece_10_bins:.6f}; its
test ECE is {selected.ece_10_bins:.6f}, compared with {historical.ece_10_bins:.6f}
for the selected historical baseline. ECE uses the same 10 fixed probability bins
as Step 4, weighted by customer counts. It depends on bin definitions. Signed
aggregate bias and customer counts accompany it in the CSVs. Predicted-PD deciles
are also saved; ties can yield fewer than ten groups. Calibration is a diagnostic,
not a guarantee of individual true probabilities or temporal stability.

AUC measures ranking and is unchanged by strictly increasing score transformations;
those transformations can substantially change probability levels and calibration.
Consequently a strong AUC can coexist with poor PD estimates. Average precision
is the step-weighted summary from sklearn, not trapezoidal PR area; its no-skill
reference is the split's default prevalence.

Secondary classification uses threshold 0.5. A separate threshold
**{context['alternative_threshold']:.2f}** maximizes validation F1 over a fixed grid
0.05 to 0.95 in steps of 0.05, breaking ties by precision then lower threshold.
It was frozen before test evaluation and does not change any PD quality metric.
It is an illustrative classification operating point, not a cost-optimal credit
approval or collections rule.

{markdown_table(secondary)}

## Coefficients and odds ratios

The full coefficient table, intercept, dropped references, and numeric training
means/standard deviations are saved. The largest absolute underlying coefficients:

{markdown_table(top)}

Odds ratio is `exp(coefficient)`. For numeric features it represents one training
standard-deviation increase, holding all other transformed features fixed. For
one-hot categories it compares the displayed category with that month's dropped
reference; `infrequent_sklearn` pools rare codes. Standardized indicators also
use one SD, not a 0-to-1 change. Signs describe conditional association, not causal
effects. Positive delayed-payment category effects indicate higher default odds
relative to the reference. Financial coverage/utilization coefficients describe
conditional associations after accounting for the other financial history.
{' '.join(coefficient_notes)}
Changing a raw input can also change its engineered summaries, so one coefficient
alone does not measure the full prediction change in such a scenario.
Correlated monthly bills and engineered summaries can make individual signs unstable.
If calibration is selected, these are the underlying logistic score's odds ratios;
a nonlinear calibrator means they are not the final calibrated probability odds ratios.

## Figures and saved artifacts

Figures are provided in PNG (200 dpi) and SVG. Validation and test views are named
explicitly; colors are consistent across the retained comparison models.

![Test ROC](figures/step_05_test_roc.png)
![Test precision-recall](figures/step_05_test_precision_recall.png)
![Test reliability](figures/step_05_test_calibration.png)
![Test confusion matrices](figures/step_05_test_confusion_matrix.png)
![Logistic coefficients](figures/step_05_logistic_coefficients.png)
![Test PD distributions](figures/step_05_test_pd_distribution.png)
![Test probability accuracy](figures/step_05_test_probability_scores.png)

The corresponding validation ROC, precision-recall, calibration, and score figures
are also under `figures/step_05_validation_*`. CSVs include all CV scores, candidate
validation results, final test results, aligned predictions, fixed-bin calibration,
deciles, coefficients, classification thresholds, and training bill diagnostics.
`step_05_run_manifest.json` identifies every created artifact and preserved input.

## Reproduction, limitations, and preparation for Step 6

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe run_step_05.py
```

Completed runs return saved results instead of repeating model selection or test
scoring. Frozen runs can resume output generation from saved predictions. To use
the final artifact from the project environment:

```python
import joblib
from logistic_pd import predict_pd

model = joblib.load("models/logistic_pd_pipeline.joblib")
pd_values = predict_pd(model, new_customers)  # preserves customer index/order
```

Reloaded predictions were checked for exact agreement. Use the recorded package
versions and keep `logistic_pd.py` importable, since joblib stores its transformer
class by module name. The tests check probability validity, record alignment,
demographic exclusion, missing/zero-bill handling, fold-specific preprocessing,
split-hash verification, common evaluation rows, and serialization.

The UCI label is next-month payment default in this historical Taiwan cohort,
not a demonstrated regulatory-default event. The
[Basel CRE36.69 reference definition](https://www.bis.org/basel_framework/chapter/CRE/36.htm?inforce=20191215&published=20191215&tldate=20050625)
uses unlikeliness to pay or material arrears beyond 90 days, with specified
supervisory provisions. No mapping from these labels to that definition, or from
one-month to one-year/lifetime PD, is established here.

The random split cannot measure time drift. Step 4 identified nine validation
records matching training across all predictors; identities cannot be established
without customer IDs. Those overlaps and possible similar records in CV/test
can make performance optimistic. Validation is used for feature/calibration/threshold
selection and is not an unbiased final assessment. Test results do not trigger
retuning. No confidence intervals, fairness certification, or causal conclusions
are supplied. Demographic exclusion alone does not prevent proxy effects.

Step 6 can consume the frozen, indexed PD predictions and this model's provenance.
Any synthetic recovery data must be clearly labeled synthetic and must not be
presented as observed UCI recoveries. No recovery dataset, LGD model, or ECL
calculation is implemented in this step.
"""
    (report_dir / "step_05_logistic_regression_pd.md").write_text(report, encoding="utf-8")
