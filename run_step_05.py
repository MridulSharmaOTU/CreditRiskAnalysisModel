"""Train, freeze, and evaluate Step 5 without changing the existing data or baselines.

Run once with the existing project environment. Completed runs return saved test
metrics. --render-only rebuilds figures/reports from saved predictions without
refitting or rerunning test inference. A frozen incomplete run resumes its final
evaluation/output phase, retaining its recorded model choices.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from historical_pd import TARGET, binary_target, check_predictions, load_model
from logistic_pd import (
    BILL_FEATURES, ENGINEERED_FEATURES, EXCLUDED_DEMOGRAPHICS, RAW_FEATURES,
    coefficient_table, fit_with_convergence_check, make_pipeline, predict_pd,
    underlying_pipeline,
)
from run_step_04 import CALIBRATION_EDGES, evaluate, file_hash, probability_groups
from step_05_reporting import (
    classification_metrics, decile_table, plot_results, probability_metrics, write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_FOLDS = 5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_split_inputs(data_dir, step_04_dir):
    """Reject changed or accidentally substituted splits before any model fitting."""
    step_04 = json.loads((step_04_dir / "step_04_run_manifest.json").read_text(encoding="utf-8"))
    paths = {name: (data_dir / f"{name}.csv").resolve() for name in ["train", "validation", "test"]}
    if len(set(paths.values())) != 3:
        raise ValueError("Training, validation, and test must be distinct input paths.")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if len(set(hashes.values())) != 3:
        raise ValueError("Two dataset files are identical; check split assignment.")
    for name, digest in hashes.items():
        if digest != step_04["inputs"][name]["sha256"]:
            raise ValueError(f"{name} no longer matches the frozen Step 4 split.")
    return paths, hashes, step_04


def make_prediction_frame(target, predictions, input_hash):
    """Use a single ordered target and enforce one prediction per evaluation row."""
    target = binary_target(target)
    frame = pd.DataFrame({
        "row_number": np.arange(len(target)),
        "row_key": [f"{input_hash[:16]}:{index}" for index in range(len(target))],
        TARGET: target.astype(int),
    })
    for model_id, values in predictions.items():
        frame[model_id] = check_predictions(values, len(frame))
    if not frame.row_key.is_unique:
        raise ValueError("Evaluation row keys are not unique.")
    return frame


def score_models(target, predictions, model_names):
    return pd.DataFrame([
        {"model_id": model_id, "model": model_names[model_id], **probability_metrics(target, values)}
        for model_id, values in predictions.items()
    ])


def assert_hashes(expected):
    for path, digest in expected.items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError(f"A preserved/frozen input changed: {path}")


def prepare_and_freeze(paths, hashes, step_04, step_04_dir, report_dir, models_dir):
    """Fit/select from training and validation, without opening the test CSV."""
    tables = report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    original_artifacts = [PROJECT_ROOT / name for name in [
        "main.py", "historical_pd.py", "run_step_04.py", "tests/test_historical_pd.py"
    ]]
    original_artifacts += [path for path in step_04_dir.rglob("*") if path.is_file() and (
        path.name.startswith(("step_04", "pay_0")) or path.parent == step_04_dir / "models"
    )]
    protected = {str(path.resolve()): file_hash(path) for path in [*paths.values(), *original_artifacts]}
    train = pd.read_csv(paths["train"])
    validation = pd.read_csv(paths["validation"])
    train_y = binary_target(train[TARGET], len(train))
    validation_y = binary_target(validation[TARGET], len(validation))
    for name, frame, target in [("train", train, train_y), ("validation", validation, validation_y)]:
        expected = step_04["split_summary"][name]
        if len(frame) != expected["customers"] or int(target.sum()) != expected["defaults"]:
            raise ValueError(f"Unexpected {name} size/target alignment.")
    train_x, validation_x = train.drop(columns=TARGET), validation.drop(columns=TARGET)
    if set(train_x.columns) != set(validation_x.columns):
        raise ValueError("Training and validation predictor columns differ.")
    selection = json.loads((step_04_dir / "models/selection.json").read_text(encoding="utf-8"))
    historical_ids = selection["retained_models"]
    models, predictions, model_names, model_paths = {}, {}, {}, {}
    for model_id in historical_ids:
        path = step_04_dir / "models" / f"{model_id}.json"
        model = load_model(path)
        if model.training_size_ != len(train) or model.training_defaults_ != int(train_y.sum()):
            raise ValueError("Historical estimator was not fitted on the expected training split.")
        models[model_id] = model
        predictions[model_id] = model.predict(validation_x)
        model_paths[model_id] = str(path.resolve())
        model_names[model_id] = (
            "Historical PAY_0 + limit" if getattr(model, "use_credit_limit", False)
            else "Historical PAY_0" if hasattr(model, "status_rates_") else "Historical portfolio"
        )
    saved_predictions = pd.read_csv(step_04_dir / "tables/step_04_validation_predictions.csv")
    if not np.array_equal(saved_predictions[TARGET], validation_y) or not np.array_equal(
        saved_predictions.row_number, np.arange(len(validation))
    ):
        raise ValueError("Step 4 validation observation order/labels differ.")
    saved_metrics = pd.read_csv(step_04_dir / "tables/step_04_validation_metrics.csv").set_index("model_id")
    for model_id in historical_ids:
        np.testing.assert_allclose(predictions[model_id], saved_predictions[model_id], rtol=0, atol=1e-14)
        current = evaluate(validation_y, predictions[model_id])
        for metric in ["roc_auc", "brier_score", "log_loss"]:
            np.testing.assert_allclose(current[metric], saved_metrics.loc[model_id, metric], rtol=0, atol=1e-14)
    print("Frozen Step 4 validation predictions and metrics reproduced.", flush=True)

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(train_x, train_y))
    membership = np.full(len(train), -1, dtype=int)
    for fold, (fit_rows, held_out_rows) in enumerate(folds):
        if np.intersect1d(fit_rows, held_out_rows).size:
            raise RuntimeError("Training CV contains overlapping fit and assessment indices.")
        membership[held_out_rows] = fold
    if (membership < 0).any():
        raise RuntimeError("Training CV missed customer rows.")
    pd.DataFrame({"training_row_number": np.arange(len(train)), "held_out_cv_fold": membership}).to_csv(
        tables / "step_05_cv_membership.csv", index=False
    )

    print("Fitting initial L2 logistic regression (C=1, lbfgs).", flush=True)
    models["initial_logistic"] = fit_with_convergence_check(make_pipeline(), train_x, train_y)
    predictions["initial_logistic"] = predict_pd(models["initial_logistic"], validation_x).to_numpy()
    model_names["initial_logistic"] = "Initial LR"
    cv_tables, cv_best = [], []
    for engineered in [False, True]:
        family = "engineered" if engineered else "original"
        model_id = f"optimized_{family}"
        print(f"Training-CV search: {family}, 12 configurations x {N_FOLDS} folds.", flush=True)
        search = GridSearchCV(
            make_pipeline(engineered=engineered, solver="liblinear"),
            {"classifier__C": C_GRID, "classifier__l1_ratio": [0.0, 1.0]},
            scoring={"log_loss": "neg_log_loss", "brier": "neg_brier_score",
                     "roc_auc": "roc_auc", "average_precision": "average_precision"},
            refit="log_loss", cv=folds, n_jobs=1, error_score="raise",
        )
        fit_with_convergence_check(search, train_x, train_y, parameter_prefix="estimator__")
        models[model_id] = search.best_estimator_
        predictions[model_id] = predict_pd(models[model_id], validation_x).to_numpy()
        model_names[model_id] = f"Optimized LR ({family})"
        results = pd.DataFrame(search.cv_results_)
        results.insert(0, "feature_family", family)
        # Preserve all fold metrics as well as readable positive CV errors.
        results["mean_cv_log_loss"] = -results["mean_test_log_loss"]
        results["mean_cv_brier"] = -results["mean_test_brier"]
        cv_tables.append(results.drop(columns="params"))
        best = results.iloc[search.best_index_]
        cv_best.append({
            "feature_family": family,
            "C": float(search.best_params_["classifier__C"]),
            "regularization": "L1" if search.best_params_["classifier__l1_ratio"] else "L2",
            "mean_cv_log_loss": float(best.mean_cv_log_loss),
            "sd_cv_log_loss": float(best.std_test_log_loss),
            "mean_cv_brier": float(best.mean_cv_brier),
        })
        print(f"Best {family}: {cv_best[-1]}", flush=True)
    pd.concat(cv_tables, ignore_index=True).to_csv(tables / "step_05_cv_results.csv", index=False)

    preliminary = score_models(validation_y, predictions, model_names).set_index("model_id")
    base_id = min(["optimized_original", "optimized_engineered"], key=lambda name: (
        preliminary.loc[name, "log_loss"], preliminary.loc[name, "brier_score"], name
    ))
    print(f"Validation selected feature family: {base_id}. Investigating training-CV calibration.", flush=True)
    calibration_rows = [{"method": "none", "model_id": base_id, **evaluate(validation_y, predictions[base_id])}]
    for method in ["sigmoid", "isotonic"]:
        model_id = f"calibrated_{method}"
        calibrator = CalibratedClassifierCV(clone(models[base_id]), method=method, cv=folds, ensemble=False, n_jobs=1)
        models[model_id] = fit_with_convergence_check(calibrator, train_x, train_y, parameter_prefix="estimator__")
        predictions[model_id] = predict_pd(models[model_id], validation_x).to_numpy()
        model_names[model_id] = f"Calibrated LR ({method})"
        calibration_rows.append({"method": method, "model_id": model_id, **evaluate(validation_y, predictions[model_id])})
    calibration_study = pd.DataFrame(calibration_rows)
    base_result = calibration_study.iloc[0]
    eligible = calibration_study.loc[
        (calibration_study.log_loss < base_result.log_loss)
        & (calibration_study.brier_score < base_result.brier_score)
    ].sort_values(["log_loss", "brier_score", "method"])
    selected_id = eligible.iloc[0].model_id if len(eligible) else base_id
    calibration_method = eligible.iloc[0].method if len(eligible) else "none"
    calibration_study["selected"] = calibration_study.model_id == selected_id
    calibration_study.to_csv(tables / "step_05_calibration_study.csv", index=False)

    threshold_results = pd.DataFrame([
        classification_metrics(validation_y, predictions[selected_id], round(float(threshold), 2))
        for threshold in np.arange(0.05, 1.0, 0.05)
    ])
    threshold = float(threshold_results.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0].threshold)
    threshold_results["selected"] = threshold_results.threshold == threshold
    threshold_results.to_csv(tables / "step_05_validation_thresholds.csv", index=False)
    validation_metrics = score_models(validation_y, predictions, model_names)
    validation_metrics.to_csv(tables / "step_05_validation_metrics.csv", index=False)
    make_prediction_frame(validation_y, predictions, hashes["validation"]).to_csv(
        tables / "step_05_validation_predictions.csv", index=False
    )
    coefficients = coefficient_table(models[selected_id])
    coefficients.to_csv(tables / "step_05_coefficients.csv", index=False)
    audit = pd.DataFrame([
        {"feature": feature, "negative_bills": int((train[feature] < 0).sum()),
         "zero_bills": int((train[feature] == 0).sum()),
         "minimum": float(train[feature].min()), "maximum": float(train[feature].max())}
        for feature in BILL_FEATURES
    ])
    audit.to_csv(tables / "step_05_training_bill_audit.csv", index=False)

    for model_id, filename in [
        ("initial_logistic", "logistic_pd_initial.joblib"),
        ("optimized_original", "logistic_pd_original.joblib"),
        ("optimized_engineered", "logistic_pd_engineered.joblib"),
    ]:
        path = models_dir / filename
        joblib.dump(models[model_id], path)
        np.testing.assert_array_equal(predict_pd(joblib.load(path), validation_x), predictions[model_id])
        model_paths[model_id] = str(path.resolve())
    final_path = models_dir / "logistic_pd_pipeline.joblib"
    joblib.dump(models[selected_id], final_path)
    np.testing.assert_array_equal(predict_pd(joblib.load(final_path), validation_x), predictions[selected_id])
    model_paths[selected_id] = str(final_path.resolve())
    test_ids = [*historical_ids, "initial_logistic", "optimized_original", "optimized_engineered"]
    if selected_id not in test_ids:
        test_ids.append(selected_id)
    pipeline = underlying_pipeline(models[selected_id])
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["status"]
    context = {
        "step": 5,
        "selected_model_id": selected_id, "selected_base_model_id": base_id,
        "calibration_method": calibration_method, "alternative_threshold": threshold,
        "default_threshold": 0.5,
        "historical_reference_id": selection["selected_segmented_model"],
        "model_names": model_names, "test_model_ids": test_ids,
        "model_paths": model_paths,
        "dataset_paths": {name: str(path) for name, path in paths.items()},
        "dataset_hashes": hashes, "protected_files": protected,
        "cv_best": cv_best, "C_grid": C_GRID, "cv_folds": N_FOLDS, "random_state": 42,
        "selection_rule": "Training CV log loss within family; validation log loss then Brier across families; calibration requires both errors to improve.",
        "raw_features": RAW_FEATURES, "engineered_features": ENGINEERED_FEATURES,
        "excluded_demographics": EXCLUDED_DEMOGRAPHICS,
        "logistic_intercept": float(pipeline.named_steps["classifier"].intercept_[0]),
        "logistic_iterations": [int(value) for value in pipeline.named_steps["classifier"].n_iter_],
        "categorical_references": {
            name: float(encoder.categories_[i][encoder.drop_idx_[i]])
            for i, name in enumerate(pipeline.named_steps["preprocessor"].transformers_[1][2])
        },
        "selected_fit_rows": pipeline.named_steps["features"].n_samples_seen_,
        "versions": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "matplotlib", "joblib"
        ]},
    }
    assert_hashes(protected)
    context["frozen_model_hashes"] = {path: file_hash(Path(path)) for path in model_paths.values()}
    context["logistic_source_sha256"] = file_hash(PROJECT_ROOT / "logistic_pd.py")
    write_json(report_dir / "step_05_context.json", context)
    # This file is the freeze boundary. Test feature/target data has not been opened.
    write_json(models_dir / "step_05_selection.json", context)
    print(f"FROZEN: {selected_id}, calibration={calibration_method}, validation F1 threshold={threshold:.2f}.", flush=True)
    return context


def evaluate_frozen_test(context, report_dir, selection_path):
    """One test inference/scoring pass, after the on-disk freeze boundary."""
    if not selection_path.is_file():
        raise FileNotFoundError("Freeze the selected configuration before loading test data.")
    if json.loads(selection_path.read_text(encoding="utf-8")) != context:
        raise ValueError("In-memory selection differs from the frozen configuration.")
    marker_path = report_dir / "step_05_test_evaluation.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker["selection_sha256"] != file_hash(selection_path):
            raise RuntimeError("The model selection changed after the recorded test evaluation.")
        assert_hashes(marker["saved_evaluation_hashes"])
        print("Using saved final test evaluation; no additional test inference.", flush=True)
        return
    assert_hashes(context["protected_files"])
    assert_hashes(context["frozen_model_hashes"])
    if context["logistic_source_sha256"] != file_hash(PROJECT_ROOT / "logistic_pd.py"):
        raise RuntimeError("The logistic transformer changed after model selection was frozen.")
    test = pd.read_csv(context["dataset_paths"]["test"])
    target = binary_target(test[TARGET], len(test))
    if len(test) != 4500:
        raise ValueError("The frozen test split must contain 4,500 rows.")
    features = test.drop(columns=TARGET)
    predictions = {}
    for model_id in context["test_model_ids"]:
        path = Path(context["model_paths"][model_id])
        if path.suffix == ".json":
            predictions[model_id] = load_model(path).predict(features)
        else:
            predictions[model_id] = predict_pd(joblib.load(path), features).to_numpy()
    table = score_models(target, predictions, context["model_names"])
    tables = report_dir / "tables"
    metrics_path = tables / "step_05_test_metrics.csv"
    predictions_path = tables / "step_05_test_predictions.csv"
    table.to_csv(metrics_path, index=False)
    make_prediction_frame(target, predictions, context["dataset_hashes"]["test"]).to_csv(predictions_path, index=False)
    write_json(marker_path, {
        "test_evaluation_passes": 1, "observations": len(test),
        "selection_sha256": file_hash(selection_path),
        "test_sha256": context["dataset_hashes"]["test"],
        "saved_evaluation_hashes": {str(path.resolve()): file_hash(path) for path in [metrics_path, predictions_path]},
        "models": context["test_model_ids"],
    })
    print("Final test evaluation saved. No test-driven retuning will occur.", flush=True)


def render_outputs(context, report_dir, models_dir):
    """Regenerate derived outputs from saved predictions; no test data/model fitting."""
    tables = report_dir / "tables"
    validation_predictions = pd.read_csv(tables / "step_05_validation_predictions.csv")
    test_predictions = pd.read_csv(tables / "step_05_test_predictions.csv")
    validation_metrics = pd.read_csv(tables / "step_05_validation_metrics.csv")
    test_metrics = pd.read_csv(tables / "step_05_test_metrics.csv")
    coefficients = pd.read_csv(tables / "step_05_coefficients.csv")
    calibration_tables, deciles, classifications, endpoints = [], [], [], []
    for split, predictions in [("validation", validation_predictions), ("test", test_predictions)]:
        target = predictions[TARGET].to_numpy()
        for model_id in context["test_model_ids"]:
            values = predictions[model_id].to_numpy()
            endpoints.append({
                "split": split, "model_id": model_id,
                "zero_pd_customers": int((values == 0).sum()),
                "defaults_at_zero_pd": int(((values == 0) & (target == 1)).sum()),
                "one_pd_customers": int((values == 1).sum()),
                "nondefaults_at_one_pd": int(((values == 1) & (target == 0)).sum()),
            })
            calibration_tables.append(probability_groups(target, predictions[model_id], CALIBRATION_EDGES).assign(
                split=split, model_id=model_id
            ))
            deciles.append(decile_table(target, predictions[model_id]).assign(split=split, model_id=model_id))
        for threshold in [0.5, context["alternative_threshold"]]:
            classifications.append({"split": split, "model_id": context["selected_model_id"],
                                    **classification_metrics(target, predictions[context["selected_model_id"]], threshold)})
    pd.concat(calibration_tables, ignore_index=True).to_csv(tables / "step_05_calibration.csv", index=False)
    pd.concat(deciles, ignore_index=True).to_csv(tables / "step_05_pd_deciles.csv", index=False)
    classification = pd.DataFrame(classifications)
    classification.to_csv(tables / "step_05_selected_classification.csv", index=False)
    endpoint_table = pd.DataFrame(endpoints)
    endpoint_table.to_csv(tables / "step_05_probability_endpoints.csv", index=False)
    print("Rendering validation/test figures and the final report from saved predictions.", flush=True)
    plot_results(report_dir / "figures", context, validation_predictions, test_predictions,
                 validation_metrics, test_metrics, coefficients)
    write_report(report_dir, context, validation_metrics, test_metrics, coefficients, classification,
                 pd.read_csv(tables / "step_05_calibration_study.csv"),
                 pd.read_csv(tables / "step_05_training_bill_audit.csv"), endpoint_table)
    assert_hashes(context["protected_files"])
    assert_hashes(context["frozen_model_hashes"])
    outputs = [path for path in report_dir.rglob("*") if path.is_file() and path.name.startswith("step_05_")]
    outputs += [path for path in models_dir.iterdir() if path.is_file() and (
        path.name.startswith("logistic_pd_") or path.name == "step_05_selection.json"
    )]
    source_names = ["logistic_pd.py", "run_step_05.py", "step_05_reporting.py", "tests/test_logistic_pd.py"]
    manifest = {
        "step": 5, "input_files_unchanged": True, "step_04_artifacts_unchanged": True,
        "dataset_hashes": context["dataset_hashes"], "selected_model_id": context["selected_model_id"],
        "selected_base_model_id": context["selected_base_model_id"],
        "calibration_method": context["calibration_method"],
        "test_evaluation_passes": 1, "model_fit_rows": context["selected_fit_rows"],
        "validation_and_test_rows": [4500, 4500],
        "model_reload_predictions_exact": True,
        "historical_validation_predictions_reproduced": True,
        "versions": context["versions"],
        "created_files": sorted(set([str((PROJECT_ROOT / name).resolve()) for name in source_names] + [
            str(path.resolve()) for path in outputs
        ] + [str((report_dir / "step_05_run_manifest.json").resolve())])),
        "modified_files": [str((PROJECT_ROOT / "requirements.txt").resolve())],
    }
    write_json(report_dir / "step_05_run_manifest.json", manifest)
    return test_metrics


def run(data_dir, step_04_dir, report_dir, models_dir, render_only=False):
    data_dir, step_04_dir, report_dir, models_dir = [
        Path(path).resolve() for path in [data_dir, step_04_dir, report_dir, models_dir]
    ]
    selection_path = models_dir / "step_05_selection.json"
    completed = report_dir / "step_05_run_manifest.json"
    paths, hashes, step_04 = verify_split_inputs(data_dir, step_04_dir)
    if selection_path.exists():
        context = json.loads(selection_path.read_text(encoding="utf-8"))
        if hashes != context["dataset_hashes"]:
            raise ValueError("Input files no longer match the frozen Step 5 selection.")
        assert_hashes(context["protected_files"])
        assert_hashes(context["frozen_model_hashes"])
        if completed.exists() and not render_only:
            evaluate_frozen_test(context, report_dir, selection_path)
            print("Step 5 is complete; returning saved results without repeating test evaluation.", flush=True)
            return pd.read_csv(report_dir / "tables/step_05_test_metrics.csv")
    else:
        if render_only:
            raise FileNotFoundError("No frozen Step 5 run is available to render.")
        context = prepare_and_freeze(paths, hashes, step_04, step_04_dir, report_dir, models_dir)
    if render_only and not (report_dir / "step_05_test_evaluation.json").exists():
        raise FileNotFoundError("No saved final test evaluation is available to render.")
    evaluate_frozen_test(context, report_dir, selection_path)
    return render_outputs(context, report_dir, models_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--step-04-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    results = run(arguments.data_dir, arguments.step_04_dir, arguments.report_dir,
                  arguments.models_dir, arguments.render_only)
    print("\nFinal held-out test results:")
    print(results[["model_id", "roc_auc", "brier_score", "log_loss", "average_precision"]].to_string(index=False))
    print(f"\nReport: {arguments.report_dir.resolve() / 'step_05_logistic_regression_pd.md'}")


if __name__ == "__main__":
    main()
