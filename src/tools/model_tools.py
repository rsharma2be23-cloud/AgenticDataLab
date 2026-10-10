"""Deterministic model training, comparison, evaluation, tuning, and artifacts."""
import hashlib
import os
import pickle
import time
import uuid
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from core.ml_schemas import ExperimentResult


class ModelTools:
    """Candidate estimators share one preprocessing pipeline and evaluation contract."""
    CLASSIFICATION_MODELS = ("logreg", "rf", "hist_gradient_boosting", "svm", "knn")
    REGRESSION_MODELS = ("linear", "ridge", "rf", "gradient_boosting", "hist_gradient_boosting", "svr")
    MODEL_LABELS = {
        "logreg": "LogisticRegression", "rf": "RandomForest", "hist_gradient_boosting": "HistGradientBoosting",
        "svm": "SVM", "knn": "KNN", "linear": "LinearRegression", "ridge": "Ridge",
        "gradient_boosting": "GradientBoosting", "svr": "SVR",
    }
    MODEL_ALIASES = {"RandomForest": "rf", "LogisticRegression": "logreg", "HistGradientBoosting": "hist_gradient_boosting",
                     "SVM": "svm", "KNN": "knn", "LinearRegression": "linear", "Ridge": "ridge",
                     "GradientBoosting": "gradient_boosting", "SVR": "svr"}

    def __init__(self, output_dir="models", cv=3, primary_metric=None, random_state=42, max_tuning_trials=8):
        self.output_dir = output_dir
        self.cv = max(2, int(cv))
        self.primary_metric = primary_metric
        self.random_state = int(random_state)
        self.max_tuning_trials = min(10, max(1, int(max_tuning_trials)))
        os.makedirs(self.output_dir, exist_ok=True)

    def _detect_task(self, y):
        non_null = y.dropna()
        if not pd.api.types.is_numeric_dtype(non_null):
            return "classification"
        unique = int(non_null.nunique())
        threshold = min(20, max(2, int(np.sqrt(max(len(non_null), 1)))))
        return "classification" if unique <= threshold else "regression"

    @staticmethod
    def _one_hot_encoder():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _build_preprocessor(self, X):
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                        ("scale", StandardScaler())]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                            ("encode", self._one_hot_encoder())]), categorical))
        if not transformers:
            raise ValueError("No usable feature columns remain after removing the target.")
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    def _make_candidates(self, task, class_weight=None, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        if task == "classification":
            return {
                "logreg": LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state),
                "rf": RandomForestClassifier(n_estimators=120, class_weight=class_weight, random_state=random_state, n_jobs=1),
                "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
                "svm": SVC(probability=True, class_weight=class_weight, random_state=random_state),
                "knn": KNeighborsClassifier(n_neighbors=5),
            }
        return {
            "linear": LinearRegression(), "ridge": Ridge(),
            "rf": RandomForestRegressor(n_estimators=120, random_state=random_state, n_jobs=1),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
            "svr": SVR(),
        }

    @staticmethod
    def _is_imbalanced(y):
        proportions = y.value_counts(normalize=True, dropna=True)
        return bool(len(proportions) > 1 and float(proportions.iloc[0]) >= 0.8)

    def _choose_primary_metric(self, task, y, requested=None):
        metric = requested or self.primary_metric
        allowed = {"classification": {"f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"},
                   "regression": {"rmse", "mae", "r2"}}
        if metric and metric not in allowed[task]:
            raise ValueError(f"Primary metric '{metric}' is not valid for {task}.")
        if metric:
            return metric
        if task == "regression":
            return "rmse"
        if self._is_imbalanced(y):
            return "pr_auc" if y.nunique() == 2 else "f1"
        return "f1"

    @staticmethod
    def _scoring_name(task, metric, y, positive_label=None):
        if task == "regression":
            return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}[metric]
        if y.nunique() == 2 and metric in {"f1", "precision", "recall", "roc_auc", "pr_auc"}:
            def binary_scorer(estimator, X, y_true):
                if metric in {"roc_auc", "pr_auc"}:
                    if not hasattr(estimator, "predict_proba"):
                        return 0.0
                    classes = list(estimator.classes_)
                    pos_index = classes.index(positive_label)
                    scores = estimator.predict_proba(X)[:, pos_index]
                    binary_y = np.asarray(y_true) == positive_label
                    if len(np.unique(binary_y)) < 2:
                        return float("nan")
                    return float(roc_auc_score(binary_y, scores) if metric == "roc_auc" else average_precision_score(binary_y, scores))
                predictions = estimator.predict(X)
                metric_func = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]
                return float(metric_func(y_true, predictions, average="binary", pos_label=positive_label, zero_division=0))
            return binary_scorer
        if metric == "f1":
            return "f1_weighted"
        if metric == "precision":
            return "precision_weighted"
        if metric == "recall":
            return "recall_weighted"
        if metric == "accuracy":
            return "accuracy"
        if metric == "roc_auc":
            return "roc_auc" if y.nunique() == 2 else "roc_auc_ovr_weighted"
        if metric == "pr_auc":
            return "average_precision" if y.nunique() == 2 else "f1_weighted"
        return "f1_weighted"

    @staticmethod
    def _probability_metrics(estimator, X, y, task, positive_label=None):
        if task != "classification" or not hasattr(estimator, "predict_proba"):
            return {}
        probabilities = estimator.predict_proba(X)
        try:
            if probabilities.shape[1] == 2:
                positive_label = estimator.classes_[1] if positive_label is None else positive_label
                score = probabilities[:, list(estimator.classes_).index(positive_label)]
                binary_y = np.asarray(y) == positive_label
                return {"roc_auc": float(roc_auc_score(binary_y, score)),
                        "pr_auc": float(average_precision_score(binary_y, score))}
            return {"roc_auc": float(roc_auc_score(y, probabilities, multi_class="ovr", average="weighted")),
                    "pr_auc": float(average_precision_score(LabelBinarizer().fit_transform(y), probabilities, average="macro"))}
        except (ValueError, TypeError):
            return {}

    def _evaluate(self, estimator, X, y, task, positive_label=None):
        predictions = estimator.predict(X)
        if task == "classification":
            if y.nunique() == 2:
                positive_label = estimator.classes_[1] if positive_label is None else positive_label
                values = {"accuracy": float(accuracy_score(y, predictions)),
                          "precision": float(precision_score(y, predictions, average="binary", pos_label=positive_label, zero_division=0)),
                          "recall": float(recall_score(y, predictions, average="binary", pos_label=positive_label, zero_division=0)),
                          "f1": float(f1_score(y, predictions, average="binary", pos_label=positive_label, zero_division=0))}
            else:
                values = {"accuracy": float(accuracy_score(y, predictions)),
                          "precision": float(precision_score(y, predictions, average="weighted", zero_division=0)),
                          "recall": float(recall_score(y, predictions, average="weighted", zero_division=0)),
                          "f1": float(f1_score(y, predictions, average="weighted", zero_division=0))}
            values["f1_score"] = values["f1"]  # Phase 1 compatibility
            values.update(self._probability_metrics(estimator, X, y, task, positive_label=positive_label))
            return values
        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        return {"mae": float(mean_absolute_error(y, predictions)), "rmse": rmse,
                "mse": float(mean_squared_error(y, predictions)), "r2": float(r2_score(y, predictions))}

    def _folds(self, task, y, cv=None, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        folds = min(int(cv or self.cv), len(y))
        if task == "classification":
            class_minimum = int(y.value_counts().min())
            folds = min(folds, class_minimum)
            if folds < 2:
                raise ValueError("At least two examples per class are required for stratified cross-validation.")
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        if folds < 2:
            raise ValueError("At least two rows are required for cross-validation.")
        return KFold(n_splits=folds, shuffle=True, random_state=random_state)

    @staticmethod
    def _primary_value(metrics, task, metric):
        return float(metrics.get(metric, float("nan")))

    def _search_space(self, key):
        if key == "rf":
            return {"model__n_estimators": [80, 120, 180], "model__max_depth": [None, 8, 16],
                    "model__min_samples_split": [2, 5, 10]}
        if key == "logreg":
            return {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0]}
        if key in {"hist_gradient_boosting", "gradient_boosting"}:
            return {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_iter" if key.startswith("hist") else "model__n_estimators": [80, 120, 180],
                    "model__max_leaf_nodes" if key.startswith("hist") else "model__max_depth": [7, 15, 31] if key.startswith("hist") else [2, 3, 5]}
        if key == "svm":
            return {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}
        if key == "knn":
            return {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]}
        if key == "ridge":
            return {"model__alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}
        if key == "svr":
            return {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}
        return {}

    def train(self, df, target_col, model_name=None, class_weight=None, task_type=None,
              primary_metric=None, cv=None, tune=False, n_iter=8, test_size=0.2, random_state=None,
              exclude_features=None):
        started = time.perf_counter()
        experiment_id = str(uuid.uuid4())
        warnings_out = []
        try:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' was not found.")
            if int((df.columns == target_col).sum()) > 1:
                raise ValueError(f"Target column '{target_col}' is duplicated and cannot be selected unambiguously.")
            working = df.dropna(subset=[target_col]).copy()
            if len(working) < 4:
                raise ValueError("At least four non-missing target rows are required.")
            X, y = working.drop(columns=[target_col]), working[target_col]
            excluded = [str(name) for name in (exclude_features or []) if name in X.columns]
            if excluded:
                warnings_out.append("Excluded possible target leakage features: " + ", ".join(excluded))
                X = X.drop(columns=excluded)
            if X.columns.duplicated().any():
                warnings_out.append("Duplicate feature names were removed, keeping their first occurrence.")
                X = X.loc[:, ~X.columns.duplicated()].copy()
            task = task_type or self._detect_task(y)
            if task not in {"classification", "regression"}:
                raise ValueError("task_type must be classification or regression.")
            if task == "classification" and y.nunique() < 2:
                raise ValueError("Classification requires at least two target classes.")
            imbalanced = task == "classification" and self._is_imbalanced(y)
            if imbalanced and class_weight is None:
                warnings_out.append("Strong class imbalance detected; weighted F1/PR-AUC used for selection.")
            metric = self._choose_primary_metric(task, y, primary_metric)
            cv_requested = max(2, int(cv or self.cv))
            random_state = self.random_state if random_state is None else int(random_state)
            stratify = y if task == "classification" and int(y.value_counts().min()) >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify)
            folds = self._folds(task, y_train, cv=cv_requested, random_state=random_state)
            positive_label = y_train.value_counts().idxmin() if task == "classification" and y.nunique() == 2 and imbalanced else None
            if task == "classification" and y.nunique() == 2 and positive_label is None:
                positive_label = sorted(y_train.unique())[-1]
            scoring = self._scoring_name(task, metric, y_train, positive_label=positive_label)
            preprocessor = self._build_preprocessor(X)
            candidates = self._make_candidates(task, class_weight=class_weight if task == "classification" else None,
                                               random_state=random_state)
            aliases = {**self.MODEL_ALIASES, "RandomForestClassifier": "rf", "RandomForestRegressor": "rf",
                       "LogisticRegression": "logreg", "rf": "rf", "logreg": "logreg"}
            chosen = aliases.get(model_name, model_name) if model_name else None
            if chosen:
                if chosen not in candidates:
                    raise ValueError(f"Model '{model_name}' is unavailable for {task}.")
                candidates = {chosen: candidates[chosen]}
            if task == "classification" and class_weight:
                candidates = {key: estimator for key, estimator in candidates.items()
                              if "class_weight" in estimator.get_params(deep=False)}
                if not candidates:
                    raise ValueError(f"Model '{model_name}' does not support class_weight for classification.")
                if len(candidates) < 3:
                    warnings_out.append("Class-weighted experiment compared only models that support class_weight.")

            comparisons = []
            fitted_for_selection = {}
            for key, estimator in candidates.items():
                pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
                try:
                    cv_result = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                               return_train_score=True, n_jobs=1, error_score="raise")
                except Exception as exc:
                    warnings_out.append(f"Skipped {self.MODEL_LABELS[key]} after cross-validation failed: {exc}")
                    continue
                cv_scores = np.asarray(cv_result["test_score"], dtype=float)
                train_scores = np.asarray(cv_result["train_score"], dtype=float)
                if task == "regression" and metric in {"rmse", "mae"}:
                    cv_mean = float(-cv_scores.mean())
                    cv_std = float(cv_scores.std(ddof=1))
                    train_mean = float(-train_scores.mean())
                else:
                    cv_mean = float(cv_scores.mean())
                    cv_std = float(cv_scores.std(ddof=1))
                    train_mean = float(train_scores.mean())
                try:
                    fitted = clone(pipeline).fit(X_train, y_train)
                    test_metrics = self._evaluate(fitted, X_test, y_test, task, positive_label=positive_label)
                except Exception as exc:
                    warnings_out.append(f"Skipped {self.MODEL_LABELS[key]} after holdout evaluation failed: {exc}")
                    continue
                entry = {"key": key, "name": self.MODEL_LABELS[key], "cv_mean": cv_mean, "cv_std": cv_std,
                         "training_score": train_mean, "validation_score": cv_mean,
                         "test_metrics": test_metrics,
                         "test_primary_score": self._primary_value(test_metrics, task, metric)}
                comparisons.append(entry)
                fitted_for_selection[key] = fitted

            if not comparisons:
                raise ValueError("No candidate model completed cross-validation and holdout evaluation.")

            maximize = task == "classification" or metric == "r2"
            best = max(comparisons, key=lambda row: row["cv_mean"]) if maximize else min(comparisons, key=lambda row: row["cv_mean"])
            winning_key = best["key"]
            winner = fitted_for_selection[winning_key]
            tuning_results = None
            if tune:
                limit = min(self.max_tuning_trials, max(1, int(n_iter)))
                space = self._search_space(winning_key)
                if space:
                    search = RandomizedSearchCV(
                        clone(winner), param_distributions=space, n_iter=limit, scoring=scoring,
                        cv=folds, random_state=random_state, n_jobs=1, refit=True,
                        return_train_score=True, error_score="raise")
                    try:
                        search.fit(X_train, y_train)
                    except Exception as exc:
                        warnings_out.append(f"Randomized tuning failed; retained the baseline winner ({exc}).")
                        search = None
                if space and search is not None:
                    winner = search.best_estimator_
                    best_index = int(search.best_index_)
                    tuned_cv_mean = float(search.best_score_)
                    tuned_train_mean = float(search.cv_results_["mean_train_score"][best_index])
                    if task == "regression" and metric in {"rmse", "mae"}:
                        tuned_cv_mean = -tuned_cv_mean
                        tuned_train_mean = -tuned_train_mean
                    tuned_cv_std = float(search.cv_results_["std_test_score"][best_index])
                    tuning_results = {"best_params": search.best_params_, "best_score": tuned_cv_mean,
                                      "search_score": float(search.best_score_),
                                      "trials": int(len(search.cv_results_["params"]))}
                    tuned_test_metrics = self._evaluate(winner, X_test, y_test, task, positive_label=positive_label)
                    best = dict(best)
                    best.update({"cv_mean": tuned_cv_mean, "cv_std": tuned_cv_std,
                                 "training_score": tuned_train_mean, "validation_score": tuned_cv_mean})
                    best["test_metrics"] = tuned_test_metrics
                    best["tuned"] = True
                    for comparison in comparisons:
                        if comparison["key"] == winning_key:
                            comparison.update({"cv_mean": tuned_cv_mean, "cv_std": tuned_cv_std,
                                               "training_score": tuned_train_mean, "validation_score": tuned_cv_mean,
                                               "test_metrics": tuned_test_metrics, "tuned": True})
                elif not space:
                    warnings_out.append(f"No bounded tuning space configured for {self.MODEL_LABELS[winning_key]}.")

            test_metrics = self._evaluate(winner, X_test, y_test, task, positive_label=positive_label)
            training_metrics = self._evaluate(winner, X_train, y_train, task, positive_label=positive_label)
            feature_schema = [{"name": str(col), "dtype": str(df[col].dtype)} for col in X.columns]
            artifact = {"pipeline": clone(winner).fit(X, y), "target": str(target_col), "task_type": task,
                        "feature_schema": feature_schema, "model_type": self.MODEL_LABELS[winning_key],
                        "experiment_id": experiment_id, "version": 1}
            artifact_path = os.path.join(self.output_dir, f"{experiment_id}.pkl")
            with open(artifact_path, "wb") as stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            predictions = winner.predict(X_test)
            comparison_output = [{k: v for k, v in row.items() if k != "key"} for row in comparisons]
            cv_record = {"folds": int(folds.get_n_splits()), "primary_metric": metric,
                         "mean": best["cv_mean"], "std": best["cv_std"], "training_score": best["training_score"],
                         "validation_score": best["validation_score"], "tuning": tuning_results}
            if task == "classification" and y.nunique() == 2:
                cv_record["positive_label"] = positive_label
            experiment = ExperimentResult.create(
                experiment_id=experiment_id,
                dataset={"rows": int(len(working)), "columns": int(working.shape[1]),
                         "fingerprint": hashlib.sha256(pd.util.hash_pandas_object(working, index=True).values.tobytes()).hexdigest()},
                target=str(target_col), task_type=task, features=[str(c) for c in X.columns],
                preprocessing={"numeric": X.select_dtypes(include=["number"]).columns.astype(str).tolist(),
                               "categorical": X.select_dtypes(exclude=["number"]).columns.astype(str).tolist(),
                               "numeric_imputation": "median", "categorical_imputation": "most_frequent",
                               "categorical_encoding": "one_hot", "numeric_scaling": "standard"},
                model=self.MODEL_LABELS[winning_key], hyperparameters=winner.named_steps["model"].get_params(deep=False),
                cv_results=cv_record, model_comparison=comparison_output,
                test_metrics={**test_metrics, "training_metrics": training_metrics,
                              "primary_metric": metric, "class_imbalance": imbalanced},
                training_time_seconds=float(time.perf_counter() - started), warnings=warnings_out,
                artifact_path=artifact_path)
            output = experiment.to_dict()
            # Phase 1 and Streamlit compatibility fields.
            output.update({"status": "success", "task_type": task, "model_name": self.MODEL_LABELS[winning_key],
                           "model_key": winning_key, "model_path": artifact_path, "artifact_path": artifact_path,
                           "metrics": test_metrics, "sample_predictions": predictions[:10].tolist(),
                           "primary_metric": metric, "class_imbalance": imbalanced,
                           "class_weight": class_weight if "class_weight" in winner.named_steps["model"].get_params(deep=False) else None,
                           "best_model": self.MODEL_LABELS[winning_key], "models": comparison_output,
                           "excluded_features": excluded,
                           "training_metrics": training_metrics, "tuning": tuning_results})
            from core.mlflow_tracking import MLflowTracker
            output["tracking"] = MLflowTracker().log(output, artifact_path=artifact_path)
            return output
        except Exception as exc:
            return {"status": "error", "experiment_id": experiment_id, "error": str(exc)}

    @staticmethod
    def load_artifact(path):
        with open(path, "rb") as stream:
            artifact = pickle.load(stream)
        required = {"pipeline", "target", "task_type", "feature_schema", "model_type", "experiment_id"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing fields: {', '.join(sorted(missing))}")
        return artifact

    @classmethod
    def predict(cls, artifact_path, frame):
        artifact = cls.load_artifact(artifact_path)
        expected = [item["name"] for item in artifact["feature_schema"]]
        missing = set(expected) - set(frame.columns)
        if missing:
            raise ValueError(f"Prediction data is missing features: {', '.join(sorted(missing))}")
        return artifact["pipeline"].predict(frame[expected])
