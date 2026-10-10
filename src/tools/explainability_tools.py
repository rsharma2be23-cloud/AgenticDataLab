"""Model-grounded feature importance and single-row contribution calculations."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from core.ml_schemas import FeatureImportance, LocalExplanation
from tools.model_tools import ModelTools


class ExplainabilityTools:
    METHODS = {"shap", "model_native", "coefficients", "permutation"}

    def _load(self, artifact_path, frame):
        artifact = ModelTools.load_artifact(artifact_path)
        schema = [feature["name"] for feature in artifact["feature_schema"]]
        missing = set(schema) - set(frame.columns)
        if missing:
            raise ValueError(f"Explanation data is missing features: {', '.join(sorted(missing))}")
        return artifact, frame[schema]

    @staticmethod
    def _feature_names(preprocessor, columns):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            return [str(name) for name in columns]

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        magnitudes = np.abs(values)
        total = float(magnitudes.sum())
        return magnitudes / total if total else magnitudes

    def _shap_values(self, artifact, frame):
        import shap
        pipeline = artifact["pipeline"]
        transformed = pipeline.named_steps["preprocess"].transform(frame)
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1] if len(values) == 2 else np.mean(np.abs(values), axis=0)
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1] if values.shape[-1] == 2 else np.mean(np.abs(values), axis=-1)
        return values

    def global_importance(self, artifact_path, frame, method="auto", target=None, top_n=30):
        artifact, X = self._load(artifact_path, frame)
        pipeline = artifact["pipeline"]
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        warnings = []
        names = self._feature_names(preprocessor, X.columns)
        selected = method
        if method == "auto":
            selected = "shap" if artifact["model_type"] in {"RandomForest", "GradientBoosting", "HistGradientBoosting"} else "model_native"
        if selected == "shap":
            try:
                shap_values = self._shap_values(artifact, X)
                raw = np.mean(np.abs(shap_values), axis=0)
                if raw.ndim > 1:
                    raw = np.mean(raw, axis=tuple(range(1, raw.ndim)))
                values = self._normalize(raw)
                features = [{"name": names[i] if i < len(names) else f"feature_{i}",
                             "importance": float(values[i])} for i in range(len(values))]
                return FeatureImportance("shap", sorted(features, key=lambda row: row["importance"], reverse=True)[:top_n], warnings).to_dict()
            except Exception as exc:
                warnings.append(f"SHAP unavailable for this model/data; used a deterministic fallback ({exc}).")

        raw = None
        if selected in {"auto", "model_native", "coefficients"}:
            if hasattr(model, "feature_importances_"):
                raw = np.asarray(model.feature_importances_, dtype=float)
                selected = "model_native"
            elif hasattr(model, "coef_"):
                coefficients = np.asarray(model.coef_, dtype=float)
                raw = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
                selected = "coefficients"
        if raw is None:
            try:
                target_name = target or artifact["target"]
                if target_name not in frame:
                    raise ValueError("Target values are required for permutation importance.")
                y = frame[target_name]
                scoring = "f1_weighted" if artifact["task_type"] == "classification" else "neg_root_mean_squared_error"
                result = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=3, random_state=42, n_jobs=1)
                raw = result.importances_mean
                names = [str(c) for c in X.columns]
                selected = "permutation"
            except Exception as exc:
                return {"status": "error", "error": str(exc), "warnings": warnings}
        values = self._normalize(raw)
        features = [{"name": names[i] if i < len(names) else f"feature_{i}",
                     "importance": float(values[i])} for i in range(len(values))]
        return FeatureImportance(selected, sorted(features, key=lambda row: row["importance"], reverse=True)[:top_n], warnings).to_dict()

    @staticmethod
    def _score_prediction(pipeline, row, task):
        if task == "classification" and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(row)[0]
            index = 1 if len(probabilities) == 2 else int(np.argmax(probabilities))
            return float(probabilities[index]), pipeline.predict(row)[0]
        prediction = pipeline.predict(row)[0]
        return float(prediction), prediction

    def local_explanation(self, artifact_path, instance, reference_data, method="auto", top_n=10):
        artifact, X = self._load(artifact_path, reference_data)
        row = instance.to_dict() if isinstance(instance, pd.Series) else dict(instance)
        row_frame = pd.DataFrame([{col: row.get(col, np.nan) for col in X.columns}], columns=X.columns)
        pipeline = artifact["pipeline"]
        prediction_score, prediction_label = self._score_prediction(pipeline, row_frame, artifact["task_type"])
        baseline_values = {}
        for col in X.columns:
            series = X[col]
            baseline_values[col] = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else np.nan
        baseline = pd.DataFrame([baseline_values], columns=X.columns)
        baseline_score, baseline_label = self._score_prediction(pipeline, baseline, artifact["task_type"])
        warnings = []
        selected = method
        if selected == "auto":
            selected = "shap" if artifact["model_type"] in {"RandomForest", "GradientBoosting", "HistGradientBoosting"} else "perturbation"
        contributions = []
        if selected == "shap":
            try:
                if artifact["task_type"] == "classification" and pd.Series(artifact["pipeline"].classes_).nunique() > 2:
                    raise ValueError("Local SHAP explanation is currently limited to binary classification.")
                names = self._feature_names(pipeline.named_steps["preprocess"], X.columns)
                shap_values = self._shap_values(artifact, row_frame)
                base_values = np.asarray(shap_values).reshape(-1)
                transformed_row = pipeline.named_steps["preprocess"].transform(row_frame)
                transformed_row = transformed_row.toarray() if hasattr(transformed_row, "toarray") else np.asarray(transformed_row)
                for index, value in enumerate(base_values):
                    contributions.append({"feature": names[index] if index < len(names) else f"feature_{index}",
                                          "actual_value": transformed_row[0, index].item() if index < transformed_row.shape[1] and hasattr(transformed_row[0, index], "item") else float(transformed_row[0, index]) if index < transformed_row.shape[1] else None,
                                          "contribution": float(value), "effect": "increases" if value > 0 else "decreases" if value < 0 else "no change"})
            except Exception as exc:
                warnings.append(f"SHAP local explanation failed; used feature perturbations ({exc}).")
                selected = "perturbation"
        if selected != "shap" or not contributions:
            selected = "perturbation"
            for col in X.columns:
                changed = row_frame.copy()
                changed.loc[:, col] = baseline_values[col]
                changed_score, _ = self._score_prediction(pipeline, changed, artifact["task_type"])
                impact = prediction_score - changed_score
                actual = row_frame.iloc[0][col]
                actual = actual.item() if hasattr(actual, "item") else actual
                base_value = baseline_values[col]
                base_value = base_value.item() if hasattr(base_value, "item") else base_value
                contributions.append({"feature": str(col), "actual_value": actual, "baseline_value": base_value,
                                      "contribution": float(impact),
                                      "effect": "increases" if impact > 1e-12 else "decreases" if impact < -1e-12 else "no change"})
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return LocalExplanation(selected, prediction_label.item() if hasattr(prediction_label, "item") else prediction_label,
                                baseline_label.item() if hasattr(baseline_label, "item") else baseline_label,
                                contributions[:top_n], warnings).to_dict()
