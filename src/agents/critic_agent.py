"""Evidence-based critique of tool failures, data quality, and ML experiments."""
from core.agentic_schemas import CriticResult


class CriticAgent:
    def review(self, task_result, metrics=None, errors=None, tool_output=None):
        errors = errors or []
        if errors:
            issue_text = [str(item.get("error", item)) if isinstance(item, dict) else str(item) for item in errors]
            return CriticResult("RETRY", "One or more planned tools failed.", ["Retry failed tool steps"],
                                confidence=1.0, issues=issue_text)
        if isinstance(task_result, dict) and task_result.get("status") == "error":
            return CriticResult("RETRY", task_result.get("error", "Tool execution failed"),
                                ["Check tool inputs and retry"], confidence=1.0,
                                issues=[task_result.get("error", "Tool execution failed")])

        results = task_result if isinstance(task_result, dict) else {}
        model = results.get("train_model") or (tool_output if isinstance(tool_output, dict) else {})
        quality = results.get("analyze_data_quality") or results.get("data_quality") or {}
        model_metrics = metrics or model.get("metrics", {}) or {}
        cv = model.get("cv_results", {}) or {}
        metric_name = model.get("primary_metric") or cv.get("primary_metric")
        quality_imbalance = quality.get("checks", {}).get("class_imbalance") if isinstance(quality, dict) else None
        quality_imbalanced = (quality_imbalance.get("is_imbalanced", quality_imbalance.get("majority_fraction", 0) >= 0.8)
                              if isinstance(quality_imbalance, dict) else False)
        imbalance = bool(model.get("class_imbalance") or quality_imbalanced)
        issues, recommendations = [], []

        if model.get("status") == "error":
            issues.append(model.get("error", "Model training failed."))
            recommendations.append("Retry with a simpler model or inspect the target/features.")
        if metric_name == "accuracy" and imbalance:
            issues.append("Accuracy is the primary metric despite target class imbalance.")
            recommendations.extend(["Use PR-AUC or F1 as the primary metric", "Use class_weight=balanced"])

        f1 = model_metrics.get("f1", model_metrics.get("f1_score"))
        recall = model_metrics.get("recall")
        precision = model_metrics.get("precision")
        if f1 is not None and float(f1) < 0.5:
            issues.append(f"Poor F1 score ({float(f1):.3f}).")
            recommendations.append("Try another candidate model or tune hyperparameters.")
        if recall is not None and float(recall) < 0.5:
            issues.append(f"Poor recall ({float(recall):.3f}).")
            recommendations.extend(["Use recall or PR-AUC for model selection", "Try class_weight=balanced"])
        if precision is not None and float(precision) < 0.5:
            issues.append(f"Poor precision ({float(precision):.3f}).")
            recommendations.append("Inspect the precision-recall tradeoff and try another decision strategy.")
        if model.get("task_type") == "regression" and model_metrics.get("r2") is not None and float(model_metrics["r2"]) < 0.3:
            issues.append(f"Low regression R2 ({float(model_metrics['r2']):.3f}).")
            recommendations.append("Try another regression model or tune the leading candidate.")
        if metric_name == "roc_auc" and model_metrics.get("roc_auc") is not None and float(model_metrics["roc_auc"]) < 0.5:
            issues.append("ROC-AUC is below the random-ranking baseline.")
            recommendations.append("Inspect target encoding and compare another classifier.")
        if metric_name == "pr_auc" and model_metrics.get("pr_auc") is not None and float(model_metrics["pr_auc"]) < 0.1:
            issues.append("PR-AUC is very low for the selected imbalanced classification task.")
            recommendations.append("Try class weighting and inspect the minority-class features.")

        train_metrics = model_metrics.get("training_metrics", {})
        train_score = train_metrics.get(metric_name) if metric_name else None
        if train_score is None and metric_name == "f1":
            train_score = train_metrics.get("f1", train_metrics.get("f1_score"))
        cv_mean = cv.get("mean")
        cv_std = cv.get("std")
        if train_score is not None and cv_mean is not None and metric_name in {"f1", "recall", "precision", "accuracy", "roc_auc", "pr_auc", "r2"} and float(train_score) - float(cv_mean) > 0.15:
            issues.append(f"Potential overfitting: training {metric_name} materially exceeds cross-validation {metric_name}.")
            recommendations.append("Reduce model complexity or tune with stronger regularization.")
        if train_score is not None and cv_mean is not None and metric_name in {"rmse", "mae"} and float(train_score) < float(cv_mean) * 0.8:
            issues.append(f"Potential overfitting: training {metric_name} is materially lower than cross-validation {metric_name}.")
            recommendations.append("Reduce model complexity or tune with stronger regularization.")
        unstable = False
        if cv_std is not None:
            if model.get("task_type") == "regression" and cv_mean not in (None, 0):
                unstable = float(cv_std) / abs(float(cv_mean)) > 0.2
            else:
                unstable = float(cv_std) > 0.1
        if unstable:
            issues.append(f"Unstable cross-validation results (standard deviation {float(cv_std):.3f}).")
            recommendations.append("Use more data or inspect fold-level variation before relying on this model.")

        quality_warnings = quality.get("warnings", []) if isinstance(quality, dict) else []
        quality_issues = quality.get("issues", []) if isinstance(quality, dict) else []
        class_weight_applied = model.get("class_weight") == "balanced"
        if imbalance and model.get("status") == "success" and not class_weight_applied:
            issues.append("Target class imbalance was detected.")
            if not any("class_weight=balanced" in recommendation for recommendation in recommendations):
                recommendations.append("Try class_weight=balanced.")
        possible_leakage = quality.get("checks", {}).get("possible_target_leakage", []) if isinstance(quality, dict) else []
        model_features = set(model.get("features", []))
        active_leakage = [str(name) for name in possible_leakage if name in model_features]
        if active_leakage and model.get("status") == "success":
            issues.append("Possible target leakage remains in model features: " + ", ".join(active_leakage))
            recommendations.append("Inspect and remove possible leakage features before accepting model results.")

        # Confidence is evidence coverage, not a subjective quality score.
        evidence = [bool(model.get("status") == "success"), bool(model_metrics), bool(cv),
                    bool(quality.get("checks")) if isinstance(quality, dict) else False]
        confidence = round(sum(evidence) / len(evidence), 2)
        if issues:
            unique_recommendations = list(dict.fromkeys(recommendations))
            return CriticResult("RETRY", "ML or data-quality checks found issues requiring another experiment.",
                                unique_recommendations, confidence=confidence, issues=list(dict.fromkeys(issues)))
        if not model and not any(isinstance(value, dict) and value.get("status") == "success" for value in results.values()):
            return CriticResult("REJECT", "No successful tool output was available for review.", [], confidence=0.0,
                                issues=["No successful tool output was available for review."])
        return CriticResult("ACCEPT", "Available model and data-quality checks passed the Phase 2 review.", [],
                            confidence=confidence, issues=[])
