import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from agents.critic_agent import CriticAgent
from agents.data_quality_agent import DataQualityAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.planner_agent import PlannerAgent
from core.execution_engine import ExecutionEngine
from core.task_state import TaskStateStore
from tools.model_tools import ModelTools


class TestDataQualityAgent(unittest.TestCase):
    def test_detects_missing_duplicates_constants_ids_and_imbalance(self):
        frame = pd.DataFrame({
            "record_id": list(range(20)) + [0],
            "constant": [1] * 21,
            "near_constant": [0] * 20 + [1],
            "category": [f"category-{i}" for i in range(21)],
            "signal": list(range(20)) + [0],
            "target": [0] * 18 + [1] * 3,
            "leak": ["no"] * 18 + ["yes"] * 3,
        })
        frame.loc[1, "signal"] = np.nan
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        result = DataQualityAgent().analyze(frame, target_col="target")
        self.assertLess(result["quality_score"], 100)
        self.assertGreater(result["checks"]["duplicate_rows"], 0)
        self.assertIn("signal", result["checks"]["missing_by_column"])
        self.assertIn("constant", result["checks"]["constant_columns"])
        self.assertIn("record_id", result["checks"]["suspicious_id_columns"])
        self.assertIsNotNone(result["checks"]["class_imbalance"])
        self.assertIn("leak", result["checks"]["possible_target_leakage"])

    def test_duplicate_column_names_are_reported_without_crashing(self):
        frame = pd.DataFrame([[1, 2, 0], [2, 3, 1], [3, 4, 0]], columns=["x", "x", "target"])
        result = DataQualityAgent().analyze(frame, target_col="target")
        self.assertEqual(result["checks"]["duplicate_columns"], 1)

    def test_detects_high_cardinality_near_constant_outliers_non_finite_and_leakage(self):
        rows = 101
        target = ["low"] * 90 + ["high"] * 11
        frame = pd.DataFrame({
            "record_id": range(rows),
            "near_constant": [0] * 100 + [1],
            "category": [f"value-{i}" for i in range(rows)],
            "numeric": [0] * 100 + [1000],
            "invalid": [1] * 100 + [np.inf],
            "leak_copy": target,
            "target": target,
        })
        result = DataQualityAgent().analyze(frame, "target")
        checks = result["checks"]
        self.assertIn("near_constant", checks["near_constant_columns"])
        self.assertIn("category", checks["high_cardinality_categorical"])
        self.assertIn("record_id", checks["suspicious_id_columns"])
        self.assertIn("invalid", checks["invalid_numeric_values"])
        self.assertIn("numeric", checks["outlier_counts"])
        self.assertIn("leak_copy", checks["possible_target_leakage"])


class TestModelTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(11)
        self.classification = pd.DataFrame({
            "numeric": rng.normal(size=100),
            "category": np.where(np.arange(100) % 2, "b", "a"),
            "target": np.where(np.arange(100) % 2, "yes", "no"),
        })
        self.classification.loc[0, "numeric"] = np.nan
        self.regression = pd.DataFrame({
            "x": np.linspace(0, 10, 80),
            "category": np.where(np.arange(80) % 2, "b", "a"),
        })
        self.regression["target"] = self.regression["x"] * 2.5 + np.sin(self.regression["x"])
        self.regression.loc[1, "x"] = np.nan

    def tearDown(self):
        self.temp.cleanup()

    def test_classification_cv_models_metrics_artifact_and_explanations(self):
        tools = ModelTools(output_dir=self.temp.name, cv=2, random_state=3)
        result = tools.train(self.classification, "target", task_type="classification", cv=2)
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertGreaterEqual(len(result["model_comparison"]), 4)
        self.assertEqual(result["cv_results"]["folds"], 2)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            self.assertIn(metric, result["metrics"])
        self.assertEqual(len(tools.predict(result["artifact_path"], self.classification.drop(columns="target"))), len(self.classification))
        agent = ExplainabilityAgent()
        global_result = agent.explain_global(result["artifact_path"], self.classification, method="model_native")
        self.assertTrue(global_result["features"])
        self.assertAlmostEqual(sum(row["importance"] for row in global_result["features"]), 1.0, places=5)
        local_result = agent.explain_prediction(result["artifact_path"], self.classification.iloc[0], self.classification,
                                                method="perturbation")
        self.assertTrue(local_result["contributions"])
        auto_result = agent.explain_global(result["artifact_path"], self.classification, method="auto")
        self.assertTrue(auto_result["features"])

    def test_regression_cv_metrics(self):
        tools = ModelTools(output_dir=self.temp.name, cv=2, random_state=3)
        result = tools.train(self.regression, "target", task_type="regression", cv=2)
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertGreaterEqual(len(result["model_comparison"]), 5)
        for metric in ("mae", "rmse", "r2"):
            self.assertIn(metric, result["metrics"])

    def test_bounded_randomized_tuning(self):
        tools = ModelTools(output_dir=self.temp.name, cv=2, random_state=3, max_tuning_trials=2)
        result = tools.train(self.classification, "target", model_name="logreg", task_type="classification",
                             primary_metric="f1", cv=2, tune=True, n_iter=2)
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(result["tuning"]["trials"], 2)

    def test_imbalance_selects_pr_auc(self):
        tools = ModelTools(output_dir=self.temp.name, cv=2)
        target = pd.Series(["major"] * 90 + ["minor"] * 10)
        self.assertTrue(tools._is_imbalanced(target))
        self.assertEqual(tools._choose_primary_metric("classification", target), "pr_auc")


class TestPhase2CriticAndPlanning(unittest.TestCase):
    def test_critic_detects_poor_recall_overfit_instability_and_imbalance(self):
        results = {"train_model": {
            "status": "success", "model_key": "rf", "target_col": "target", "class_weight": None,
            "primary_metric": "f1", "class_imbalance": True,
            "metrics": {"f1": 0.39, "recall": 0.25, "precision": 0.55},
            "training_metrics": {"f1": 0.98}, "cv_results": {"mean": 0.68, "std": 0.15},
        }, "analyze_data_quality": {"checks": {"class_imbalance": {"majority_fraction": 0.9}}, "warnings": []}}
        result = CriticAgent().review(results).to_dict()
        self.assertEqual(result["status"], "RETRY")
        self.assertGreater(result["confidence"], 0.5)
        self.assertTrue(any("overfitting" in issue.lower() for issue in result["issues"]))
        self.assertTrue(any("unstable" in issue.lower() for issue in result["issues"]))
        self.assertTrue(any("recall" in issue.lower() for issue in result["issues"]))

    def test_critic_retry_changes_next_experiment(self):
        planner = PlannerAgent(tool_descriptions={"train_model": "train", "evaluate_model": "evaluate",
                                                   "explain_global": "explain"})
        plan = planner.plan("identify important factors", columns=["target"],
                            previous_results={"train_model": {"target_col": "target", "model_key": "rf"}},
                            critic_result={"status": "RETRY", "recommendations": [
                                "Use recall or PR-AUC for model selection", "Try class_weight=balanced", "Tune hyperparameters"]})
        train_step = next(step for step in plan.steps if step.tool == "train_model")
        self.assertEqual(train_step.inputs["class_weight"], "balanced")
        self.assertEqual(train_step.inputs["primary_metric"], "recall")
        self.assertTrue(train_step.inputs["tune"])
        self.assertEqual(train_step.inputs["model_name"], "logreg")

    def test_planner_excludes_reported_leakage_feature_on_retry(self):
        planner = PlannerAgent(tool_descriptions={"train_model": "train", "evaluate_model": "evaluate"})
        plan = planner.plan("analyze target", columns=["leak", "target"],
                            previous_results={"analyze_data_quality": {"checks": {"possible_target_leakage": ["leak"]}}},
                            critic_result={"status": "RETRY", "recommendations": ["Remove possible leakage features"]})
        train_step = next(step for step in plan.steps if step.tool == "train_model")
        self.assertEqual(train_step.inputs["exclude_features"], ["leak"])

    def test_execution_engine_replans_from_poor_recall(self):
        class ExperimentTools:
            def __init__(self):
                self.calls = []

            def list_tools(self):
                return {"train_model": "train", "evaluate_model": "evaluate"}

            def execute(self, name, inputs, df, context):
                self.calls.append((name, inputs.copy()))
                if name == "train_model":
                    is_retry = len([call for call in self.calls if call[0] == "train_model"]) > 1
                    return {"status": "success", "task_type": "classification", "target_col": "target",
                            "model_key": "logreg" if is_retry else "rf",
                            "class_weight": inputs.get("class_weight"),
                            "primary_metric": inputs.get("primary_metric", "f1"),
                            "metrics": {"f1": 0.8, "recall": 0.7, "precision": 0.9} if is_retry else {"f1": 0.39, "recall": 0.25, "precision": 0.6},
                            "training_metrics": {"f1": 0.82 if is_retry else 0.6},
                            "cv_results": {"mean": 0.79 if is_retry else 0.39, "std": 0.02}}
                return {"status": "success", "metrics": context["latest_model"].get("metrics", {})}

        tools = ExperimentTools()
        with tempfile.TemporaryDirectory() as directory:
            result = ExecutionEngine(planner=PlannerAgent(tool_descriptions=tools.list_tools()), tools=tools,
                                     critic=CriticAgent(), state_store=TaskStateStore(os.path.join(directory, "tasks.json")),
                                     max_retries=2).run(pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]}),
                                                       "train a model for this target", task_id="ml-replan")
        train_calls = [inputs for name, inputs in tools.calls if name == "train_model"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(train_calls), 2)
        self.assertEqual(train_calls[1]["class_weight"], "balanced")
        self.assertEqual(train_calls[1]["primary_metric"], "recall")


if __name__ == "__main__":
    unittest.main()
