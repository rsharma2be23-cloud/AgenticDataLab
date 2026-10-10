import io
import importlib.util
import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "streamlit_app"))

from workflow_service import read_uploaded_dataset
from core.agentic_schemas import AgentPlan, CriticResult
from core.execution_engine import ExecutionEngine
from core.task_state import TaskStateStore
from core.tracing import ExecutionTracer
from tools.file_tools import FileTools


class Upload(io.BytesIO):
    def __init__(self, name, payload):
        super().__init__(payload)
        self.name = name

    def getvalue(self):
        return super().getvalue()


class Phase4IntegrationTests(unittest.TestCase):
    def test_upload_validates_extension_content_size_and_rows(self):
        frame = read_uploaded_dataset(Upload("safe.csv", b"x,y\n1,a\n2,b\n"))
        self.assertEqual(frame.shape, (2, 2))
        with self.assertRaisesRegex(ValueError, r"\.csv"):
            read_uploaded_dataset(Upload("data.xlsx", b"x,y\n1,a\n"))
        with self.assertRaisesRegex(ValueError, "empty"):
            read_uploaded_dataset(Upload("empty.csv", b""))
        with self.assertRaisesRegex(ValueError, "unique"):
            read_uploaded_dataset(Upload("duplicate.csv", b"x,x\n1,2\n"))
        with self.assertRaisesRegex(ValueError, "row limit"):
            read_uploaded_dataset(Upload("many.csv", b"x\n1\n2\n"), max_rows=1)

    def test_file_tools_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            files = FileTools(os.path.join(temp, "store"))
            result = files.write("..\\outside.txt", "blocked")
            self.assertEqual(result["status"], "error")
            self.assertIn("inside", result["error"])

    def test_planner_tool_state_and_trace_use_actual_outputs(self):
        class Planner:
            def plan(self, *args, **kwargs):
                return AgentPlan.from_dict({"goal": "inspect", "steps": [
                    {"id": "s1", "agent": "profiler", "tool": "profile_dataset", "reason": "Profile rows"}]})

        class Critic:
            def review(self, *args, **kwargs):
                return CriticResult("ACCEPT", "Profile output is available.")

        with tempfile.TemporaryDirectory() as temp:
            state = TaskStateStore(os.path.join(temp, "tasks.json"))
            trace = ExecutionTracer(os.path.join(temp, "trace.jsonl"))
            engine = ExecutionEngine(planner=Planner(), critic=Critic(), state_store=state, tracer=trace,
                                     approval_required=[], memory_store=None)
            result = engine.run(pd.DataFrame({"x": [1, None], "kind": ["a", "b"]}), "inspect", task_id="e2e")
            self.assertEqual(result["status"], "completed")
            self.assertIn("profile_dataset", result["results"])
            self.assertEqual(state.get("e2e")["status"], "completed")
            self.assertTrue(any(event.get("event") == "tool_execution" for event in trace.list("e2e")))

    def test_data_quality_returns_computed_findings(self):
        from core.tool_registry import ToolRegistry
        frame = pd.DataFrame({"target": ["a", "a", None], "constant": [1, 1, 1]})
        result = ToolRegistry().execute("analyze_data_quality", {"target_col": "target"}, frame, {"results": {}})
        self.assertEqual(result["status"], "success")
        self.assertIn("target", result["checks"]["missing_by_column"])
        self.assertGreaterEqual(result["quality_score"], 0)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_classification_and_regression_comparison_report_metrics(self):
        from tools.model_tools import ModelTools
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            model = ModelTools(output_dir=temp, cv=2, random_state=11)
            x = np.arange(40, dtype=float)
            classification = pd.DataFrame({"x": x, "target": (x % 2).astype(int)})
            classify = model.train(classification, "target", model_name="logreg", cv=2)
            self.assertEqual(classify["status"], "success")
            self.assertEqual(classify["task_type"], "classification")
            self.assertTrue(classify.get("model_comparison"))
            self.assertIn("training_metrics", classify)
            self.assertIn("cv_results", classify)

            regression = pd.DataFrame({"x": x, "target": 2 * x + 1})
            regress = model.train(regression, "target", model_name="ridge", cv=2)
            self.assertEqual(regress["status"], "success")
            self.assertEqual(regress["task_type"], "regression")
            self.assertTrue(regress.get("model_comparison"))
            self.assertIn("training_metrics", regress)
            self.assertIn("cv_results", regress)

    def test_approval_blocks_tool_and_dataset_change_blocks_resume(self):
        class Planner:
            def plan(self, *args, **kwargs):
                return AgentPlan.from_dict({"goal": "fit", "steps": [
                    {"id": "s1", "agent": "model", "tool": "train_model", "reason": "Fit model"}]})

        class Tools:
            def __init__(self):
                self.calls = []

            def list_tools(self):
                return {"train_model": "fit"}

            def execute(self, *args):
                self.calls.append(args[0])
                return {"status": "success", "metrics": {"valid": True}}

        with tempfile.TemporaryDirectory() as temp:
            tools = Tools()
            state = TaskStateStore(os.path.join(temp, "tasks.json"))
            engine = ExecutionEngine(planner=Planner(), tools=tools, critic=CriticResultReview(), state_store=state,
                                     tracer=ExecutionTracer(os.path.join(temp, "trace")), approval_required={"train_model"})
            first = engine.run(pd.DataFrame({"x": [1, 2]}), "fit", task_id="approval")
            self.assertEqual(first["status"], "pending_approval")
            self.assertEqual(tools.calls, [])
            engine.resolve_approval("approval", "s1", True)
            changed = engine.run(pd.DataFrame({"x": [7, 8]}), "fit", task_id="approval")
            self.assertEqual(changed["status"], "failed")
            self.assertEqual(state.get("approval")["status"], "failed")
            self.assertEqual(tools.calls, [])


class CriticResultReview:
    def review(self, *args, **kwargs):
        return CriticResult("ACCEPT", "Synthetic tool output available.")


if __name__ == "__main__":
    unittest.main()
