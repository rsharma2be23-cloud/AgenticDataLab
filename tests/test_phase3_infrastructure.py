import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from core.a2a_bus import A2ABus
from core.execution_engine import ExecutionEngine
from core.task_state import AnalyticalMemoryStore, TaskStateStore
from core.tracing import ExecutionTracer
from core.agentic_schemas import AgentPlan, CriticResult
from core.mlflow_tracking import MLflowTracker


class TestA2ABus(unittest.TestCase):
    def test_structured_message_and_successful_deduplicated_dispatch(self):
        bus = A2ABus(persist=False)
        message = bus.publish("planner", "model", "train", {"target": "y"}, task_id="t1")["message"]
        self.assertEqual(message["correlation_id"], "t1")
        self.assertEqual(message["status"], "queued")
        first = bus.dispatch(message, lambda payload: {"trained": payload["target"]})
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(bus.task_status("t1")["status"], "succeeded")
        self.assertTrue(bus.dispatch(message, lambda _: self.fail("duplicate handler invoked"))["duplicate"])

    def test_validation_failure_and_retry_exhaustion(self):
        bus = A2ABus(persist=False, max_retries=1)
        with self.assertRaises(ValueError):
            bus.publish("", "agent", "task", {})
        with self.assertRaises(ValueError):
            bus.publish("profiler", "eda", "profile", {"sample_rows": [{"x": 1}]})
        msg = bus.publish("planner", "model", "train", {}, task_id="failure-task")["message"]
        result = bus.dispatch(msg, lambda _: (_ for _ in ()).throw(RuntimeError("failed")))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(bus.task_status(msg["task_id"])["status"], "failed")
        self.assertEqual(result["retry_count"], 1)
        timeout_msg = bus.publish("planner", "model", "slow", {})["message"]
        timed = bus.dispatch(timeout_msg, lambda _: time.sleep(0.15), timeout_seconds=0.05, max_retries=0)
        self.assertEqual(timed["status"], "failed")
        self.assertIn("timed out", timed["error"])


class TestPersistentInfrastructure(unittest.TestCase):
    def test_task_state_survives_new_store_and_recovers_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tasks.json")
            TaskStateStore(path).save({"task_id": "t", "status": "completed", "api_key": "must-not-persist",
                                       "tool_outputs": {"s": {"sample_rows": [{"x": 1}], "count": 2}}})
            recovered = TaskStateStore(path).get("t")
            self.assertEqual(recovered["status"], "completed")
            self.assertNotIn("api_key", recovered)
            self.assertNotIn("sample_rows", recovered["tool_outputs"]["s"])
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("{bad")
            self.assertIsNone(TaskStateStore(path).get("t"))

    def test_analytical_memory_retrieval_and_secret_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = AnalyticalMemoryStore(os.path.join(directory, "memory.json"))
            memory.add({"dataset_fingerprint": "abc123", "finding": "missing values in age"})
            self.assertEqual(len(memory.retrieve("abc123 missing values")), 1)
            with self.assertRaises(ValueError):
                memory.add({"finding": "api_key=not-safe"})

    def test_trace_correlates_events_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            tracer = ExecutionTracer(os.path.join(directory, "trace.jsonl"))
            tracer.record({"task_id": "t", "correlation_id": "t", "event": "tool", "api_key": "secret-value",
                           "detail": "token=secret-value", "prompt": "private"})
            row = tracer.list("t")[0]
            self.assertEqual(row["correlation_id"], "t")
            self.assertNotIn("api_key", row)
            self.assertNotIn("prompt", row)
            self.assertIn("[REDACTED]", row["detail"])

    def test_mlflow_is_optional_and_reports_unavailability(self):
        self.assertEqual(MLflowTracker(enabled=False).log({})["status"], "disabled")
        result = MLflowTracker(tracking_uri="file:./mlruns", enabled=True).log({})
        self.assertIn(result["status"], {"logged", "unavailable"})

    def test_mlflow_logs_metadata_metrics_and_run_ids_with_mock(self):
        class Run:
            info = types.SimpleNamespace(experiment_id="exp-1", run_id="run-1")
            def __enter__(self): return self
            def __exit__(self, *args): return False
        fake = types.ModuleType("mlflow")
        fake.__path__ = []
        fake.set_tracking_uri = lambda uri: None
        fake.get_tracking_uri = lambda: "file:./mlruns"
        fake.set_experiment = lambda name: None
        fake.start_run = lambda: Run()
        fake.set_tags = lambda tags: None
        fake.log_param = lambda key, value: None
        fake.log_metric = lambda key, value: None
        fake.log_artifact = lambda path, artifact_path=None: None
        with patch.dict(sys.modules, {"mlflow": fake, "mlflow.sklearn": types.ModuleType("mlflow.sklearn")}):
            result = MLflowTracker(tracking_uri="file:./mlruns", enabled=True).log({
                "experiment_id": "experiment-1", "model": "Ridge", "task_type": "regression",
                "hyperparameters": {"alpha": 1.0}, "cv_results": {"mean": 0.9, "std": 0.1},
                "test_metrics": {"r2": 0.8}, "dataset": {"fingerprint": "safe-fingerprint", "rows": 20, "columns": 3}})
        self.assertEqual(result["status"], "logged")
        self.assertEqual(result["run_id"], "run-1")


class TestApprovalGate(unittest.TestCase):
    def test_pending_blocks_execution_then_approval_allows_it(self):
        class Planner:
            def plan(self, *args, **kwargs):
                return AgentPlan.from_dict({"goal": "train", "steps": [
                    {"id": "s0", "agent": "quality", "tool": "inspect", "reason": "inspect schema"},
                    {"id": "s1", "agent": "model", "tool": "train_model", "reason": "fit model"}]})
        class Tools:
            def __init__(self):
                self.calls = []
            def list_tools(self):
                return {"inspect": "inspect", "train_model": "train"}
            def execute(self, *args):
                self.calls.append(args[0])
                return {"status": "success", "metrics": {"score": 1}}
        class Critic:
            def review(self, *args, **kwargs):
                return CriticResult("ACCEPT", "completed")
        with tempfile.TemporaryDirectory() as directory:
            store, tools = TaskStateStore(os.path.join(directory, "tasks.json")), Tools()
            engine = ExecutionEngine(planner=Planner(), tools=tools, critic=Critic(), state_store=store,
                                     approval_required={"train_model"}, tracer=ExecutionTracer(os.path.join(directory, "trace")),
                                     memory_store=AnalyticalMemoryStore(os.path.join(directory, "memory.json")))
            import pandas as pd
            frame = pd.DataFrame({"x": [1]})
            pending = engine.run(frame, "train", task_id="approval-1")
            self.assertEqual(pending["status"], "pending_approval")
            self.assertEqual(tools.calls, ["inspect"])
            engine.resolve_approval("approval-1", "s1", True)
            done = engine.run(frame, "train", task_id="approval-1")
            self.assertEqual(done["status"], "completed")
            self.assertEqual(tools.calls, ["inspect", "train_model"])

    def test_rejection_is_terminal(self):
        class Planner:
            def plan(self, *args, **kwargs):
                return AgentPlan.from_dict({"goal": "train", "steps": [
                    {"id": "s1", "agent": "model", "tool": "train_model", "reason": "fit model"}]})
        class Tools:
            def list_tools(self): return {"train_model": "train"}
            def execute(self, *args): raise AssertionError("must not execute")
        with tempfile.TemporaryDirectory() as directory:
            engine = ExecutionEngine(planner=Planner(), tools=Tools(), state_store=TaskStateStore(os.path.join(directory, "tasks.json")),
                                     approval_required={"train_model"}, tracer=ExecutionTracer(os.path.join(directory, "trace")))
            import pandas as pd
            engine.run(pd.DataFrame({"x": [1]}), "train", task_id="approval-2")
            engine.resolve_approval("approval-2", "s1", False)
            result = engine.run(pd.DataFrame({"x": [1]}), "train", task_id="approval-2")
            self.assertEqual(result["status"], "rejected")


class TestBenchmarkMetrics(unittest.TestCase):
    def test_metric_calculation_includes_failures_and_expected_rejections(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks"))
        from run_benchmark import calculate_metrics
        rows = [{"status": "success", "tool_selection_correct": True, "schema_valid": True,
                 "evidence_present": True, "duration_seconds": 1},
                {"status": "expected_rejection", "expected_rejection": True,
                 "tool_selection_correct": True, "schema_valid": True, "duration_seconds": 0.2},
                {"status": "failure", "tool_selection_correct": False, "schema_valid": False,
                 "duration_seconds": 0.8}]
        result = calculate_metrics(rows)
        self.assertAlmostEqual(result["task_completion_rate"], 2 / 3)
        self.assertAlmostEqual(result["failure_rate"], 1 / 3)
        self.assertAlmostEqual(result["evidence_coverage"], 1)


def mcp_client_available():
    try:
        return importlib.util.find_spec("mcp.client") is not None
    except (ImportError, ModuleNotFoundError):
        return False


@unittest.skipUnless(mcp_client_available(), "MCP SDK not installed in this Python environment")
class TestMCPProtocol(unittest.TestCase):
    def test_discovery_valid_call_and_invalid_dataset(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        async def exercise():
            environment = dict(os.environ)
            environment["AGENTIC_DATA_DIR"] = os.path.join(PROJECT_ROOT, "benchmarks")
            params = StdioServerParameters(command=sys.executable, args=[os.path.join(PROJECT_SRC, "mcp_server.py")], env=environment)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    discovered = await session.list_tools()
                    names = {tool.name for tool in discovered.tools}
                    self.assertIn("profile_dataset", names)
                    good = await session.call_tool("profile_dataset", {"dataset": "synthetic_classification.csv"})
                    self.assertFalse(good.isError)
                    bad = await session.call_tool("profile_dataset", {"dataset": "..\\README.md"})
                    self.assertTrue(bad.isError)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
