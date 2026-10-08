import os
import sys
import tempfile
import unittest

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import pandas as pd

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from core.a2a_bus import A2ABus
from core.agentic_schemas import AgentPlan, CriticResult
from core.execution_engine import ExecutionEngine
from core.task_state import TaskStateStore
from core.tool_registry import ToolInput, ToolRegistry


class RecordingTools:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return {"inspect": "test tool"}

    def execute(self, name, inputs, df, context):
        self.calls.append((name, inputs))
        return {"status": "success", "rows": len(df)}


class RetryCritic:
    def __init__(self, count=1):
        self.count = count
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.count:
            return CriticResult("RETRY", "Need a second pass", ["Add schema inspection"])
        return CriticResult("ACCEPT", "Passed", [])


class FakeLLM:
    def generate_content(self, prompt):
        class Response:
            text = '{"goal":"inspect","steps":[{"id":"s1","agent":"quality","tool":"inspect","reason":"requested schema"}]}'
        return Response()


class TestAgenticArchitecture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_store = TaskStateStore(os.path.join(self.temp.name, "tasks.json"))
        self.df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_schema_and_allowlist_validation(self):
        plan = AgentPlan.from_dict({"goal": "inspect", "steps": [{"id": "s1", "agent": "quality", "tool": "inspect", "reason": "need schema"}]}, {"inspect"})
        self.assertEqual(plan.steps[0].tool, "inspect")
        with self.assertRaises(ValueError):
            AgentPlan.from_dict({"goal": "inspect", "steps": [{"id": "s1", "agent": "x", "tool": "arbitrary", "reason": "bad"}]}, {"inspect"})
        llm_plan = PlannerAgent(llm_client=FakeLLM(), tool_descriptions={"inspect": "test"}).plan("inspect")
        self.assertEqual(llm_plan.to_dict()["steps"][0]["tool"], "inspect")

    def test_tool_input_schema_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            ToolInput.parse({"unexpected": "value"})
        registry = ToolRegistry()
        with self.assertRaises(ValueError):
            registry.execute("train_model", {"exec": "print(1)"}, self.df, {})
        with self.assertRaises(ValueError):
            registry.execute("train_model", {"target_col": 5}, self.df, {})

    def test_engine_execution_state_and_critic(self):
        tools = RecordingTools()
        planner = PlannerAgent(tool_descriptions=tools.list_tools())
        engine = ExecutionEngine(planner=planner, tools=tools, critic=CriticAgent(), state_store=self.state_store)
        result = engine.run(self.df, "inspect the schema", task_id="task-1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(tools.calls), 1)
        saved = self.state_store.get("task-1")
        self.assertEqual(saved["status"], "completed")
        self.assertTrue(saved["completed_steps"])
        self.assertEqual(saved["critic_result"]["status"], "ACCEPT")

    def test_retry_and_retry_limit(self):
        tools = RecordingTools()
        engine = ExecutionEngine(planner=PlannerAgent(tool_descriptions=tools.list_tools()), tools=tools,
                                  critic=RetryCritic(count=1), state_store=self.state_store, max_retries=1)
        result = engine.run(self.df, "inspect", task_id="retry")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["state"]["retry_count"], 1)

        tools2 = RecordingTools()
        engine2 = ExecutionEngine(planner=PlannerAgent(tool_descriptions=tools2.list_tools()), tools=tools2,
                                  critic=RetryCritic(count=10), state_store=self.state_store, max_retries=1)
        failed = engine2.run(self.df, "inspect", task_id="retry-limit")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["state"]["retry_count"], 1)

    def test_critic_result_validation(self):
        self.assertEqual(CriticResult("accept", "Looks good").status, "ACCEPT")
        with self.assertRaises(ValueError):
            CriticResult("maybe", "Unknown")

    def test_a2a_message_contract_and_compatibility(self):
        bus = A2ABus(persist=False)
        sent = bus.send("planner", "critic", "review", {"ok": True}, task_id="task-7")
        message = sent["message"]
        for field in ("id", "task_id", "sender", "receiver", "type", "payload", "timestamp", "status"):
            self.assertIn(field, message)
        self.assertEqual(bus.get_inbox("critic")[0]["task_id"], "task-7")


if __name__ == "__main__":
    unittest.main()
