"""Bounded local planner/tool/critic loop."""
import uuid

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from core.agentic_schemas import AgentMessage
from core.task_state import TaskStateStore, json_safe
from core.tool_registry import ToolRegistry


class ExecutionEngine:
    def __init__(self, planner=None, tools=None, critic=None, state_store=None, a2a_bus=None, max_retries=2):
        self.tools = tools or ToolRegistry()
        self.planner = planner or PlannerAgent(tool_descriptions=self.tools.list_tools())
        self.critic = critic or CriticAgent()
        self.state_store = state_store or TaskStateStore()
        self.a2a_bus = a2a_bus
        self.max_retries = max(0, int(max_retries))

    def run(self, df, user_request, task_id=None):
        task_id = task_id or str(uuid.uuid4())
        state = {"task_id": task_id, "user_request": user_request, "plan": None, "current_step": None,
                 "completed_steps": [], "tool_outputs": {}, "critic_result": None, "status": "planning",
                 "errors": [], "retry_count": 0}
        results = {}
        critic_hint = None
        while True:
            plan = self.planner.plan(user_request, list(df.columns), results, critic_hint)
            state["plan"] = plan.to_dict()
            state["status"] = "running"
            self.state_store.save(state)
            for step in plan.steps:
                state["current_step"] = step.id
                try:
                    output = self.tools.execute(step.tool, step.inputs, df,
                                                {"results": results, "latest_model": results.get("train_model")})
                    results[step.tool] = json_safe(output)
                    state["tool_outputs"][step.id] = {"tool": step.tool, "output": results[step.tool]}
                    if isinstance(output, dict) and output.get("status") == "error":
                        state["errors"].append({"step": step.id, "error": output.get("error", "Tool failed")})
                    else:
                        state["completed_steps"].append(step.id)
                    if self.a2a_bus:
                        self.a2a_bus.publish(from_agent=step.agent, to="critic", topic="tool.completed",
                                             payload={"task_id": task_id, "step_id": step.id, "tool": step.tool,
                                                      "output": results[step.tool]}, task_id=task_id)
                except Exception as exc:
                    error = str(exc)
                    state["errors"].append({"step": step.id, "error": error})
                    results[step.tool] = {"status": "error", "error": error}
                self.state_store.save(state)
            latest = results.get("evaluate_model") or results.get("train_model") or next(reversed(results.values()), {})
            critic = self.critic.review(latest, errors=state["errors"], tool_output=latest)
            state["critic_result"] = critic.to_dict()
            if critic.status == "ACCEPT":
                state["status"] = "completed"
                self.state_store.save(state)
                return {"task_id": task_id, "status": state["status"], "results": results, "state": state}
            if critic.status == "REJECT" or state["retry_count"] >= self.max_retries:
                state["status"] = "failed"
                self.state_store.save(state)
                return {"task_id": task_id, "status": state["status"], "results": results, "state": state}
            state["retry_count"] += 1
            state["status"] = "replanning"
            state["errors"] = []
            critic_hint = critic.to_dict()
            self.state_store.save(state)
