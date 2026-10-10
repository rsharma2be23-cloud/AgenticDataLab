"""Bounded local planner/tool/critic loop with traces and approval gates."""
import os
import time
import uuid

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from core.agentic_schemas import PlanStep
from core.agentic_schemas import AgentPlan
from core.task_state import AnalyticalMemoryStore, TaskStateStore, json_safe
from core.tool_registry import ToolRegistry
from core.tracing import ExecutionTracer


class ExecutionEngine:
    def __init__(self, planner=None, tools=None, critic=None, state_store=None, a2a_bus=None, max_retries=2,
                 tracer=None, approval_required=None, memory_store=None):
        self.tools = tools or ToolRegistry()
        self.planner = planner or PlannerAgent(tool_descriptions=self.tools.list_tools())
        self.critic = critic or CriticAgent()
        self.state_store = state_store or TaskStateStore()
        self.a2a_bus = a2a_bus
        self.max_retries = max(0, min(10, int(max_retries)))
        self.tracer = tracer or ExecutionTracer()
        self.memory_store = memory_store or AnalyticalMemoryStore()
        if approval_required is None:
            approval_required = [v.strip() for v in os.getenv("AGENTIC_APPROVAL_TOOLS", "").split(",") if v.strip()]
        self.approval_required = approval_required

    def _requires_approval(self, step):
        return bool(self.approval_required(step) if callable(self.approval_required)
                    else step.tool in set(self.approval_required or []))

    def resolve_approval(self, task_id, step_id, approved, actor="user"):
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean")
        state = self.state_store.get(task_id)
        if not state or state.get("status") != "pending_approval":
            raise ValueError("Task is not waiting for approval")
        if step_id not in state.get("approvals", {}):
            raise ValueError("Step has no pending approval")
        state["approvals"][step_id] = {"status": "approved" if approved else "rejected", "actor": str(actor)[:100]}
        state["status"] = "approved" if approved else "rejected"
        self.state_store.save(state)
        self.tracer.record({"task_id": task_id, "correlation_id": task_id, "event": "approval_decision",
                            "step_id": step_id, "status": state["status"], "actor": str(actor)[:100]})
        return state

    def run(self, df, user_request, task_id=None):
        task_id = task_id or str(uuid.uuid4())
        import hashlib
        dataset_metadata = {"rows": int(len(df)), "columns": int(len(df.columns)),
                            "column_names": [str(x) for x in df.columns[:200]],
                            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}}
        dataset_metadata["signature"] = hashlib.sha256(
            (str(dataset_metadata["rows"]) + "|" + "|".join(dataset_metadata["column_names"]) + "|" +
             "|".join(dataset_metadata["dtypes"].values())).encode("utf-8")).hexdigest()
        prior = self.state_store.get(task_id)
        resuming = bool(prior and prior.get("status") in {"pending_approval", "approved"})
        if prior and prior.get("status") == "rejected":
            return {"task_id": task_id, "status": "rejected", "results": {}, "state": prior}
        state = {"task_id": task_id, "user_request": user_request, "plan": None, "current_step": None,
                 "completed_steps": [], "tool_outputs": {}, "critic_result": None, "status": "planning",
                 "errors": [], "retry_count": 0, "approvals": {}, "execution_history": []}
        state["dataset_metadata"] = dataset_metadata
        try:
            related = self.memory_store.retrieve(dataset_metadata["signature"], limit=5)
            state["related_memory_ids"] = [item.get("task_id") for item in related if item.get("task_id")]
        except Exception:
            state["related_memory_ids"] = []
        results, critic_hint = {}, None
        resume_plan = None
        if resuming:
            state = prior
            state["dataset_metadata"] = dataset_metadata
            resume_plan = AgentPlan.from_dict(prior.get("plan"), allowed_tools=set(self.tools.list_tools()))
            for record in prior.get("tool_outputs", {}).values():
                if isinstance(record, dict) and isinstance(record.get("tool"), str):
                    results[record["tool"]] = record.get("output")
        while True:
            plan = resume_plan or self.planner.plan(user_request, list(df.columns), results, critic_hint)
            resume_plan = None
            state["plan"] = plan.to_dict()
            state["workflow_id"] = task_id
            state["status"] = "running"
            self.state_store.save(state)
            steps = list(plan.steps)
            index, quality_replanned = 0, False
            if resuming:
                while index < len(steps) and steps[index].id in state.get("completed_steps", []):
                    index += 1
            while index < len(steps):
                step = steps[index]
                state["current_step"] = step.id
                if self._requires_approval(step):
                    approval = state["approvals"].get(step.id, {})
                    if approval.get("status") != "approved":
                        status = "rejected" if approval.get("status") == "rejected" else "pending_approval"
                        state["status"] = status
                        if status == "pending_approval":
                            state["approvals"][step.id] = {"status": status, "tool": step.tool}
                        self.state_store.save(state)
                        self.tracer.record({"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                                            "event": "approval_gate", "step_id": step.id, "tool": step.tool,
                                            "status": status})
                        return {"task_id": task_id, "status": status, "results": results, "state": state}
                started_wall = time.time()
                started = time.perf_counter()
                try:
                    output = self.tools.execute(step.tool, step.inputs, df,
                                                {"results": results, "latest_model": results.get("train_model")})
                    results[step.tool] = json_safe(output)
                    state["tool_outputs"][step.id] = {"tool": step.tool, "output": results[step.tool]}
                    if isinstance(output, dict) and output.get("status") == "error":
                        state["errors"].append({"step": step.id, "error": output.get("error", "Tool failed")})
                    else:
                        state["completed_steps"].append(step.id)
                except Exception as exc:
                    error = str(exc)[:1000]
                    state["errors"].append({"step": step.id, "error": error})
                    results[step.tool] = {"status": "error", "error": error}
                duration = time.perf_counter() - started
                failed = isinstance(results.get(step.tool), dict) and results[step.tool].get("status") == "error"
                trace = {"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                         "event": "tool_execution", "agent_name": step.agent, "tool": step.tool,
                         "step_id": step.id, "started_at_epoch": started_wall,
                         "ended_at_epoch": started_wall + duration, "duration_seconds": duration,
                         "status": "failed" if failed else "succeeded", "retry_count": state["retry_count"],
                         "error": state["errors"][-1]["error"] if failed else None}
                self.tracer.record(trace)
                state["execution_history"].append({k: v for k, v in trace.items() if k != "error"})
                if self.a2a_bus:
                    self.a2a_bus.publish(from_agent=step.agent, to="critic", topic="tool.completed",
                                         payload={"task_id": task_id, "step_id": step.id, "tool": step.tool,
                                                  "status": trace["status"]}, task_id=task_id, correlation_id=task_id)
                self.state_store.save(state)
                index += 1
                quality_result = results.get("analyze_data_quality", {})
                recommendations = quality_result.get("recommendations", []) if isinstance(quality_result, dict) else []
                pending_model = any(item.tool == "train_model" for item in steps[index:])
                if (not quality_replanned and step.tool == "analyze_data_quality" and pending_model and
                        any(token in str(item).lower() for item in recommendations
                            for token in ("class_weight=balanced", "possible leakage"))):
                    hint = {"status": "RETRY", "reason": "Data quality recommends class-aware training.",
                            "recommendations": recommendations, "issues": quality_result.get("warnings", [])}
                    revised = self.planner.plan(user_request, list(df.columns), results, hint)
                    prefix = steps[:index]
                    revised_steps = [PlanStep(id=f"step_{len(prefix) + pos + 1}", agent=item.agent, tool=item.tool,
                                              reason=item.reason, inputs=item.inputs)
                                     for pos, item in enumerate(revised.steps)]
                    steps = prefix + revised_steps
                    state["plan"] = {"goal": revised.goal, "steps": [item.__dict__ for item in steps]}
                    state["status"] = "replanning"
                    quality_replanned = True
                    self.tracer.record({"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                                        "event": "replanning", "status": "started"})
                    self.state_store.save(state)
            latest = results.get("evaluate_model") or results.get("train_model") or next(reversed(results.values()), {})
            critic = self.critic.review(results, errors=state["errors"], tool_output=latest)
            state["critic_result"] = critic.to_dict()
            if isinstance(results.get("train_model"), dict):
                results["train_model"]["critic_result"] = critic.to_dict()
                for record in state["tool_outputs"].values():
                    if record.get("tool") == "train_model" and isinstance(record.get("output"), dict):
                        record["output"]["critic_result"] = critic.to_dict()
            if critic.status == "ACCEPT":
                state["status"] = "completed"
                self.state_store.save(state)
                final_status = state["status"]
                self._remember(task_id, dataset_metadata, results)
                self.tracer.record({"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                                    "event": "workflow_finished", "status": final_status,
                                    "retry_count": state["retry_count"]})
                return {"task_id": task_id, "status": final_status, "results": results, "state": state}
            if critic.status == "REJECT" or state["retry_count"] >= self.max_retries:
                state["status"] = "failed"
                self.state_store.save(state)
                self._remember(task_id, dataset_metadata, results)
                self.tracer.record({"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                                    "event": "workflow_finished", "status": "failed",
                                    "retry_count": state["retry_count"], "error": critic.reason})
                return {"task_id": task_id, "status": "failed", "results": results, "state": state}
            state["retry_count"] += 1
            state["status"] = "replanning"
            state["errors"] = []
            critic_hint = critic.to_dict()
            self.tracer.record({"task_id": task_id, "workflow_id": task_id, "correlation_id": task_id,
                                "event": "replanning", "status": "retry", "retry_count": state["retry_count"]})
            self.state_store.save(state)

    def _remember(self, task_id, dataset_metadata, results):
        """Keep only reusable summaries, metrics, and artifact references; no prompts or rows."""
        summary = {}
        for key in ("train_model", "evaluate_model", "analyze_data_quality"):
            value = results.get(key)
            if not isinstance(value, dict):
                continue
            keep = {name: value[name] for name in (
                "status", "experiment_id", "model_name", "task_type", "target_col", "metrics", "cv_results",
                "model_comparison", "artifact_path", "quality_score", "warnings", "checks") if name in value}
            summary[key] = keep
        if not summary:
            return
        try:
            self.memory_store.add({"task_id": task_id, "dataset_signature": dataset_metadata["signature"],
                                   "dataset_metadata": dataset_metadata, "summary": summary})
        except (OSError, ValueError):
            # Memory is optional and must not fail a workflow if data is unavailable/unsafe.
            pass
