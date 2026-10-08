"""Small validated schemas for local agent plans and tool calls.

Kept dependency-free so the application and unit tests do not require an LLM
provider or an additional schema package.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class PlanStep:
    id: str
    agent: str
    tool: str
    reason: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Each plan step must be an object")
        required = ("id", "agent", "tool", "reason")
        if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
            raise ValueError("Plan steps require non-empty id, agent, tool, and reason strings")
        inputs = value.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("Plan step inputs must be an object")
        return cls(*(value[key] for key in required), inputs=inputs)


@dataclass
class AgentPlan:
    goal: str
    steps: List[PlanStep]

    @classmethod
    def from_dict(cls, value, allowed_tools=None):
        if not isinstance(value, dict) or not isinstance(value.get("goal"), str) or not value["goal"].strip():
            raise ValueError("Plan requires a non-empty goal")
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Plan must contain at least one step")
        steps = [PlanStep.from_dict(item) for item in raw_steps]
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan step ids must be unique")
        if allowed_tools is not None:
            unknown = [step.tool for step in steps if step.tool not in allowed_tools]
            if unknown:
                raise ValueError(f"Plan references unavailable tools: {', '.join(unknown)}")
        return cls(goal=value["goal"], steps=steps)

    def to_dict(self):
        return {"goal": self.goal, "steps": [step.__dict__ for step in self.steps]}


@dataclass
class CriticResult:
    status: str
    reason: str
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.status = self.status.upper()
        if self.status not in {"ACCEPT", "RETRY", "REJECT"}:
            raise ValueError("Critic status must be ACCEPT, RETRY, or REJECT")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Critic reason must be a non-empty string")
        if not isinstance(self.recommendations, list) or not all(isinstance(x, str) for x in self.recommendations):
            raise ValueError("Critic recommendations must be a list of strings")

    def to_dict(self):
        return {"status": self.status, "reason": self.reason, "recommendations": self.recommendations}


@dataclass
class AgentMessage:
    task_id: str
    sender: str
    receiver: str
    type: str
    payload: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "queued"

    def to_dict(self):
        return self.__dict__.copy()
