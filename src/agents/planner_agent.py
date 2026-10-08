"""Plans controlled tool calls, optionally using Gemini for goal-driven choices."""
import json
import os
import re

from core.agentic_schemas import AgentPlan


class PlannerAgent:
    def __init__(self, llm_client=None, tool_descriptions=None):
        self.llm_client = llm_client
        self.tool_descriptions = tool_descriptions or {}

    def _gemini_client(self):
        if self.llm_client is not None:
            return self.llm_client
        if not os.getenv("GEMINI_API_KEY"):
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            return genai.GenerativeModel("gemini-2.0-flash")
        except Exception:
            return None

    def _heuristic_plan(self, goal, columns, previous_results=None, critic_result=None):
        """Dependency-free fallback: choose analyses from the stated goal and context."""
        text = goal.lower()
        prior = previous_results or {}
        steps = []
        def add(agent, tool, reason, inputs=None):
            steps.append({"id": f"step_{len(steps) + 1}", "agent": agent, "tool": tool,
                          "reason": reason, "inputs": inputs or {}})
        if "quality" in text or "schema" in text or "column" in text:
            add("data_quality", "inspect_schema", "The goal asks about dataset structure.")
        if "missing" in text or "quality" in text or not steps:
            add("data_quality", "analyze_missing_values", "Check missingness relevant to data quality.")
        if "target" in text or "factor" in text or "predict" in text or "model" in text:
            add("data_quality", "detect_target", "Identify a target candidate for the requested factor or prediction analysis.")
            target = (critic_result or {}).get("recommendations", [])
            chosen = next((x.split(":", 1)[1].strip() for x in target if x.lower().startswith("target:")), None)
            goal_target = re.search(r"\btarget\s*:\s*['\"]?([^,'\".]+)", goal, re.IGNORECASE)
            chosen = goal_target.group(1).strip() if goal_target else chosen
            if not chosen and previous_results:
                chosen = (previous_results.get("train_model") or {}).get("target_col")
            add("model", "train_model", "Train a controlled baseline to quantify target relationships.", {"target_col": chosen})
            add("model", "evaluate_model", "Review the model metrics and feature importance.")
        if "eda" in text or "explor" in text or "analy" in text or "factor" in text or not steps:
            add("eda", "run_eda", "Summarize distributions and relationships requested by the goal.")
        if critic_result and critic_result.get("status") == "RETRY":
            # Modify retry choice based on critic guidance rather than replaying the same plan.
            recommendations = " ".join(critic_result.get("recommendations", [])).lower()
            if "logistic" in recommendations or "logreg" in recommendations:
                steps = [s for s in steps if s["tool"] != "train_model"]
                add("model", "train_model", "Critic requested an alternate logistic regression attempt.",
                    {"target_col": next((r.split(":", 1)[1].strip() for r in critic_result.get("recommendations", []) if r.lower().startswith("target:")), None) or ((previous_results or {}).get("train_model") or {}).get("target_col"), "model_name": "logreg", "class_weight": "balanced" if "balanced" in recommendations else None})
            elif "missing" in recommendations and "analyze_missing_values" not in [s["tool"] for s in steps]:
                add("data_quality", "analyze_missing_values", "Critic requested missing-value analysis.")
        return {"goal": goal, "steps": steps}

    def plan(self, goal, columns=None, previous_results=None, critic_result=None):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("A user goal is required")
        context = {"goal": goal, "columns": columns or [], "previous_results": previous_results or {},
                   "critic_result": critic_result, "available_tools": self.tool_descriptions}
        client = self._gemini_client()
        if client is not None:
            prompt = ("Create a concise JSON analysis plan only. Do not include markdown. "
                      "Each step has id, agent, tool, reason, and optional inputs. Select only available tools. "
                      "Do not run Python or request code execution. Context: " + json.dumps(context, default=str))
            try:
                response = client.generate_content(prompt)
                raw = getattr(response, "text", response)
                match = re.search(r"\{[\s\S]*\}", str(raw))
                if not match:
                    raise ValueError("Planner response did not contain JSON")
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = self._heuristic_plan(goal, columns, previous_results, critic_result)
        else:
            parsed = self._heuristic_plan(goal, columns, previous_results, critic_result)
        return AgentPlan.from_dict(parsed, allowed_tools=set(self.tool_descriptions) or None)
