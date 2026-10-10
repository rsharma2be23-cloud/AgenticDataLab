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
        previous_results = previous_results or {}
        steps = []
        def add(agent, tool, reason, inputs=None):
            if self.tool_descriptions and tool not in self.tool_descriptions:
                return
            steps.append({"id": f"step_{len(steps) + 1}", "agent": agent, "tool": tool,
                          "reason": reason, "inputs": inputs or {}})
        if critic_result and critic_result.get("status") == "RETRY":
            recommendations = " ".join(critic_result.get("recommendations", [])).lower()
            old_model = previous_results.get("train_model", {})
            quality_target = previous_results.get("detect_target", {}).get("suggestion")
            target = old_model.get("target_col") or quality_target
            goal_target = re.search(r"\btarget\s*:\s*['\"]?([^,'\".]+)", goal, re.IGNORECASE)
            if goal_target:
                target = goal_target.group(1).strip()
            retry_inputs = {"target_col": target}
            quality = previous_results.get("analyze_data_quality", {})
            quality_checks = quality.get("checks", {}) if isinstance(quality, dict) else {}
            leakage_features = quality_checks.get("possible_target_leakage", [])
            if "leakage" in recommendations and leakage_features:
                retry_inputs["exclude_features"] = leakage_features
            if "class_weight=balanced" in recommendations or "class imbalance" in recommendations:
                retry_inputs["class_weight"] = "balanced"
            if "recall" in recommendations and ("primary metric" in recommendations or "use recall" in recommendations):
                retry_inputs["primary_metric"] = "recall"
            elif "pr-auc" in recommendations or "pr_auc" in recommendations:
                retry_inputs["primary_metric"] = "pr_auc"
            if "tune" in recommendations or "hyperparameter" in recommendations:
                retry_inputs.update({"tune": True, "n_iter": 6})
            if old_model.get("model_key"):
                retry_inputs["model_name"] = "logreg" if old_model["model_key"] != "logreg" else "rf"
            add("model", "train_model", "Run a revised experiment using the critic's measured recommendations.", retry_inputs)
            add("model", "evaluate_model", "Compare the revised experiment metrics.")
            if any(word in text for word in ("factor", "important", "explain", "feature")):
                add("explainability", "explain_global", "Explain the revised model using computed feature contributions.")
            if "local" in text or "prediction" in text or "row " in text:
                row_match = re.search(r"\brow(?:_index)?\s*[:#]?\s*(\d+)", text)
                add("explainability", "explain_prediction", "Explain one prediction from computed per-feature effects.",
                    {"row_index": int(row_match.group(1)) if row_match else 0})
            if not steps:
                for tool in self.tool_descriptions:
                    add("data_quality", tool, "Repeat an available deterministic tool because the suggested retry tool is unavailable.")
            return {"goal": goal, "steps": steps}

        explicit_target = re.search(r"\btarget\s*:\s*['\"]?([^,'\".]+)", goal, re.IGNORECASE)
        chosen = explicit_target.group(1).strip() if explicit_target else None
        wants_model = any(word in text for word in ("target", "factor", "predict", "model", "affecting", "important"))
        wants_quality = any(word in text for word in ("quality", "missing", "duplicate", "leakage", "imbalance"))
        wants_eda = any(word in text for word in ("eda", "explor", "analy", "distribution", "correlation"))
        wants_report = "report" in text or "notebook" in text

        if wants_model:
            if not chosen:
                add("data_quality", "detect_target", "Identify candidate targets before selecting a model task.")
            add("data_quality", "analyze_data_quality", "Check dataset risks before model fitting.", {"target_col": chosen})
            train_inputs = {"target_col": chosen}
            if "regression" in text:
                train_inputs["task_type"] = "regression"
            elif "classification" in text:
                train_inputs["task_type"] = "classification"
            if "tune" in text or "hyperparameter" in text:
                train_inputs.update({"tune": True, "n_iter": 6})
            if "recall" in text:
                train_inputs["task_type"] = "classification"
                train_inputs["primary_metric"] = "recall"
            elif "pr-auc" in text or "pr_auc" in text:
                train_inputs["task_type"] = "classification"
                train_inputs["primary_metric"] = "pr_auc"
            add("model", "train_model", "Compare cross-validated candidate models for the requested target.", train_inputs)
            add("model", "evaluate_model", "Review cross-validation and held-out metrics.")
            if any(word in text for word in ("factor", "important", "explain", "feature")):
                add("explainability", "explain_global", "Compute global feature importance from the selected model.")
            if "local" in text or "prediction" in text or "row " in text:
                row_match = re.search(r"\brow(?:_index)?\s*[:#]?\s*(\d+)", text)
                add("explainability", "explain_prediction", "Explain one prediction from computed per-feature effects.",
                    {"row_index": int(row_match.group(1)) if row_match else 0})
        elif wants_quality:
            add("data_quality", "analyze_data_quality", "Measure the requested dataset quality indicators.", {"target_col": chosen})
            if "missing" in text:
                add("data_quality", "analyze_missing_values", "Return per-column missing-value counts.")
        else:
            add("data_quality", "inspect_schema", "Inspect dataset shape, column names, and dtypes.")
            add("data_quality", "profile_dataset", "Profile dataset types and missingness.")
        if wants_eda and "run_eda" in self.tool_descriptions:
            add("eda", "run_eda", "Summarize distributions and relationships requested by the goal.")
        if wants_report:
            if "profile_dataset" in self.tool_descriptions and not any(step["tool"] == "profile_dataset" for step in steps):
                add("data_quality", "profile_dataset", "Collect dataset context for the requested report.")
            if "run_eda" in self.tool_descriptions and not any(step["tool"] == "run_eda" for step in steps):
                add("eda", "run_eda", "Collect descriptive statistics for the requested report.")
            add("report", "generate_report", "Generate a report from the collected analysis results.")
        if not steps and self.tool_descriptions:
            fallback = next(iter(self.tool_descriptions))
            add("data_quality", fallback, "Use the only tool available in this planner configuration.")
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
                      "When critic_result contains recommendations, implement them in revised tool inputs. "
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
