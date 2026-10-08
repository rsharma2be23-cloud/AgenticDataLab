"""Allowlisted, typed tool boundary used by the local execution engine."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, Optional, get_args, get_type_hints

from tools.dataset_tools import DatasetTools
from tools.model_tools import ModelTools
from agents.profiler_agent import ProfilerAgent
from agents.eda_agent import EDAAgent
from agents.model_agent import ModelAgent


@dataclass
class ToolInput:
    """Base class for lightweight structured tool input schemas."""
    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Tool inputs must be an object")
        annotations = get_type_hints(cls)
        allowed = set(annotations)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unexpected tool input(s): {', '.join(sorted(unknown))}")
        for field_name, field_value in value.items():
            expected = annotations[field_name]
            choices = get_args(expected)
            nullable = type(None) in choices
            valid_types = tuple(item for item in choices if item is not type(None)) if choices else (expected,)
            if field_value is None and nullable:
                continue
            if not any(isinstance(field_value, item) for item in valid_types if isinstance(item, type)):
                expected_names = " or ".join(item.__name__ for item in valid_types if isinstance(item, type))
                raise ValueError(f"Tool input '{field_name}' must be {expected_names}")
        return cls(**value)


@dataclass
class EmptyInput(ToolInput):
    pass


@dataclass
class TrainInput(ToolInput):
    target_col: Optional[str] = None
    model_name: Optional[str] = None
    class_weight: Optional[str] = None


@dataclass
class ToolOutput:
    """Shared result contract; tool-specific fields remain in the payload dict."""
    status: str
    payload: Dict[str, Any]

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or value.get("status") not in {"success", "error", "skipped"}:
            raise ValueError("Tool output must be an object with success, error, or skipped status")
        if value.get("status") == "error" and not isinstance(value.get("error"), str):
            raise ValueError("Error tool output must include a string error")
        return cls(status=value["status"], payload=value)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Type[ToolInput]
    output_schema: Type[ToolOutput]
    handler: Callable


class ToolRegistry:
    def __init__(self, model_tools=None, dataset_tools=None, profiler=None, eda=None):
        self.model_tools = model_tools or ModelTools()
        self.model_agent = ModelAgent(model_tools=self.model_tools)
        self.dataset_tools = dataset_tools or DatasetTools()
        self.profiler = profiler or ProfilerAgent()
        self.eda = eda or EDAAgent()
        self._tools: Dict[str, ToolSpec] = {
            "profile_dataset": ToolSpec("profile_dataset", "Profile rows, types, and missingness.", EmptyInput, ToolOutput,
                                         lambda _, df, __: self.profiler.run(df)),
            "inspect_schema": ToolSpec("inspect_schema", "Inspect column names, types, and dataset shape.", EmptyInput, ToolOutput,
                                        lambda _, df, __: self.dataset_tools.summary(df)),
            "detect_target": ToolSpec("detect_target", "Suggest target candidates from column cardinality and types.", EmptyInput, ToolOutput,
                                      self._detect_target),
            "analyze_missing_values": ToolSpec("analyze_missing_values", "Count missing values by column.", EmptyInput, ToolOutput,
                                                lambda _, df, __: self.dataset_tools.missing_values(df)),
            "run_eda": ToolSpec("run_eda", "Run descriptive statistics, correlations, and outlier analysis.", EmptyInput, ToolOutput,
                                lambda _, df, __: self.eda.run(df)),
            "train_model": ToolSpec("train_model", "Train a selected, preprocessed scikit-learn model.", TrainInput, ToolOutput,
                                    lambda inputs, df, __: self._train_model(inputs, df)),
            "evaluate_model": ToolSpec("evaluate_model", "Summarize the latest model metrics and feature importances.", EmptyInput, ToolOutput,
                                       self._evaluate_model),
            "generate_report": ToolSpec("generate_report", "Create a notebook report from existing analysis results.", EmptyInput, ToolOutput,
                                        self._generate_report),
        }

    @staticmethod
    def _detect_target(_, df, context=None):
        candidates = []
        for name in df.columns:
            unique = int(df[name].nunique(dropna=True))
            ratio = unique / max(len(df), 1)
            if unique >= 2 and ratio <= 0.5:
                candidates.append({"column": str(name), "unique_values": unique,
                                   "suggested_task": "classification" if unique <= 20 else "regression"})
        return {"status": "success", "candidates": candidates,
                "suggestion": candidates[-1]["column"] if candidates else None,
                "note": "Candidates are suggestions; specify target_col to choose explicitly."}

    def _train_model(self, inputs, df):
        if not inputs.target_col:
            detection = self._detect_target(None, df)
            inputs.target_col = detection.get("suggestion")
        if not inputs.target_col:
            return {"status": "error", "error": "No target candidate found; provide target_col explicitly."}
        decision = self.model_agent.choose_strategy(df, inputs.target_col)
        model_name = inputs.model_name if inputs.model_name is not None else decision["model_name"]
        class_weight = inputs.class_weight if inputs.class_weight is not None else decision["class_weight"]
        result = self.model_tools.train(df, inputs.target_col, model_name=model_name,
                                        class_weight=class_weight)
        result["target_col"] = inputs.target_col
        result["agent_decision"] = {**decision, "model_name": model_name, "class_weight": class_weight}
        return result

    @staticmethod
    def _evaluate_model(_, df, context):
        model = context.get("latest_model")
        if not model:
            return {"status": "error", "error": "No model output is available to evaluate."}
        metrics = model.get("metrics", {})
        importance = model.get("feature_importance", {})
        return {"status": "success", "metrics": metrics, "feature_importance": importance,
                "model_name": model.get("model_name"), "task_type": model.get("task_type")}

    @staticmethod
    def _generate_report(_, df, context):
        from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent
        results = context.get("results", {})
        result = NotebookSynthesizerAgent().run(results.get("profile_dataset"), results.get("run_eda"),
                                                results.get("train_model"), results.get("evaluate_model"))
        return result

    def get(self, name):
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_tools(self):
        return {name: spec.description for name, spec in self._tools.items()}

    def execute(self, name, inputs, df, context):
        spec = self.get(name)
        parsed = spec.input_schema.parse(inputs or {})
        # Handler signatures remain explicit and allowlisted. No user/LLM code is evaluated.
        output = spec.handler(parsed, df, context)
        spec.output_schema.parse(output)
        return output
