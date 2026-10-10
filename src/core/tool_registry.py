"""Allowlisted, typed tool boundary used by the local execution engine."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, Optional, get_args, get_type_hints



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
            if field_name == "exclude_features" and isinstance(field_value, list) and not all(isinstance(item, str) for item in field_value):
                raise ValueError("Tool input 'exclude_features' must be a list of strings")
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
    task_type: Optional[str] = None
    primary_metric: Optional[str] = None
    cv: Optional[int] = None
    tune: Optional[bool] = None
    n_iter: Optional[int] = None
    exclude_features: Optional[list] = None


@dataclass
class DataQualityInput(ToolInput):
    target_col: Optional[str] = None


@dataclass
class GlobalExplainInput(ToolInput):
    method: Optional[str] = None
    top_n: Optional[int] = None


@dataclass
class LocalExplainInput(ToolInput):
    row_index: Optional[int] = None
    method: Optional[str] = None
    top_n: Optional[int] = None


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
    def __init__(self, model_tools=None, dataset_tools=None, profiler=None, eda=None, quality_agent=None, explainability_agent=None):
        # Keep profiling/quality/MCP discovery available if the optional ML stack is absent.
        self.model_tools = model_tools
        self.model_agent = None
        self.dataset_tools = dataset_tools
        self.profiler = profiler
        self.eda = eda
        self.quality_agent = quality_agent
        self.explainability_agent = explainability_agent
        self._tools: Dict[str, ToolSpec] = {
            "profile_dataset": ToolSpec("profile_dataset", "Profile rows, types, and missingness.", EmptyInput, ToolOutput,
                                         lambda _, df, __: self._profile(df)),
            "inspect_schema": ToolSpec("inspect_schema", "Inspect column names, types, and dataset shape.", EmptyInput, ToolOutput,
                                        lambda _, df, __: self._dataset_tool("summary", df)),
            "detect_target": ToolSpec("detect_target", "Suggest target candidates from column cardinality and types.", EmptyInput, ToolOutput,
                                      self._detect_target),
            "analyze_data_quality": ToolSpec("analyze_data_quality", "Inspect missingness, duplicates, leakage, outliers, and class balance.", DataQualityInput, ToolOutput,
                                              self._analyze_data_quality),
            "analyze_missing_values": ToolSpec("analyze_missing_values", "Count missing values by column.", EmptyInput, ToolOutput,
                                                lambda _, df, __: self._dataset_tool("missing_values", df)),
            "run_eda": ToolSpec("run_eda", "Run descriptive statistics, correlations, and outlier analysis.", EmptyInput, ToolOutput,
                                lambda _, df, __: self._eda(df)),
            "train_model": ToolSpec("train_model", "Train a selected, preprocessed scikit-learn model.", TrainInput, ToolOutput,
                                    lambda inputs, df, context: self._train_model(inputs, df, context)),
            "evaluate_model": ToolSpec("evaluate_model", "Summarize the latest model metrics and feature importances.", EmptyInput, ToolOutput,
                                       self._evaluate_model),
            "explain_global": ToolSpec("explain_global", "Compute global model feature importance with SHAP or grounded fallbacks.", GlobalExplainInput, ToolOutput,
                                       self._explain_global),
            "explain_prediction": ToolSpec("explain_prediction", "Compute actual feature contributions for one saved-model prediction.", LocalExplainInput, ToolOutput,
                                           self._explain_prediction),
            "generate_report": ToolSpec("generate_report", "Create a notebook report from existing analysis results.", EmptyInput, ToolOutput,
                                        self._generate_report),
        }

    @staticmethod
    def _detect_target(_, df, context=None):
        candidates = []
        for position, name in enumerate(df.columns):
            series = df.iloc[:, position]
            unique = int(series.nunique(dropna=True))
            ratio = unique / max(len(df), 1)
            if unique >= 2 and ratio <= 0.5:
                candidates.append({"column": str(name), "unique_values": unique,
                                   "suggested_task": "classification" if unique <= 20 else "regression"})
        return {"status": "success", "candidates": candidates,
                "suggestion": candidates[-1]["column"] if candidates else None,
                "note": "Candidates are suggestions; specify target_col to choose explicitly."}

    def _analyze_data_quality(self, inputs, df, context):
        if self.quality_agent is None:
            from agents.data_quality_agent import DataQualityAgent
            self.quality_agent = DataQualityAgent()
        target_col = inputs.target_col
        detection = context.get("results", {}).get("detect_target", {})
        target_col = target_col or detection.get("suggestion")
        return self.quality_agent.analyze(df, target_col=target_col)

    def _train_model(self, inputs, df, context):
        if not inputs.target_col:
            detection = context.get("results", {}).get("detect_target") or self._detect_target(None, df)
            inputs.target_col = detection.get("suggestion")
        if not inputs.target_col:
            return {"status": "error", "error": "No target candidate found; provide target_col explicitly."}
        if inputs.target_col not in df.columns or int((df.columns == inputs.target_col).sum()) != 1:
            return {"status": "error", "error": f"Target column '{inputs.target_col}' is missing or ambiguous."}
        if self.model_tools is None:
            from tools.model_tools import ModelTools
            self.model_tools = ModelTools()
        if self.model_agent is None:
            from agents.model_agent import ModelAgent
            self.model_agent = ModelAgent(model_tools=self.model_tools)
        decision = self.model_agent.choose_strategy(df, inputs.target_col)
        model_name = inputs.model_name if inputs.model_name is not None else decision["model_name"]
        class_weight = inputs.class_weight if inputs.class_weight is not None else decision["class_weight"]
        result = self.model_tools.train(df, inputs.target_col, model_name=model_name,
                                        class_weight=class_weight, task_type=inputs.task_type,
                                        primary_metric=inputs.primary_metric, cv=inputs.cv,
                                        tune=bool(inputs.tune), n_iter=inputs.n_iter or 8,
                                        exclude_features=inputs.exclude_features)
        result["target_col"] = inputs.target_col
        result["agent_decision"] = {**decision, "model_name": model_name, "class_weight": class_weight}
        return result

    @staticmethod
    def _evaluate_model(_, df, context):
        model = context.get("latest_model")
        if not model:
            return {"status": "error", "error": "No model output is available to evaluate."}
        metrics = model.get("metrics", {})
        importance = context.get("results", {}).get("explain_global", model.get("feature_importance", {}))
        return {"status": "success", "metrics": metrics, "feature_importance": importance,
                "model_name": model.get("model_name"), "task_type": model.get("task_type")}

    def _explain_global(self, inputs, df, context):
        if self.explainability_agent is None:
            from agents.explainability_agent import ExplainabilityAgent
            self.explainability_agent = ExplainabilityAgent()
        model = context.get("latest_model") or {}
        path = model.get("artifact_path") or model.get("model_path")
        if not path:
            return {"status": "error", "error": "No saved model artifact is available to explain."}
        result = self.explainability_agent.explain_global(path, df, method=inputs.method or "auto",
                                                          target=model.get("target"), top_n=inputs.top_n or 30)
        if "error" in result:
            return {"status": "skipped", "method": "unavailable", "features": [],
                    "warnings": [result["error"], *result.get("warnings", [])]}
        result.setdefault("status", "success")
        return result

    def _explain_prediction(self, inputs, df, context):
        if self.explainability_agent is None:
            from agents.explainability_agent import ExplainabilityAgent
            self.explainability_agent = ExplainabilityAgent()
        model = context.get("latest_model") or {}
        path = model.get("artifact_path") or model.get("model_path")
        if not path:
            return {"status": "error", "error": "No saved model artifact is available to explain."}
        row_index = 0 if inputs.row_index is None else inputs.row_index
        if row_index < 0 or row_index >= len(df):
            return {"status": "error", "error": "row_index is outside the dataset."}
        result = self.explainability_agent.explain_prediction(path, df.iloc[row_index], df,
                                                              method=inputs.method or "auto", top_n=inputs.top_n or 10)
        if "error" in result:
            return {"status": "skipped", "method": "unavailable", "prediction": None,
                    "baseline_prediction": None, "contributions": [],
                    "warnings": [result["error"], *result.get("warnings", [])]}
        result.setdefault("status", "success")
        return result

    @staticmethod
    def _generate_report(_, df, context):
        from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent
        results = context.get("results", {})
        result = NotebookSynthesizerAgent().run(results.get("profile_dataset"), results.get("run_eda"),
                                                results.get("train_model"), results.get("evaluate_model"))
        return result

    def _dataset_tool(self, method, df):
        if self.dataset_tools is None:
            from tools.dataset_tools import DatasetTools
            self.dataset_tools = DatasetTools()
        return getattr(self.dataset_tools, method)(df)

    def _profile(self, df):
        if self.profiler is None:
            from agents.profiler_agent import ProfilerAgent
            self.profiler = ProfilerAgent()
        return self.profiler.run(df)

    def _eda(self, df):
        if self.eda is None:
            from agents.eda_agent import EDAAgent
            self.eda = EDAAgent()
        return self.eda.run(df)

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
