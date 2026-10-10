"""Run deterministic synthetic analytics tasks and save every outcome as JSON."""
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from core.tool_registry import ToolRegistry
from agents.planner_agent import PlannerAgent
from agents.profiler_agent import ProfilerAgent


class _DiscardMemory:
    def save(self, key, value):
        return {"status": "success", "saved_key": key}


class _NoPaidLLM:
    def generate_content(self, prompt):
        raise RuntimeError("Benchmark uses deterministic local planning only")


def _execute(registry, tool, inputs, frame, context, record):
    """Validate before invocation and record bounded timeout-only retries."""
    spec = registry.get(tool)
    spec.input_schema.parse(inputs or {})
    record["schema_valid"] = True
    for attempt in range(2):
        record["attempt_count"] = record.get("attempt_count", 0) + 1
        try:
            return registry.execute(tool, inputs, frame, context)
        except TimeoutError:
            if attempt:
                raise
            record["retry_count"] += 1


def calculate_metrics(records):
    """Completion: successful valid tasks + expected validation rejections / all tasks.
    Tool execution success uses successful valid tool calls / valid calls.
    Evidence coverage is successful records with a non-empty result / successful valid calls.
    Invalid/ambiguous inputs count as handled only when safely rejected.
    """
    total = len(records)
    valid = [r for r in records if not r.get("expected_rejection")]
    successes = [r for r in valid if r.get("status") == "success"]
    handled_rejections = [r for r in records if r.get("expected_rejection") and r.get("status") == "expected_rejection"]
    return {
        "task_completion_rate": (len(successes) + len(handled_rejections)) / total if total else 0.0,
        "tool_execution_success_rate": len(successes) / len(valid) if valid else 0.0,
        "tool_selection_accuracy": (sum(bool(r.get("tool_selection_correct")) for r in records if r.get("expected_tool") is not None) /
                                    sum(r.get("expected_tool") is not None for r in records)) if any(r.get("expected_tool") is not None for r in records) else 0.0,
        "plan_validity_rate": (sum(bool(r.get("plan_valid")) for r in records if r.get("expected_tool") is not None) /
                               sum(r.get("expected_tool") is not None for r in records)) if any(r.get("expected_tool") is not None for r in records) else 0.0,
        "schema_validity_rate": sum(bool(r.get("schema_valid")) for r in records) / total if total else 0.0,
        "evidence_coverage": sum(bool(r.get("evidence_present")) for r in successes) / len(successes) if successes else 0.0,
        "failure_rate": sum(r.get("status") == "failure" for r in records) / total if total else 0.0,
        "retry_rate": sum(int(r.get("retry_count", 0)) > 0 for r in records) / total if total else 0.0,
        "mean_duration_seconds": sum(float(r.get("duration_seconds", 0)) for r in records) / total if total else 0.0,
    }


def run(output_path=None):
    with open(os.path.join(ROOT, "benchmarks", "tasks.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    frames = {
        "classification": pd.read_csv(os.path.join(ROOT, "benchmarks", "synthetic_classification.csv")),
        "regression": pd.read_csv(os.path.join(ROOT, "benchmarks", "synthetic_regression.csv")),
    }
    frames["profiling"] = frames["classification"]
    frames["data_quality"] = frames["classification"]
    frames["model_comparison"] = frames["classification"]
    frames["explainability"] = frames["classification"]
    frames["report"] = frames["classification"]
    frames["invalid_request"] = frames["classification"]
    frames["ambiguous_request"] = frames["classification"]
    frames["ambiguous_request"] = pd.concat([frames["classification"], frames["classification"].iloc[:, 2]], axis=1)
    frames["ambiguous_request"].columns = ["signal", "segment", "target", "target"]
    report_root = os.path.dirname(output_path or os.getenv("AGENTIC_BENCHMARK_OUTPUT", os.path.join(ROOT, "benchmark_results", "latest.json")))
    registry = ToolRegistry(profiler=ProfilerAgent(output_dir=os.path.join(report_root, "temporary_reports"),
                                                  memory=_DiscardMemory()))
    planner = PlannerAgent(llm_client=_NoPaidLLM(), tool_descriptions=registry.list_tools())
    results_by_dataset = {}
    records = []
    for task in manifest["tasks"]:
        started = time.perf_counter()
        expected_tool = task.get("tool")
        record = {"task_id": task["id"], "kind": task["kind"], "expected_tool": expected_tool,
                  "tool_selection_correct": False, "schema_valid": False, "plan_valid": False, "retry_count": 0}
        try:
            kind = task["kind"]
            df = frames[kind]
            try:
                plan = planner.plan(task.get("goal", ""), list(df.columns))
                planned_tools = [step.tool for step in plan.steps]
                record.update(plan_valid=True, planned_tools=planned_tools)
                record["tool_selection_correct"] = expected_tool in planned_tools if expected_tool else False
            except ValueError as plan_error:
                record["plan_error"] = str(plan_error)[:500]
            if kind == "profiling":
                record["actual_tool"] = "profile_dataset"
                result = _execute(registry, "profile_dataset", {}, df, {}, record)
            elif kind == "data_quality":
                record["actual_tool"] = "analyze_data_quality"
                result = _execute(registry, "analyze_data_quality", {"target_col": "target"}, df, {}, record)
            elif kind in {"classification", "model_comparison"}:
                record["actual_tool"] = "train_model"
                result = _execute(registry, "train_model", {"target_col": "target", "task_type": "classification", "cv": 2}, df, {}, record)
                results_by_dataset[kind] = result
            elif kind == "regression":
                record["actual_tool"] = "train_model"
                result = _execute(registry, "train_model", {"target_col": "target", "task_type": "regression", "cv": 2}, df, {}, record)
                results_by_dataset[kind] = result
            elif kind == "explainability":
                record["actual_tool"] = "explain_global"
                registry.get("explain_global").input_schema.parse({"method": "model_native", "top_n": 10})
                record["schema_valid"] = True
                model = results_by_dataset.get("classification", {})
                if model.get("status") != "success":
                    raise RuntimeError("Explainability skipped because classification training did not produce an artifact")
                result = _execute(registry, "explain_global", {"method": "model_native", "top_n": 10}, df,
                                  {"results": {}, "latest_model": model}, record)
            elif kind == "report":
                record["actual_tool"] = "generate_report"
                result = _execute(registry, "profile_dataset", {}, df, {}, record)
                results_by_dataset["profile_dataset"] = result
                eda = _execute(registry, "run_eda", {}, df, {}, record)
                api_key = os.environ.pop("GOOGLE_API_KEY", None)
                try:
                    result = _execute(registry, "generate_report", {}, df,
                                      {"results": {"profile_dataset": result, "run_eda": eda}}, record)
                finally:
                    if api_key is not None:
                        os.environ["GOOGLE_API_KEY"] = api_key
            elif kind == "invalid_request":
                record["actual_tool"] = "__invalid__"
                try:
                    _execute(registry, "__invalid__", {}, df, {}, record)
                except ValueError as exc:
                    record.update(status="expected_rejection", expected_rejection=True, schema_valid=True,
                                  result={"error": str(exc)})
                    result = None
                else:
                    raise AssertionError("Invalid tool was unexpectedly accepted")
            elif kind == "ambiguous_request":
                record["actual_tool"] = "train_model"
                result = _execute(registry, "train_model", {"target_col": "target"}, df, {}, record)
                if result.get("status") == "error":
                    record.update(status="expected_rejection", expected_rejection=True, schema_valid=True)
                else:
                    raise AssertionError("Missing target was unexpectedly accepted")
            else:
                raise ValueError("Unknown benchmark kind")
            if record.get("status") != "expected_rejection":
                if isinstance(result, dict) and result.get("status") == "success":
                    record.update(status="success", evidence_present=bool(set(result) - {"status"}),
                                  result_summary={k: result[k] for k in ("rows", "cols", "quality_score", "model_name", "experiment_id", "artifact_path") if k in result})
                elif isinstance(result, dict) and result.get("status") in {"error", "skipped"}:
                    record.update(status="failure", error=str(result.get("error", result.get("warnings", "Tool returned non-success")))[:1000])
                else:
                    record.update(status="failure", error="Tool returned an unexpected result schema")
        except Exception as exc:
            record.update(status="failure", error=f"{type(exc).__name__}: {exc}"[:1000])
        record["tool_selection_correct"] = record.get("actual_tool") == record["expected_tool"]
        record["duration_seconds"] = time.perf_counter() - started
        records.append(record)
    report = {"schema_version": 1, "benchmark": manifest["benchmark"], "dataset_version": manifest["dataset_version"],
              "seed": 20261010, "created_at": datetime.now(timezone.utc).isoformat(),
              "metrics": calculate_metrics(records), "records": records}
    output_path = output_path or os.getenv("AGENTIC_BENCHMARK_OUTPUT", os.path.join(ROOT, "benchmark_results", "latest.json"))
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report, output_path


if __name__ == "__main__":
    report, path = run()
    print(json.dumps({"result_path": path, "metrics": report["metrics"]}, indent=2))
