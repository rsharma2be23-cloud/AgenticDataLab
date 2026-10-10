"""Real MCP stdio server exposing the existing allowlisted analytics tools.

The dataset argument is a basename located under AGENTIC_DATA_DIR. No arbitrary
paths, code execution, or generated Python are accepted at this boundary.
"""
import os
import sys
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import pandas as pd
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("MCP server dependencies are missing. Install requirements.txt in the project environment.") from exc

from core.tool_registry import ToolRegistry

server = FastMCP("AgenticDataLab")
registry = ToolRegistry()
DATA_ROOT = Path(os.getenv("AGENTIC_DATA_DIR", "streamlit_app_storage/uploads")).resolve()
MAX_DATASET_BYTES = max(1024, int(os.getenv("AGENTIC_MCP_MAX_DATASET_BYTES", str(50 * 1024 * 1024))))
MAX_DATASET_ROWS = max(100, int(os.getenv("AGENTIC_MCP_MAX_DATASET_ROWS", "100000")))
_SESSION_RESULTS = {}


def _dataset(dataset: str):
    if not isinstance(dataset, str) or not dataset.strip() or Path(dataset).name != dataset:
        raise ValueError("dataset must be a CSV filename in the configured data directory")
    if not dataset.lower().endswith(".csv"):
        raise ValueError("Only CSV datasets are supported")
    path = (DATA_ROOT / dataset).resolve()
    if path.parent != DATA_ROOT or not path.is_file():
        raise ValueError("Dataset was not found in the configured data directory")
    if path.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("Dataset exceeds AGENTIC_MCP_MAX_DATASET_BYTES")
    frame = pd.read_csv(path)
    if len(frame) > MAX_DATASET_ROWS:
        raise ValueError("Dataset exceeds AGENTIC_MCP_MAX_DATASET_ROWS")
    return frame


def _invoke(tool: str, dataset: str, inputs=None):
    frame = _dataset(dataset)
    context = _SESSION_RESULTS.setdefault(dataset, {})
    result = registry.execute(tool, inputs or {}, frame,
                              {"results": context, "latest_model": context.get("train_model")})
    context[tool] = result
    return result


@server.tool()
def list_analytics_tools() -> dict:
    """List available AgenticDataLab analytics tools."""
    return {"tools": registry.list_tools()}


@server.tool()
def profile_dataset(dataset: str) -> dict:
    """Profile dataset shape, types, and missingness."""
    return _invoke("profile_dataset", dataset)


@server.tool()
def analyze_data_quality(dataset: str, target_col: str = "") -> dict:
    """Analyze dataset quality; optionally supply a target column."""
    return _invoke("analyze_data_quality", dataset, {"target_col": target_col or None})


@server.tool()
def run_eda(dataset: str) -> dict:
    """Run descriptive statistics, correlations, and outlier analysis."""
    return _invoke("run_eda", dataset)


@server.tool()
def train_model(dataset: str, target_col: str, task_type: str = "", cv: int = 3,
                tune: bool = False, n_iter: int = 5) -> dict:
    """Train and compare bounded candidate models for a target column."""
    if not target_col:
        raise ValueError("target_col is required")
    if task_type and task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    if not 2 <= cv <= 10 or not 1 <= n_iter <= 10:
        raise ValueError("cv must be 2–10 and n_iter must be 1–10")
    return _invoke("train_model", dataset, {"target_col": target_col, "task_type": task_type or None,
                                              "cv": cv, "tune": tune, "n_iter": n_iter})


@server.tool()
def evaluate_model(dataset: str) -> dict:
    """Evaluate the model trained in this MCP session."""
    frame = _dataset(dataset)
    context = _SESSION_RESULTS.get(dataset, {})
    return registry.execute("evaluate_model", {}, frame,
                            {"results": context, "latest_model": context.get("train_model")})


@server.tool()
def compare_models(dataset: str, target_col: str, task_type: str = "", cv: int = 3) -> dict:
    """Compare supported candidate models with deterministic cross-validation."""
    if not target_col or not 2 <= cv <= 10:
        raise ValueError("target_col is required and cv must be 2–10")
    if task_type and task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    result = _invoke("train_model", dataset, {"target_col": target_col, "task_type": task_type or None,
                                                "cv": cv, "tune": False})
    return {"status": result.get("status"), "model_comparison": result.get("model_comparison", result.get("models", [])),
            "selected_model": result.get("model_name"), "cv_results": result.get("cv_results"),
            "experiment_id": result.get("experiment_id"), "error": result.get("error")}


@server.tool()
def explain_model(dataset: str, target_col: str = "", method: str = "auto", top_n: int = 20) -> dict:
    """Explain the latest model trained for this dataset in the current MCP process."""
    if not 1 <= top_n <= 100:
        raise ValueError("top_n must be 1–100")
    if method not in {"auto", "model_native", "permutation", "shap"}:
        raise ValueError("method must be auto, model_native, permutation, or shap")
    frame = _dataset(dataset)
    context = _SESSION_RESULTS.get(dataset, {})
    if not context.get("train_model"):
        return {"status": "error", "error": "Train a model for this dataset in this MCP process before requesting an explanation."}
    return registry.execute("explain_global", {"method": method, "top_n": top_n}, frame,
                            {"results": context, "latest_model": context["train_model"]})


@server.tool()
def generate_report(dataset: str) -> dict:
    """Generate a report from deterministic profiling and EDA results."""
    frame = _dataset(dataset)
    profile = registry.execute("profile_dataset", {}, frame, {})
    eda = registry.execute("run_eda", {}, frame, {})
    from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent
    return NotebookSynthesizerAgent().run(profile, eda, None, None)


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
