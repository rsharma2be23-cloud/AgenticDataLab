from core.agentic_schemas import CriticResult


class CriticAgent:
    """Basic task-quality review; model thresholds are intentionally modest in Phase 1."""
    def review(self, task_result, metrics=None, errors=None, tool_output=None):
        errors = errors or []
        if errors or not task_result or task_result.get("status") == "error":
            return CriticResult("RETRY", "A tool failed before the requested analysis completed.",
                                ["Retry with a narrower analysis", "Check tool inputs"])
        metrics = metrics or (task_result.get("metrics") if isinstance(task_result, dict) else {}) or {}
        output = tool_output or task_result or {}
        if isinstance(output, dict) and output.get("status") == "error":
            return CriticResult("RETRY", output.get("error", "Tool execution failed"), ["Retry with valid inputs"])
        if "f1_score" in metrics and metrics["f1_score"] < 0.5:
            return CriticResult("RETRY", "Model F1 score is below the basic review threshold.",
                                ["Try logistic regression with balanced class weights", "Try another model"])
        if "r2" in metrics and metrics["r2"] < 0:
            return CriticResult("RETRY", "Model R2 is below a constant baseline.", ["Try another model"])
        if isinstance(output, dict) and output.get("status") == "success":
            return CriticResult("ACCEPT", "Requested tool work completed and passed basic checks.", [])
        return CriticResult("ACCEPT", "No execution errors were reported.", [])
