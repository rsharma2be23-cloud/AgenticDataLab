"""Chooses an explanation method; ExplainabilityTools computes all values."""
from tools.explainability_tools import ExplainabilityTools


class ExplainabilityAgent:
    def __init__(self, tools=None):
        self.tools = tools or ExplainabilityTools()

    def explain_global(self, artifact_path, frame, method="auto", target=None, top_n=30):
        return self.tools.global_importance(artifact_path, frame, method=method, target=target, top_n=top_n)

    def explain_prediction(self, artifact_path, instance, reference_data, method="auto", top_n=10):
        return self.tools.local_explanation(artifact_path, instance, reference_data, method=method, top_n=top_n)
