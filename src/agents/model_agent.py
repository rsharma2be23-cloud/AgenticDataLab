# File: src/agents/model_agent.py
# Updated ModelAgent to register to bus and accept messages (trigger training from EDA if desired)
import pandas as pd
from tools.model_tools import ModelTools
from tools.memory_tools import MemoryTools

try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None

class ModelAgent:
    """Choose a training strategy; ModelTools owns the actual computation."""
    def __init__(self, a2a_bus: A2ABus = None, output_dir="models", random_state=42, n_iter_search=20, cv=3, model_tools=None):
        self.output_dir = output_dir
        self.memory = MemoryTools()
        self.model_tools = model_tools or ModelTools(output_dir=output_dir)
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.cv = cv
        self.a2a_bus = a2a_bus
        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("model")

    def choose_strategy(self, df, target_col):
        """Select a controlled model configuration from target balance and type."""
        y = df[target_col]
        if self.model_tools._detect_task(y) == "classification":
            counts = y.value_counts(normalize=True, dropna=True)
            skewed = bool(len(counts) > 1 and counts.iloc[0] >= 0.7)
            return {"model_name": None, "class_weight": "balanced" if skewed else None,
                    "reason": "Balance class weights for a strongly imbalanced target." if skewed else "Compare supported classifiers."}
        return {"model_name": None, "class_weight": None, "reason": "Compare supported regression models."}

    def poll_messages_and_run(self, df: pd.DataFrame):
        """
        Check for A2A messages sent to the model agent
        and auto-run training if EDA is completed.
        """
        if not self.a2a_bus:
            return None

        messages = self.a2a_bus.fetch("model", consume=True)
        trained = None

        for msg in messages:
            if msg.get("topic") == "eda.completed":
                # UI will still supply target column.
                self.memory.save("eda_output", msg.get("payload", {}))
                trained = True

        return trained

    def run(self, df: pd.DataFrame, target_col: str = None, test_size = 0.2, random_state = None ):
        try:
            if target_col not in df.columns:
                return {"status": "error", "error": f"Target column '{target_col}' was not found."}
            strategy = self.choose_strategy(df, target_col)
            result = self.model_tools.train(df, target_col, model_name=strategy["model_name"],
                                            class_weight=strategy["class_weight"])
            result["agent_decision"] = strategy
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

        # Save result to memory
        self.memory.save("model_output", result)

        # publish to verifier
        if self.a2a_bus:
            self.a2a_bus.publish(
                from_agent="model",
                to="verifier",
                topic="model.trained",
                payload={"model_output": result}
            )

        return result
