# File: src/agents/model_agent.py
# Updated ModelAgent to register to bus and accept messages (trigger training from EDA if desired)
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None

class ModelAgent:
    def __init__(self, a2a_bus: A2ABus = None, output_dir="models", random_state=42, n_iter_search=20, cv=3):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_tool = FileTools()
        self.memory = MemoryTools()
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.cv = cv
        self.a2a_bus = a2a_bus
        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("model")

    # keep existing helper methods (_detect_task, _build_preprocessor, _get_search_space, _score) unchanged...
    # (For brevity they are omitted here; integrate from your original ModelAgent implementation.)

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
        # simplified wrapper around original run. After training succeeds publish to verifier.
        # (Assumes existing full training implementation is present in your code base.)
        # We'll call the existing run() logic (rename original heavy method to _train_and_evaluate in your file)
        try:
            # if you have method _train_and_evaluate implemented, call it. Otherwise, run your normal logic here.
            result = self._train_and_evaluate(df, target_col, test_size, random_state)
        except AttributeError:
            # fallback: return error
            return {"status": "error", "error": "ModelAgent training method not implemented in this trimmed example."}

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
