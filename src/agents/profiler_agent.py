# File: src/agents/profiler_agent.py
# Updated ProfilerAgent with A2A publish
import os
import json
import pandas as pd
from typing import Any, Dict

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

# import A2ABus if available (use relative import)
try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None


class ProfilerAgent:
    def __init__(self, a2a_bus: A2ABus = None, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_tool = FileTools()
        self.memory = MemoryTools()
        self.a2a_bus = a2a_bus
        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("profiler")

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            profile = {
                "status": "success",
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
                "sample_rows": df.head(5).to_dict(orient="records"),
            }

            # persist to memory
            self.memory.save("profiler_output", profile)

            # publish message on bus for EDAAgent (A2A)
            if self.a2a_bus:
                self.a2a_bus.publish(
                    from_agent="profiler",
                    to="eda",
                    topic="profiler.completed",
                    payload={"profiler_output": profile}
                )

            return profile
        except Exception as e:
            return {"status": "error", "error": str(e)}
