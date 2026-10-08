import os
import json
from datetime import datetime
import numpy as np

class MemoryAgent:
    def __init__(self, memory_file="memory/project_memory.json"):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w") as f:
                json.dump({"history": []}, f, indent=4)

    def _make_json_safe(self, value):
        try:
            # Convert pandas/numpy objects
            if isinstance(value, float) and np.isnan(value):
                return None

            if isinstance(value, dict):
                return {k: self._make_json_safe(v) for k, v in value.items()}

            if isinstance(value, list):
                return [self._make_json_safe(v) for v in value]

            json.dumps(value)
            return value

        except:
            return str(value)

    def load(self):
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
            return {"status": "success", "memory": data}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def save(self, key, value):
        safe_value = self._make_json_safe(value)

        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)

            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "key": key,
                "value": safe_value
            }

            data["history"].append(entry)

            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=4)

            return {"status": "success", "saved": entry}

        except Exception as e:
            return {"status": "error", "error": str(e)}
