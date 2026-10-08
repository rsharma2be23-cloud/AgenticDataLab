"""Minimal JSON persistence for agentic task runs."""
import json
import os
import threading
from datetime import datetime, timezone


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if hasattr(value, "item"):
            return json_safe(value.item())
        return str(value)


class TaskStateStore:
    def __init__(self, path="project_storage/tasks.json"):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({}, stream)

    def save(self, state):
        with self._lock:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                with open(self.path, "r", encoding="utf-8") as stream:
                    all_states = json.load(stream)
            except (json.JSONDecodeError, OSError):
                all_states = {}
            all_states[state["task_id"]] = json_safe(state)
            temporary = self.path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(all_states, stream, indent=2, default=str)
            os.replace(temporary, self.path)
        return state

    def get(self, task_id):
        with self._lock, open(self.path, "r", encoding="utf-8") as stream:
            return json.load(stream).get(task_id)

    def list(self):
        with self._lock, open(self.path, "r", encoding="utf-8") as stream:
            return json.load(stream)
