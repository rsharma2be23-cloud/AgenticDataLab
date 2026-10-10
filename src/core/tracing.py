"""Small JSONL execution trace with secret redaction and correlation fields."""
import json
import os
import re
import threading
from datetime import datetime, timezone

_SECRET = re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)")


def redact(value):
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()
                if str(key).lower() not in {"prompt", "dataset", "data", "api_key", "token", "password", "secret"}}
    if isinstance(value, list):
        return [redact(item) for item in value[:1000]]
    if isinstance(value, str):
        return _SECRET.sub(r"\1\2[REDACTED]", value[:2000])
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:1000]


class ExecutionTracer:
    def __init__(self, path=None, enabled=True):
        self.path = os.path.abspath(path or os.getenv("AGENTIC_TRACE_PATH", "project_storage/execution_traces.jsonl"))
        self.enabled = enabled
        self._lock = threading.Lock()

    def record(self, event):
        if not self.enabled:
            return None
        item = redact(event)
        item["recorded_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
        return item

    def list(self, task_id=None):
        try:
            with open(self.path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []
        return [row for row in rows if task_id is None or row.get("task_id") == task_id]
