"""Atomic JSON task persistence and separate reusable analytical memory."""
import json
import os
import re
import threading
from datetime import datetime, timezone


_SECRET_VALUE = re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)")


def json_safe(value, parent_key=""):
    """Convert values into bounded, non-executable JSON primitives."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _SECRET_VALUE.sub(r"\1\2[REDACTED]", value[:10000])
    if isinstance(value, dict):
        return {str(k)[:200]: json_safe(v, str(k)) for k, v in value.items()
                if str(k).lower() not in {"dataframe", "raw_data", "sample_rows", "dataset_content"}
                and not any(secret in str(k).lower() for secret in ("api_key", "token", "password", "secret"))}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, parent_key) for v in value[:10000]]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item(), parent_key)
        except Exception:
            pass
    return str(value)[:2000]


class TaskStateStore:
    """Persist versioned workflow snapshots using atomic replace and safe recovery."""
    def __init__(self, path="project_storage/tasks.json", retention_days=90):
        self.path = os.path.abspath(path)
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _read(self):
        try:
            with open(self.path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("task_id"), str) or not state["task_id"].strip():
            raise ValueError("Task state requires a non-empty task_id")
        snapshot = json_safe(dict(state))
        snapshot["schema_version"] = 1
        snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            all_states = self._read()
            all_states[snapshot["task_id"]] = snapshot
            temp = self.path + ".tmp"
            with open(temp, "w", encoding="utf-8") as stream:
                json.dump(all_states, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        return snapshot

    def get(self, task_id):
        with self._lock:
            return self._read().get(task_id)

    def list(self):
        with self._lock:
            return self._read()

    def cleanup(self, older_than_days=None):
        """Remove old terminal workflow snapshots; active/pending tasks are retained."""
        cutoff = datetime.now(timezone.utc).timestamp() - 86400 * (older_than_days or self.retention_days)
        with self._lock:
            states = self._read()
            kept = {}
            for key, state in states.items():
                terminal = state.get("status") in {"completed", "failed", "rejected"}
                stamp = state.get("updated_at", "")
                try:
                    old = datetime.fromisoformat(stamp).timestamp() < cutoff
                except (TypeError, ValueError):
                    old = False
                if not (terminal and old):
                    kept[key] = state
            if len(kept) != len(states):
                temp = self.path + ".tmp"
                with open(temp, "w", encoding="utf-8") as stream:
                    json.dump(kept, stream, indent=2)
                os.replace(temp, self.path)
            return len(states) - len(kept)


class AnalyticalMemoryStore:
    """Separate compact, opt-in reusable findings indexed by task/dataset metadata."""
    SECRET = re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]")

    def __init__(self, path="project_storage/analytical_memory.json", max_records=1000):
        self.path = os.path.abspath(path)
        self.max_records = max(1, int(max_records))
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as stream:
                data = json.load(stream)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def add(self, record):
        if not isinstance(record, dict):
            raise ValueError("Memory record must be an object")
        safe = json_safe(record)
        text = json.dumps(safe, ensure_ascii=False)
        if self.SECRET.search(text):
            raise ValueError("Memory record appears to contain a secret")
        if len(text) > 20000:
            raise ValueError("Memory record exceeds 20 KB")
        safe.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self._lock:
            records = (self._read() + [safe])[-self.max_records:]
            temp = self.path + ".tmp"
            with open(temp, "w", encoding="utf-8") as stream:
                json.dump(records, stream, indent=2, allow_nan=False)
            os.replace(temp, self.path)
        return safe

    def retrieve(self, query, limit=5):
        """Return records matching task tokens / dataset fingerprint, newest relevant first."""
        terms = {part.lower() for part in re.findall(r"[\w.-]{3,}", str(query))}
        scored = []
        for record in self._read():
            haystack = json.dumps(record, ensure_ascii=False).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, record.get("created_at", ""), record))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [row[2] for row in scored[:max(1, min(int(limit), 20))]]
