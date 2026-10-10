"""Validated local agent message bus with correlated delivery and bounded dispatch."""
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone


class A2ABus:
    STORAGE_KEY = "a2a_messages"
    STATUSES = {"queued", "processing", "succeeded", "failed"}
    SECRET_TEXT = re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]")

    def __init__(self, memory=None, persist=True, timeout_seconds=10, max_retries=2):
        self._lock = threading.RLock()
        self._persist = persist
        if persist and memory is None:
            from tools.memory_tools import MemoryTools
            memory = MemoryTools()
        self.memory = memory
        self.timeout_seconds = max(0.05, float(timeout_seconds))
        self.max_retries = max(0, min(10, int(max_retries)))
        self._messages = {}
        self._processed = set()
        if persist and memory:
            try:
                stored = memory.load_all().get(self.STORAGE_KEY, {})
                if isinstance(stored, dict):
                    self._messages = stored
                    self._processed = {m["id"] for ms in stored.values() for m in ms
                                       if isinstance(m, dict) and m.get("status") == "succeeded" and m.get("id")}
            except Exception:
                self._messages = {}

    @staticmethod
    def _validate(sender, receiver, message_type, payload):
        for label, value in (("sender", sender), ("receiver", receiver), ("type", message_type)):
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError(f"A2A {label} must be a non-empty string of at most 200 characters")
        if not isinstance(payload, (dict, list, str, int, float, bool, type(None))):
            raise ValueError("A2A payload must be JSON-compatible")
        try:
            encoded = json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("A2A payload must be finite JSON data") from exc
        if len(encoded) > 1_000_000:
            raise ValueError("A2A payload exceeds 1 MB")
        if A2ABus.SECRET_TEXT.search(encoded) or A2ABus._secret_key(payload):
            raise ValueError("A2A payload must not contain secrets")
        if A2ABus._contains_raw_dataset(payload):
            raise ValueError("A2A payload must not contain raw dataset rows")

    @staticmethod
    def _secret_key(value):
        if isinstance(value, dict):
            return any(any(term in str(key).lower() for term in ("api_key", "token", "password", "secret"))
                       or A2ABus._secret_key(item) for key, item in value.items())
        if isinstance(value, list):
            return any(A2ABus._secret_key(item) for item in value)
        return False

    @staticmethod
    def _contains_raw_dataset(value):
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key).lower()
                if name in {"sample_rows", "dataset_content", "raw_data", "dataframe"}:
                    return True
                if name in {"rows", "records"} and isinstance(item, list):
                    return True
                if A2ABus._contains_raw_dataset(item):
                    return True
        if isinstance(value, list):
            return any(A2ABus._contains_raw_dataset(item) for item in value)
        return False

    def publish(self, from_agent, to, topic, payload, meta=None, task_id=None, correlation_id=None):
        self._validate(from_agent, to, topic, payload)
        meta = meta or {}
        if not isinstance(meta, dict):
            raise ValueError("A2A metadata must be an object")
        self._validate(from_agent, to, topic, meta)
        now = datetime.now(timezone.utc).isoformat()
        recipients = ([name for name in self.list_agents()] if to.lower() == "broadcast" else
                      [part.strip() for part in to.split(",") if part.strip()])
        msg = {"id": str(uuid.uuid4()), "message_id": None, "task_id": task_id or meta.get("task_id"),
               "correlation_id": correlation_id or meta.get("correlation_id") or task_id or meta.get("task_id"),
               "sender": from_agent, "receiver": to, "type": topic, "payload": payload,
               "status": "queued", "created_at": now, "timestamp": now, "error": None,
               "retry_count": 0, "meta": meta}
        msg["message_id"] = msg["id"]
        with self._lock:
            for receiver in recipients:
                self._messages.setdefault(receiver, []).append(msg)
            self._messages.setdefault("_audit", []).append(msg)
        self._persist_messages()
        return {"status": "success", "queued_for": recipients, "message": dict(msg)}

    def _persist_messages(self):
        if self._persist and self.memory:
            saved = self.memory.save(self.STORAGE_KEY, self._messages)
            if isinstance(saved, dict) and saved.get("status") == "error":
                raise OSError("Could not persist A2A messages")

    def fetch(self, agent_name, consume=True):
        with self._lock:
            messages = list(self._messages.get(agent_name, []))
            if consume:
                self._messages[agent_name] = []
        if consume:
            self._persist_messages()
        return messages

    def peek(self, agent_name):
        with self._lock:
            return list(self._messages.get(agent_name, []))

    def register_agent(self, agent_name):
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Agent name is required")
        with self._lock:
            self._messages.setdefault(agent_name, [])

    def list_agents(self):
        with self._lock:
            return [name for name in self._messages if not name.startswith("_")]

    def audit_log(self):
        with self._lock:
            return list(self._messages.get("_audit", []))

    def task_status(self, task_id):
        messages = [message for message in self.audit_log() if message.get("task_id") == task_id]
        if not messages:
            return {"task_id": task_id, "status": "unknown", "messages": []}
        statuses = [message.get("status") for message in messages]
        status = "failed" if "failed" in statuses else "queued" if "queued" in statuses else "succeeded"
        return {"task_id": task_id, "status": status, "messages": statuses}

    def dispatch(self, message, handler, timeout_seconds=None, max_retries=None):
        """Run one message handler with timeout/retry limits and explicit terminal state."""
        if not isinstance(message, dict) or not isinstance(message.get("id"), str):
            raise ValueError("Dispatch requires a validated message with an id")
        self._validate(message.get("sender"), message.get("receiver"), message.get("type"), message.get("payload"))
        with self._lock:
            stored = next((item for item in self._messages.get("_audit", []) if item.get("id") == message["id"]), None)
            if stored is not None:
                message = stored
        if message["id"] in self._processed:
            return {"status": "succeeded", "duplicate": True, "message_id": message["id"]}
        timeout = max(0.05, float(timeout_seconds or self.timeout_seconds))
        retries = self.max_retries if max_retries is None else max(0, min(10, int(max_retries)))
        with self._lock:
            message["status"] = "processing"
        error = None
        for attempt in range(retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(handler, message["payload"])
            try:
                result = future.result(timeout=timeout)
                if isinstance(result, dict) and result.get("status") == "error":
                    raise RuntimeError(result.get("error", "Agent handler failed"))
                message.update(status="succeeded", retry_count=attempt, error=None)
                self._processed.add(message["id"])
                self._persist_messages()
                return {"status": "succeeded", "message_id": message["id"], "result": result, "retry_count": attempt}
            except FutureTimeout:
                future.cancel()
                error = f"Agent handler timed out after {timeout:g}s"
            except Exception as exc:
                error = str(exc)[:1000]
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            message["retry_count"] = attempt + 1
        message.update(status="failed", error=error)
        self._persist_messages()
        return {"status": "failed", "message_id": message["id"], "error": error, "retry_count": retries}

    # Compatibility with existing dashboard/tool callers.
    def send(self, sender, receiver, topic, payload, task_id=None):
        return self.publish(sender, receiver, topic, payload, task_id=task_id)

    def direct_message(self, sender, receiver, topic, payload):
        return self.send(sender, receiver, topic, payload)

    get_inbox = peek
    get_audit_log = audit_log
