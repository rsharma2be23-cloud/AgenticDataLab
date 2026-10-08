# File: src/core/a2a_bus.py
import threading
import time
import json
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from tools.memory_tools import MemoryTools

class A2ABus:
    """
    Simple in-memory Agent-to-Agent message bus with optional persistence via MemoryTools.
    Agents can publish messages to targets (agent names or 'broadcast') and each agent can
    fetch pending messages intended for it.
    """

    STORAGE_KEY = "a2a_messages"

    def __init__(self, memory: Optional[MemoryTools] = None, persist: bool = True):
        self._lock = threading.Lock()
        self._persist = persist
        self.memory = memory or MemoryTools()
        # internal messages dict: { recipient_agent: [msg, ...], ... }
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        # load persisted messages if any
        if self._persist:
            try:
                data = self.memory.load_all() if hasattr(self.memory, "load_all") else {}
                msgs = data.get(self.STORAGE_KEY, {})
                if isinstance(msgs, dict):
                    self._messages = msgs
            except Exception:
                # if load fails, just start empty
                self._messages = {}

    def _persist_messages(self):
        if not self._persist:
            return
        try:
            self.memory.save(self.STORAGE_KEY, self._messages)
        except Exception:
            # Don't fail on persistence
            pass

    def publish(self, from_agent: str, to: str, topic: str, payload: Any, meta: Optional[Dict] = None, task_id: Optional[str] = None):
        """
        Publish a message from one agent to another (to can be one agent name or 'broadcast' or comma-separated list).
        Message structure: {from, to, topic, payload, meta, timestamp}
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        msg = {
            "id": str(uuid.uuid4()),
            "task_id": task_id or (meta or {}).get("task_id"),
            "from": from_agent,
            "sender": from_agent,
            "to": to,
            "receiver": to,
            "topic": topic,
            "type": topic,
            "payload": payload,
            "meta": meta or {},
            "timestamp": timestamp,
            "status": "queued",
        }

        recipients = []
        if isinstance(to, str) and to.lower() == "broadcast":
            # broadcast to everyone known so far (keys in _messages)
            with self._lock:
                recipients = [name for name in self._messages if not name.startswith("_")]
        elif isinstance(to, str) and "," in to:
            recipients = [r.strip() for r in to.split(",") if r.strip()]
        else:
            recipients = [to]

        with self._lock:
            for r in recipients:
                if r not in self._messages:
                    self._messages[r] = []
                self._messages[r].append(msg)
            # also allow a "global" inbox for audit
            if "_audit" not in self._messages:
                self._messages["_audit"] = []
            self._messages["_audit"].append(msg)

        self._persist_messages()
        return {"status": "success", "queued_for": recipients, "message": msg}

    def fetch(self, agent_name: str, consume: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch pending messages for agent_name. By default, consume them (they are removed).
        """
        with self._lock:
            msgs = self._messages.get(agent_name, []).copy()
            if consume:
                self._messages[agent_name] = []
        # persist snapshot
        if consume:
            self._persist_messages()
        return msgs

    def peek(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Return messages for an agent without consuming them.
        """
        with self._lock:
            return self._messages.get(agent_name, []).copy()

    def register_agent(self, agent_name: str):
        """
        Ensure an inbox exists for an agent.
        """
        with self._lock:
            if agent_name not in self._messages:
                self._messages[agent_name] = []
        self._persist_messages()

    def list_agents(self) -> List[str]:
        with self._lock:
            return [k for k in self._messages.keys() if not k.startswith("_")]

    def audit_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._messages.get("_audit", []).copy()

    # Compatibility with the former tools.a2a_tools bus API.
    def send(self, sender, receiver, topic, payload, task_id=None):
        return self.publish(sender, receiver, topic, payload, task_id=task_id)

    def direct_message(self, sender, receiver, topic, payload):
        return self.send(sender, receiver, topic, payload)

    def get_inbox(self, agent):
        return self.peek(agent)

    def get_audit_log(self):
        return self.audit_log()
