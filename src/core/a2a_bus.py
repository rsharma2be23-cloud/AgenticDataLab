# File: src/core/a2a_bus.py
import threading
import time
import json
import os
from typing import Dict, Any, List, Optional

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
                data = self.memory.load().get("memory", {})
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
            # load current memory, update the storate key, write back
            mem = self.memory.load().get("memory", {})
            mem[self.STORAGE_KEY] = self._messages
            # memory.save appends a new entry; here we want to write a stable snapshot
            # so we call lower-level write to the memory file if available
            # fallback: use memory.save as snapshot entry
            self.memory.save("a2a_snapshot", self._messages)
        except Exception:
            # Don't fail on persistence
            pass

    def publish(self, from_agent: str, to: str, topic: str, payload: Any, meta: Optional[Dict] = None):
        """
        Publish a message from one agent to another (to can be one agent name or 'broadcast' or comma-separated list).
        Message structure: {from, to, topic, payload, meta, timestamp}
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = {
            "from": from_agent,
            "to": to,
            "topic": topic,
            "payload": payload,
            "meta": meta or {},
            "timestamp": timestamp,
        }

        recipients = []
        if isinstance(to, str) and to.lower() == "broadcast":
            # broadcast to everyone known so far (keys in _messages)
            with self._lock:
                recipients = list(self._messages.keys())
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
