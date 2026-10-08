# File: src/agents/notebook_synthesizer_agent.py

import json
from tools.notebook_tools import NotebookTools
from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

try:
    from core.a2a_bus import A2ABus
except:
    A2ABus = None


class NotebookSynthesizerAgent:

    def __init__(self, a2a_bus: A2ABus = None):
        self.notebook_tool = NotebookTools()
        self.file_tool = FileTools()
        self.memory = MemoryTools()        # ✅ FIX ADDED
        self.a2a_bus = a2a_bus

        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("notebook")


    def run(self, profiler_output, eda_output, model_output, verifier_output):

        nb_path = "reports/auto_report.ipynb"

        notebook_content = {
            "profiler": profiler_output,
            "eda": eda_output,
            "model": model_output,
            "verifier": verifier_output
        }

        # Generate notebook
        nb_bytes = self.notebook_tool.create_notebook(notebook_content)

        # Save using FileTools
        self.file_tool.write_binary(nb_path, nb_bytes)

        # Save output in memory
        self.memory.save("notebook_output", {"path": nb_path})

        return {"status": "success", "notebook_path": nb_path}


    def poll_messages_and_run(self):
        """Auto-create notebook when verifier finishes."""
        if not self.a2a_bus:
            return None

        messages = self.a2a_bus.fetch("notebook", consume=True)

        for msg in messages:
            if msg["topic"] == "verifier.completed":
                payload = msg.get("payload", {})
                verifier_output = payload.get("verifier_output")
                profiler = self.memory.load("profiler_output")
                eda = self.memory.load("eda_output")
                model = self.memory.load("model_output")

                return self.run(profiler, eda, model, verifier_output)

        return None
