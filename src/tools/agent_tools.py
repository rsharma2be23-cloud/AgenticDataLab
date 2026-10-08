# src/tools/registry.py

from src.tools.file_tools import FileTools
from src.tools.dataset_tools import DatasetTools
from src.tools.model_tools import ModelTools
from src.tools.notebook_tools import NotebookTools
from src.tools.job_tools import JobTools
from src.tools.memory_tools import MemoryTools


class ToolRegistry:
    """Central registry for all internal MCP-style tools."""

    def __init__(self):
        self.tools = {
            "file": FileTools(),
            "dataset": DatasetTools(),
            "model": ModelTools(),
            "notebook": NotebookTools(),
            "job": JobTools(),
            "memory": MemoryTools(),
        }

    def get_tool(self, name: str):
        """Get a tool instance by name."""
        return self.tools.get(name, None)

    def list_tools(self):
        """List all tool names."""
        return list(self.tools.keys())
