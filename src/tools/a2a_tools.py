"""Backward-compatible import for the canonical A2A bus implementation."""
from core.a2a_bus import A2ABus


# Keep the dashboard's historic singleton API while using the same bus class.
A2A_GLOBAL_BUS = A2ABus()

__all__ = ["A2ABus", "A2A_GLOBAL_BUS"]
