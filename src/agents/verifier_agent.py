# File: src/agents/verifier_agent.py
# Updated VerifierAgent to register to bus and optionally auto-run when model is trained
import os
from tools.memory_tools import MemoryTools

try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None


class VerifierAgent:
    def __init__(self, a2a_bus: A2ABus = None):
        self.memory = MemoryTools()
        self.a2a_bus = a2a_bus
        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("verifier")

    def run(self, model_result):
        try:
            # simple sanity checks
            if not model_result or model_result.get("status") != "success":
                return {"status": "error", "error": "No trained model provided."}

            metrics = model_result.get("metrics", {})
            # naive quality check
            quality = "Unknown"
            if "r2" in metrics:
                r2 = metrics.get("r2")
                if r2 is None:
                    quality = "Unreliable"
                elif r2 > 0.7:
                    quality = "Good"
                elif r2 > 0.4:
                    quality = "Acceptable"
                else:
                    quality = "Weak"
            elif "f1" in metrics:
                f1 = metrics.get("f1")
                if f1 > 0.75:
                    quality = "Good"
                elif f1 > 0.5:
                    quality = "Acceptable"
                else:
                    quality = "Weak"

            result = {
                "status": "success",
                "quality": quality,
                "metrics": metrics,
                "notes": ["Basic verification completed"]
            }

            # save to memory
            self.memory.save("verifier_output", result)

            # publish audit message
            if self.a2a_bus:
                self.a2a_bus.publish(from_agent="verifier", to="notebook", topic="verifier.completed", payload={"verifier_output": result})

            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def poll_messages_and_run(self):
        """
        If there are messages from 'model' indicating training finished, fetch and run verification.
        """
        if not self.a2a_bus:
            return None
        msgs = self.a2a_bus.fetch("verifier", consume=True)
        ran = None
        for m in msgs:
            if m.get("topic") == "model.trained":
                payload = m.get("payload", {})
                model_out = payload.get("model_output")
                res = self.run(model_out)
                ran = res
        return ran
