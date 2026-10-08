import os
import sys
import streamlit as st

# allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from core.a2a_bus import A2ABus
from agents.verifier_agent import VerifierAgent

a2a = A2ABus()
agent = VerifierAgent(a2a_bus=a2a)

# Listen for model.trained
auto_res = agent.poll_messages_and_run()
if auto_res:
    st.success("Auto-verified model from A2A!")


from agents.verifier_agent import VerifierAgent
from tools.memory_tools import MemoryTools

st.set_page_config(page_title="Verifier", layout="wide")
st.title("🔎 Verifier (Agent)")

# Requires model output to exist
memory = MemoryTools()

# Check session or memory for model_output
model_output = st.session_state.get("model_output", None)
if model_output is None:
    # attempt to load from memory history
    try:
        mem = memory.load()
        if mem.get("status") == "success":
            hist = mem.get("memory", {}).get("history", [])
            for entry in reversed(hist):
                if entry.get("key") == "model_output":
                    model_output = entry.get("value")
                    break
    except Exception:
        model_output = None

if model_output is None:
    st.warning("No trained model found. Please run the AutoML (Train model) page first.")
    st.stop()

st.write("Model found — you can run the Verifier agent to evaluate model reliability and get advice.")

if st.button("✅ Run Verifier Agent"):
    st.info("Running VerifierAgent…")
    try:
        verifier = VerifierAgent()
        result = verifier.run(model_output)

        if result.get("status") != "success":
            st.error("Verifier failed: " + str(result.get("error", "unknown error")))
        else:
            memory.save("verifier_output", result)
            st.session_state["verifier_output"] = result

            st.success("Verifier completed and saved to memory.")
            st.subheader("Verdict")
            st.write(result.get("quality", "Unknown"))

            st.subheader("Metrics / Notes")
            st.json(result.get("metrics", result.get("notes", {})))
    except Exception as e:
        st.exception(e)
else:
    st.info("Click the button to run the Verifier agent.")
    if "verifier_output" in st.session_state:
        st.json(st.session_state["verifier_output"])
