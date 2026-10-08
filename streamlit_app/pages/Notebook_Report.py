import os
import sys
import streamlit as st

# allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent
from tools.memory_tools import MemoryTools
from tools.file_tools import FileTools

st.set_page_config(page_title="Notebook Report", layout="wide")
st.title("📘 Auto-Generated Notebook Report")

memory = MemoryTools()
file_tool = FileTools()

from core.a2a_bus import A2ABus
from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent

a2a = A2ABus()
agent = NotebookSynthesizerAgent(a2a_bus=a2a)

auto = agent.poll_messages_and_run()
if auto:
    st.success("📘 Auto-generated notebook via A2A!")


# Helper to retrieve outputs from session or memory
def get_from_session_or_memory(key):
    if key in st.session_state:
        return st.session_state[key]
    # else search memory history for latest entry
    try:
        mem = memory.load()
        if mem.get("status") == "success":
            hist = mem.get("memory", {}).get("history", [])
            for entry in reversed(hist):
                if entry.get("key") == key:
                    return entry.get("value")
    except Exception:
        pass
    return None

profiler_output = get_from_session_or_memory("profiler_output")
eda_output = get_from_session_or_memory("eda_output")
model_output = get_from_session_or_memory("model_output")
verifier_output = get_from_session_or_memory("verifier_output")

# Show which steps are present
st.markdown("### ⚠️ These steps (outputs) must exist before generating the notebook:")
cols = st.columns(4)
cols[0].write("**Profiler**")
cols[1].write("**EDA**")
cols[2].write("**Model**")
cols[3].write("**Verifier**")

def tick(x):
    return "✅" if x is not None else "❌"

st.write(f"{tick(profiler_output)} Profiler output")
st.write(f"{tick(eda_output)} EDA output")
st.write(f"{tick(model_output)} Model output")
st.write(f"{tick(verifier_output)} Verifier output")

if not (profiler_output and eda_output and model_output and verifier_output):
    st.error("Run Profiler → EDA → AutoML → Verifier first. The Notebook synthesizer needs all four outputs.")
    st.info("Use the pages: Profiler, EDA Dashboard, AutoML, Verifier (in this order).")
    st.stop()

st.success("All required outputs found. You can now generate the notebook.")

if st.button("🧾 Generate Notebook"):
    st.info("Generating notebook (this will call NotebookSynthesizerAgent)...")
    try:
        nb_agent = NotebookSynthesizerAgent()
        res = nb_agent.run(profiler_output, eda_output, model_output, verifier_output)

        if res.get("status") != "success":
            st.error("Notebook generation failed: " + str(res.get("error", "unknown")))
            st.stop()

        nb_path = res.get("notebook_path", "reports/auto_report.ipynb")

        # If file exists, offer download
        if os.path.exists(nb_path):
            st.success(f"Notebook saved at: `{nb_path}`")
            # read bytes and provide download button
            try:
                with open(nb_path, "rb") as f:
                    nb_bytes = f.read()
                st.download_button(label="⬇️ Download notebook (.ipynb)", data=nb_bytes, file_name=os.path.basename(nb_path), mime="application/x-ipynb+json")
            except Exception:
                st.warning("Notebook generated but failed to create download button. Check file at: " + nb_path)
        else:
            st.warning("Notebook claimed saved but file not found at: " + nb_path)

    except Exception as e:
        st.exception(e)

else:
    st.info("Click 'Generate Notebook' to synthesize a notebook from the stored agent outputs.")
