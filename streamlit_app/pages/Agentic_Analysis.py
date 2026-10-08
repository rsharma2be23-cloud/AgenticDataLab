import os
import sys
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from core.execution_engine import ExecutionEngine
from core.task_state import TaskStateStore

st.set_page_config(page_title="Agentic Analysis", layout="wide")
st.title("🧭 Agentic Analysis")
st.caption("Describe the analysis goal. The planner selects from controlled dataset tools; each step and result is recorded.")

if "uploaded_df" not in st.session_state:
    st.warning("Upload a CSV file from the Home page first.")
    st.stop()

goal = st.text_area("Analysis goal", "Analyze this dataset and identify the most important factors affecting the target.")
target = st.selectbox("Optional target column", ["Auto-detect"] + list(st.session_state["uploaded_df"].columns))
if st.button("Plan and run", type="primary"):
    engine = ExecutionEngine()
    request = goal
    # Explicit target selection is passed as clear user intent to the planner and tool.
    if target != "Auto-detect":
        request += f" Target: {target}"
    with st.spinner("Planning and executing selected tools…"):
        result = engine.run(st.session_state["uploaded_df"], request)
    st.session_state["agentic_task"] = result

result = st.session_state.get("agentic_task")
if result:
    state = result["state"]
    st.subheader(f"Task status: {result['status']}")
    st.markdown("**Plan**")
    st.json(state.get("plan"))
    st.markdown("**Critic review**")
    st.json(state.get("critic_result"))
    st.markdown("**Tool results**")
    st.json(result.get("results", {}))
    with st.expander("Persisted task state"):
        st.json(TaskStateStore().get(result["task_id"]))
