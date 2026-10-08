import streamlit as st
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from agents.profiler_agent import ProfilerAgent
from core.a2a_bus import A2ABus

st.set_page_config(page_title="Profiler Agent", layout="wide")
st.title("🧪 Profiler Agent")

# Load dataset
if "uploaded_df" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]

# Load global bus
a2a = A2ABus()

if st.button("Run Profiler"):
    profiler = ProfilerAgent(a2a_bus=a2a)
    result = profiler.run(df)

    if result["status"] == "error":
        st.error(result["error"])
    else:
        st.success("Profiler completed!")
        st.session_state["profiler_output"] = result
        st.json(result)
