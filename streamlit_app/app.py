import streamlit as st
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from tools.file_tools import FileTools
from tools.dataset_tools import DatasetTools
from tools.memory_tools import MemoryTools


st.title("🤖 Multi-Agent Data Analyst")


# --------------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------------
st.header("📤 Upload Dataset")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["uploaded_df"] = df
    st.success("File uploaded successfully!")
    st.dataframe(df)

    # Save to memory using MCP
    MemoryTools().save("file_uploaded", {"rows": len(df), "cols": list(df.columns)})


# --------------------------------------------------------
# MCP TOOL BUTTONS
# --------------------------------------------------------
st.header("🛠 MCP Tools Panel")

col1, col2, col3 = st.columns(3)

# ------------------- FILE TOOLS -------------------
with col1:
    st.subheader("📁 FileTools")

    if st.button("List files in current folder"):
        ft = FileTools()
        st.write(ft.list("."))

    if st.button("Write test.txt"):
        ft = FileTools()
        st.write(ft.write("test.txt", "Hello from MCP!"))

    if st.button("Read test.txt"):
        ft = FileTools()
        st.write(ft.read("test.txt"))


# ------------------- DATASET TOOLS -------------------
with col2:
    st.subheader("📊 DatasetTools")

    dt = DatasetTools()

    if st.button("Show dataset shape"):
        df = st.session_state.get("uploaded_df")
        if df is not None:
            st.write(dt.get_shape(df))

    if st.button("Show dataset columns"):
        df = st.session_state.get("uploaded_df")
        if df is not None:
            st.json(dt.get_columns(df))


# ------------------- MEMORY TOOLS -------------------
with col3:
    st.subheader("🧠 MemoryTools")

    mem = MemoryTools()

    if st.button("Show stored memory"):
        st.json(mem.load_all())
