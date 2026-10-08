import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from tools.dataset_tools import DatasetTools
from tools.memory_tools import MemoryTools

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("Dataset Explorer")
st.write("Beautiful & interactive dashboard powered by Multi-Agent System")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if "uploaded_df" not in st.session_state:
    st.warning("Please upload a CSV file first.")
    st.stop()

df = st.session_state["uploaded_df"]
dataset_tools = DatasetTools()
memory = MemoryTools()

# ---------------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Numeric Columns", df.select_dtypes(include="number").shape[1])

with col4:
    st.metric("Categorical Columns", df.select_dtypes(include="object").shape[1])

st.markdown("---")

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Dataset Summary")
summary = dataset_tools.summary(df)
st.json(summary)
memory.save("dataset_summary", summary)

# ---------------------------------------------------------
# SEARCHABLE / SCROLLABLE DATAFRAME
# ---------------------------------------------------------
st.subheader("📄 Data Preview")

search = st.text_input("🔎 Search rows (case-insensitive)")

filtered_df = (
    df[df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
    if search
    else df
)

st.dataframe(filtered_df, use_container_width=True, height=400)

st.markdown("---")

# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------
num_df = df.select_dtypes(include="number")

if num_df.shape[1] > 1:
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="Blues")
    st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------
st.subheader("🚨 Missing Data Overview")

missing = df.isnull().sum()
fig, ax = plt.subplots(figsize=(8, 4))
missing.plot(kind="bar", color="salmon", ax=ax)
ax.set_title("Missing Values per Column")
st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------
# COLUMN TYPE PANELS
# ---------------------------------------------------------
st.subheader("🔠 Column Types")

colA, colB = st.columns(2)

with colA:
    st.write("### 🔢 Numeric Columns")
    st.write(num_df.columns.tolist())

with colB:
    st.write("### 🔤 Categorical Columns")
    st.write(df.select_dtypes(include="object").columns.tolist())

st.markdown("---")

# ---------------------------------------------------------
# NEW FEATURE: COLUMN PROFILE
# ---------------------------------------------------------
st.subheader("🔬 Column Profile")

selected_col = st.selectbox("Select a column", df.columns)

profile = dataset_tools.profile_column(df, selected_col)
st.json(profile)

memory.save("column_profile", profile)

st.markdown("---")

# ---------------------------------------------------------
# NEW FEATURE: DUPLICATE ANALYSIS
# ---------------------------------------------------------
st.subheader("🧩 Duplicate Analysis")

dupes = dataset_tools.duplicate_analysis(df)

c1, c2 = st.columns(2)
with c1:
    st.metric("Total Duplicates", dupes["total_duplicates"])

with c2:
    st.write("### Duplicate Count per Column")
    st.json(dupes["duplicates_per_column"])

st.write("### Sample Duplicate Rows (first 20)")
st.dataframe(pd.DataFrame(dupes["sample_duplicate_rows"]))

memory.save("duplicate_analysis", dupes)

st.markdown("---")

# ---------------------------------------------------------
# NEW FEATURE: AUTO-DTYPE SUGGESTIONS
# ---------------------------------------------------------
st.subheader("🧠 Auto Dtype Suggestions (AI-powered)")

dtype_suggestions = dataset_tools.auto_dtype_suggestions(df)
st.json(dtype_suggestions)

memory.save("dtype_suggestions", dtype_suggestions)

st.success("✨ Dataset Explorer upgraded successfully with new tools!")
