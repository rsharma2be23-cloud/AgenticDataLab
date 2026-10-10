import os
import sys

import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from workflow_service import read_uploaded_dataset

st.set_page_config(page_title="AgenticDataLab", page_icon="📊", layout="wide")
st.title("AgenticDataLab")
st.caption("An auditable data analysis workspace powered by bounded agents and deterministic tools.")

with st.sidebar:
    st.header("Dataset workspace")
    upload = st.file_uploader("Upload a CSV", type=["csv"], help="Maximum 50 MiB and 100,000 rows.")
    if upload is not None:
        try:
            frame = read_uploaded_dataset(upload)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
        else:
            st.session_state["uploaded_df"] = frame
            st.session_state["dataset_name"] = os.path.basename(upload.name)
            st.success(f"Loaded {len(frame):,} rows × {len(frame.columns):,} columns")

frame = st.session_state.get("uploaded_df")
if frame is None:
    st.info("Upload a CSV to begin. The sample `benchmarks/synthetic_classification.csv` is a small generated demo dataset.")
else:
    st.subheader(st.session_state.get("dataset_name", "Current dataset"))
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(frame):,}")
    b.metric("Columns", f"{len(frame.columns):,}")
    c.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    st.write("Column types")
    st.dataframe(frame.dtypes.astype(str).rename("Type").to_frame(), use_container_width=True)
    st.write("Preview")
    st.dataframe(frame.head(20), use_container_width=True)
    st.caption("Use the Agentic Analysis page for profiling, quality checks, EDA, model evaluation, reports, traces, and approval controls.")
