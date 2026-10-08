# streamlit_app/pages/EDA_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import json

# allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agents.eda_agent import EDAAgent

st.set_page_config(page_title="EDA Dashboard", layout="wide")
st.title("📊 EDA Dashboard")

# Ensure dataset is loaded in session_state by Home page
if "uploaded_df" not in st.session_state:
    st.warning("⚠️ Please upload a CSV file from the Home page first.")
    st.stop()

df = st.session_state["uploaded_df"]

# Run agent or use existing cached eda_output
if "eda_output" not in st.session_state:
    st.session_state["eda_output"] = None

col_left, col_right = st.columns([1, 3])

with col_left:
    if st.button("🤖 Run EDA Agent"):
        st.info("Running EDA agent... this may take a moment.")
        agent = EDAAgent()
        out = agent.run(df)
        if out.get("status") != "success":
            st.error("Agent error: " + str(out.get("error", "unknown")))
            st.stop()
        st.session_state["eda_output"] = out
        st.success("EDA Completed!")
        # scroll to results
        st.rerun()

with col_right:
    st.caption("Level 2: correlation matrices (Pearson / Spearman / Kendall), VIF (if available), skew/kurtosis, histograms and boxplots.")

# show results if available
out = st.session_state.get("eda_output")
if not out:
    st.info("No EDA output found. Click 'Run EDA Agent' to generate EDA.")
    st.stop()

# ----------------------
# Summary cards
# ----------------------
st.markdown("### 📌 Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", out.get("rows", df.shape[0]))
c2.metric("Columns", out.get("cols", df.shape[1]))
c3.metric("Numeric Columns", len(out.get("numeric_columns", [])))
c4.metric("Histograms Generated", len(out.get("histograms", [])))

st.markdown("---")

# ----------------------
# Correlation Controls
# ----------------------
st.markdown("### 🔥 Correlation Matrices")
corrs = out.get("correlations", {})
corr_types_available = [k for k, v in corrs.items() if isinstance(v, dict)]
if not corr_types_available:
    st.info("Not enough numeric columns to compute correlations.")
else:
    typ = st.selectbox("Choose correlation type", options=corr_types_available, index=0)
    # show heatmap image if present
    img_path = out.get("corr_image_paths", {}).get(typ)
    if img_path and os.path.exists(img_path):
        st.image(img_path, caption=f"{typ.title()} correlation heatmap", use_column_width=True)
    else:
        # fallback: build interactive plotly heatmap from matrix dict
        corr_matrix = corrs.get(typ)
        if isinstance(corr_matrix, dict):
            # convert dict back to DataFrame
            try:
                corr_df = pd.DataFrame(corr_matrix)
                fig = px.imshow(corr_df, text_auto=".2f", aspect="auto", title=f"{typ.title()} correlation (interactive)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("Could not render correlation: " + str(e))

# show quick list of top correlated pairs (absolute value)
def top_pairs(corr_df, top_n=8):
    dfm = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    s = dfm.abs().unstack().dropna().sort_values(ascending=False)
    pairs = []
    for (a, b), v in s.items():
        pairs.append((a, b, float(np.round(v, 4))))
        if len(pairs) >= top_n:
            break
    return pairs

import numpy as np
if corr_types_available:
    try:
        corr_df = pd.DataFrame(corrs[typ])
        pairs = top_pairs(corr_df)
        if pairs:
            st.markdown("**Top correlated pairs (abs value):**")
            for a, b, v in pairs:
                st.write(f"- **{a}** ↔ **{b}** : {v}")
    except Exception:
        pass

st.markdown("---")

# ----------------------
# Missing values & outliers & skews
# ----------------------
st.markdown("### 🧩 Missing values & Outliers")

col1, col2 = st.columns(2)
with col1:
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values detected.")
    else:
        st.table(pd.DataFrame(missing[missing > 0], columns=["missing_count"]))

with col2:
    outlier_counts = out.get("outlier_counts", {})
    if outlier_counts:
        st.table(pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["outliers"]))
    else:
        st.info("No outlier information available.")

st.markdown("---")

# ----------------------
# Skewness / Kurtosis / VIF
# ----------------------
st.markdown("### 📈 Skewness, Kurtosis, VIF")
skew_kurt = out.get("skew_kurtosis", {})
if skew_kurt:
    sk_df = pd.DataFrame(skew_kurt).T
    st.dataframe(sk_df, use_container_width=True)
else:
    st.info("No skew/kurtosis info.")

vif = out.get("vif", {})
if isinstance(vif, dict) and vif.get("status") == "success":
    st.markdown("**VIF (multicollinearity)**")
    st.table(pd.DataFrame(vif.get("vif", {}), index=["VIF"]).T)
elif isinstance(vif, dict) and vif.get("status") == "skipped":
    st.info("VIF skipped: " + str(vif.get("reason", "")))
else:
    st.info("VIF not available or failed.")

st.markdown("---")

# ----------------------
# Histograms & Boxplots (thumbnails)
# ----------------------
st.markdown("### 📊 Sample Histograms & Boxplots")
cols = out.get("numeric_columns", [])
if cols:
    choose = st.selectbox("Select numeric column to preview", cols)
    # show histogram and boxplot if available
    hist_path = os.path.join(out.get("histograms", [])[0]) if out.get("histograms") else None
    # better: find file path containing chosen column
    hist_file = next((p for p in out.get("histograms", []) if f"hist_{choose}" in p), None)
    box_file = next((p for p in out.get("boxplots", []) if f"box_{choose}" in p), None)
    c1, c2 = st.columns(2)
    with c1:
        if hist_file and os.path.exists(hist_file):
            st.image(hist_file, caption=f"Histogram: {choose}", use_column_width=True)
        else:
            # fallback to interactive plotly histogram
            try:
                fig = px.histogram(df, x=choose, nbins=30, title=f"Distribution: {choose}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("Could not render histogram: " + str(e))
    with c2:
        if box_file and os.path.exists(box_file):
            st.image(box_file, caption=f"Boxplot: {choose}", use_column_width=True)
        else:
            try:
                fig = px.box(df, y=choose, title=f"Boxplot: {choose}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("Could not render boxplot: " + str(e))
else:
    st.info("No numeric columns to show histograms.")

st.markdown("---")
st.success("EDA panel rendered from agent output.")
