import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

import sys

import google.generativeai as genai
import json

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Allow imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from core.a2a_bus import A2ABus
from agents.model_agent import ModelAgent
from agents.explainability_agent import ExplainabilityAgent
from tools.model_tools import ModelTools

# ------------------------------------------------------
# INIT GLOBAL A2A BUS + MODEL AGENT
# ------------------------------------------------------
a2a = A2ABus()
model_agent = ModelAgent(a2a_bus=a2a)

st.set_page_config(page_title="AutoML Dashboard", layout="wide")
st.title("🤖 AutoML Dashboard")

# ------------------------------------------------------
# ENSURE DATA EXISTS
# ------------------------------------------------------
if "uploaded_df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first from the Home page.")
    st.stop()

df = st.session_state["uploaded_df"]

# ------------------------------------------------------
# TARGET COLUMN
# ------------------------------------------------------
st.subheader("🎯 Select Target Column")
columns = df.columns.tolist()
target = st.selectbox("Choose the column to predict:", columns)
tune_candidate = st.checkbox("Tune the leading candidate (bounded randomized search)", value=False)

st.write("### 📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

def gemini_explain_model(result):
    if not os.getenv("GEMINI_API_KEY"):
        return "LLM explanation is unavailable because GEMINI_API_KEY is not configured. Metrics and comparisons shown above are computed by the local model tools."
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        You are an ML expert. Explain this model result simply and clearly.

        MODEL OUTPUT JSON:
        {json.dumps(result, indent=2)}

        Explain:
        1. What task type is this (classification or regression)
        2. How good is the model performance?
        3. Is the accuracy/MSE/F1/R2 good or bad?
        4. Give 3 improvements the user can try
        5. Explain like you're talking to a beginner
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception:
        return "Gemini could not generate an explanation. The computed model results remain available above."


# ======================================================
# 🔥 BUTTON 1 — NORMAL TRAINING (ModelTools)
# ======================================================
if st.button("🚀 Train Model (Direct)", key="manual_train_btn"):
    st.info("⏳ Training model… please wait.")

    model_tool = ModelTools()
    result = model_tool.train(df, target, tune=tune_candidate)

    if result["status"] == "error":
        st.error(f"❌ Error: {result['error']}")
        st.stop()

    st.session_state["model_output"] = result
    st.session_state.pop("global_explanation", None)
    st.session_state.pop("local_explanation", None)
    st.success("🎉 Model trained successfully!")
    
    # DISPLAY RESULTS
    st.write("### 🔍 Task Type")
    st.code(result["task_type"])

    st.write("### 📂 Model Saved At")
    st.code(result["model_path"])

    st.write("### 📊 Metrics")
    st.json(result["metrics"])
    st.write("### Cross-validated model comparison")
    st.dataframe(pd.DataFrame(result.get("model_comparison", [])), use_container_width=True)

    if "sample_predictions" in result:
        st.subheader("🔮 Sample Predictions")
        st.json(result["sample_predictions"])


# ======================================================
# 🔥 BUTTON 2 — A2A TRIGGERED TRAINING (ModelAgent)
# ======================================================
if st.button("🤖 Train Model via Agent (A2A)", key="agent_train_btn"):
    st.info("⏳ Agent running… please wait.")

    result = model_agent.run(df, target)

    if result["status"] == "error":
        st.error(result["error"])
    else:
        st.session_state["model_output"] = result
        st.session_state.pop("global_explanation", None)
        st.session_state.pop("local_explanation", None)
        st.success("🤖 ModelAgent finished training!")
        st.json(result)


# ======================================================
# 🔄 A2A Auto Trigger (EDA → Model)
# ======================================================
st.markdown("---")
st.subheader("🔄 Agent Chain Trigger Status")

auto_res = model_agent.poll_messages_and_run(df)

if auto_res:
    st.success("🚀 Auto-triggered model training completed!")
    st.json(auto_res)


# ======================================================
# IF MODEL EXISTS — SHOW SUMMARY
# ======================================================
st.markdown("---")
if "model_output" in st.session_state:

    result = st.session_state["model_output"]
    st.subheader("📊 Current Model Summary")

    metrics = result.get("metrics", {})

    st.write("### 🔍 Task Type")
    st.code(result["task_type"])

    # Classification
    if result["task_type"] == "classification":
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", round(acc, 3) if acc else "N/A")
        col2.metric("F1 Score", round(f1, 3) if f1 else "N/A")

    # Regression
    else:
        mse = metrics.get("mse")
        r2 = metrics.get("r2")

        col1, col2 = st.columns(2)
        col1.metric("MSE", round(mse, 3) if mse else "N/A")
        col2.metric("R² Score", round(r2, 3) if r2 else "N/A")

    st.json(metrics)
    if result.get("model_comparison"):
        st.write("### Cross-validated model comparison")
        st.dataframe(pd.DataFrame(result["model_comparison"]), use_container_width=True)
    if result.get("artifact_path") and result.get("status") == "success":
        st.markdown("### Model explanations")
        explain_col, local_col = st.columns(2)
        with explain_col:
            if st.button("Compute global feature importance"):
                explanation = ExplainabilityAgent().explain_global(result["artifact_path"], df, target=target)
                st.session_state["global_explanation"] = explanation
            if st.session_state.get("global_explanation"):
                st.json(st.session_state["global_explanation"])
        with local_col:
            row_index = st.number_input("Row index to explain", min_value=0, max_value=max(0, len(df) - 1), value=0)
            if st.button("Explain this prediction"):
                explanation = ExplainabilityAgent().explain_prediction(
                    result["artifact_path"], df.iloc[int(row_index)], df)
                st.session_state["local_explanation"] = explanation
            if st.session_state.get("local_explanation"):
                st.json(st.session_state["local_explanation"])

# ---------------------------------------------------------
# GEMINI INSIGHTS: Explain model results
# ---------------------------------------------------------
st.markdown("---")
st.subheader("✨ Gemini Insights (AutoML Explanation)")

if st.button("💡 Explain My Model with Gemini"):
    with st.spinner("Generating insights…"):
        current_result = st.session_state.get("model_output")
        explanation = (gemini_explain_model(current_result) if current_result else
                      "Train a model first. No model result is available to explain.")

    st.markdown("### 📘 Model Explanation")
    st.write(explanation)
