import os
import sys
import uuid

import pandas as pd
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.execution_engine import ExecutionEngine
from core.task_state import TaskStateStore
from core.tracing import ExecutionTracer, redact

st.set_page_config(page_title="Agentic Analysis | AgenticDataLab", layout="wide")
st.title("Agentic Analysis")
st.caption("Planner-selected operations run through the allowlisted tool registry; results and trace events are persisted.")

df = st.session_state.get("uploaded_df")
if df is None:
    st.info("Upload a CSV on the Home page to start an analysis.")
    st.stop()

st.caption(f"Dataset: {st.session_state.get('dataset_name', 'current session')} · {len(df):,} rows · {len(df.columns)} columns")
with st.expander("Dataset quality and schema", expanded=True):
    left, right = st.columns(2)
    left.write("Column types and missing values")
    left.dataframe(pd.DataFrame({"type": df.dtypes.astype(str), "missing": df.isna().sum()}), use_container_width=True)
    right.write("Preview")
    right.dataframe(df.head(10), use_container_width=True)
    if st.button("Run data-quality analysis", key="quality"):
        try:
            result = ExecutionEngine().tools.execute("analyze_data_quality", {}, df, {"results": {}})
            st.session_state["quality_result"] = result
        except Exception as exc:
            st.error(f"Quality analysis failed ({type(exc).__name__}). Check dependencies and input format.")
    if st.session_state.get("quality_result"):
        st.json(st.session_state["quality_result"])

st.subheader("Ask the analyst")
goal = st.text_area("Analysis request", "Profile the data, check its quality, explore relationships, compare suitable models, and report findings.")
target = st.selectbox("Target column (optional)", ["Auto-detect"] + [str(col) for col in df.columns])
if st.button("Plan and run", type="primary"):
    request = goal.strip()
    if not request:
        st.error("Enter an analysis request.")
    else:
        if target != "Auto-detect":
            request += f" Target: {target}"
        task_id = str(uuid.uuid4())
        st.session_state["agentic_request"] = request
        st.session_state["agentic_task_id"] = task_id
        with st.spinner("Planner and tools are running…"):
            try:
                st.session_state["agentic_result"] = ExecutionEngine().run(df, request, task_id=task_id)
            except Exception as exc:
                st.error(f"Workflow failed ({type(exc).__name__}). Review dependency setup and try a supported goal.")

store = TaskStateStore()
states = store.list()
pending = [(task_id, state) for task_id, state in states.items() if state.get("status") == "pending_approval"]
if pending:
    st.subheader("Approval required")
    for task_id, state in pending:
        for step in state.get("plan", {}).get("steps", []):
            approval = state.get("approvals", {}).get(step.get("id"), {})
            if approval.get("status") != "pending_approval":
                continue
            st.warning(f"Task {task_id}: {step.get('agent')} requests `{step.get('tool')}`")
            st.json({"reason": step.get("reason"), "parameters": step.get("inputs", {})})
            approve_col, reject_col = st.columns(2)
            if approve_col.button("Approve and continue", key=f"approve-{task_id}-{step['id']}"):
                try:
                    engine = ExecutionEngine()
                    engine.resolve_approval(task_id, step["id"], True)
                    st.session_state["agentic_result"] = engine.run(df, state.get("user_request", ""), task_id=task_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not resume the approved workflow ({type(exc).__name__}).")
            if reject_col.button("Reject operation", key=f"reject-{task_id}-{step['id']}"):
                try:
                    ExecutionEngine().resolve_approval(task_id, step["id"], False)
                    st.session_state["agentic_result"] = {"task_id": task_id, "status": "rejected", "state": store.get(task_id), "results": {}}
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save the approval decision ({type(exc).__name__}).")

result = st.session_state.get("agentic_result")
if result:
    state = result.get("state") or store.get(result.get("task_id")) or {}
    st.subheader(f"Workflow status: {result.get('status', state.get('status', 'unknown'))}")
    if state.get("plan"):
        st.markdown("**Plan**")
        st.json(redact(state["plan"]))
    if state.get("critic_result"):
        st.markdown("**Critic review**")
        st.json(redact(state["critic_result"]))
    outputs = result.get("results") or {record.get("tool"): record.get("output") for record in state.get("tool_outputs", {}).values()}
    if outputs:
        for tool, output in outputs.items():
            with st.expander(tool, expanded=tool in {"train_model", "analyze_data_quality", "run_eda"}):
                st.json(redact(output))
                if tool == "train_model" and isinstance(output, dict):
                    st.caption("CV metrics summarize cross-validation folds. Final metrics are held-out test results; training metrics describe the fitted training partition.")
                    if output.get("model_comparison"):
                        st.dataframe(pd.DataFrame(output["model_comparison"]), use_container_width=True)
                    if output.get("artifact_path"):
                        st.caption(f"Model artifact: {output['artifact_path']}")
                if tool == "generate_report" and isinstance(output, dict):
                    report_path = output.get("path") or output.get("notebook_path")
                    if report_path and output.get("status") == "success":
                        st.caption(f"Generated report: {report_path}")
    st.caption("Numerical results come from deterministic tools. Planner/explanation text may use Gemini only when configured; unavailable tools are shown as errors or skipped results.")
    trace_events = ExecutionTracer().list(result.get("task_id"))
    st.subheader("Execution trace")
    if trace_events:
        st.dataframe(pd.DataFrame(trace_events), use_container_width=True)
    else:
        st.info("No trace events recorded for this workflow.")

st.subheader("Persistent experiment history")
history = []
for task_id, record in states.items():
    for output in record.get("tool_outputs", {}).values():
        if output.get("tool") == "train_model" and isinstance(output.get("output"), dict):
            model = output["output"]
            history.append({"task_id": task_id, "status": record.get("status"), "model": model.get("model_name"),
                            "task_type": model.get("task_type"), "metrics": model.get("metrics"),
                            "CV": model.get("cv_results"), "experiment_id": model.get("experiment_id")})
if history:
    st.dataframe(pd.DataFrame(history), use_container_width=True)
else:
    st.info("No persisted model runs yet. MLflow history is available only when MLflow tracking is configured.")
