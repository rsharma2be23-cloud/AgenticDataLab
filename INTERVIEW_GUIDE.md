# AgenticDataLab interview guide

## 60-second architecture summary

AgenticDataLab is a Streamlit workspace around a bounded Planner → Executor → Critic loop. The planner selects from a typed allowlist in `ToolRegistry`; deterministic Python tools perform profiling, data-quality checks, EDA, model training, evaluation, explanations, and report generation. `ExecutionEngine` persists task state, trace events, approvals, and bounded re-planning decisions. The optional Gemini integration selects or explains work; it does not run generated code or calculate metrics.

## Talking points

- **Why agents?** Small roles make planning, tool execution, critique, and communication easier to inspect and evolve than one opaque prompt.
- **How tools run:** A plan is parsed against `AgentPlan`, each input is parsed against its tool schema, and only registered handlers can execute.
- **Why deterministic computation?** Data transformations and metrics should be reproducible, testable Python operations; an LLM is useful for intent and language, not arithmetic authority.
- **How evaluation avoids overclaiming:** Candidate selection uses cross-validation. The output separates CV summaries, training metrics, and the held-out test metrics; users should prefer the held-out results for final reporting and consider dataset size and leakage risks.
- **How failure and replanning work:** Tool failures are recorded, the critic may request a bounded retry, and retry count and events are persisted. A tool failure is surfaced rather than replaced with invented content.
- **MCP and A2A:** MCP exposes registered tools to an external MCP client when the optional SDK is installed. A2A messages carry validated IDs, payloads, and lifecycle status in the local bus; it is not a distributed queue.
- **Security trade-offs:** There is no arbitrary shell or generated Python execution. Uploads are size/row bounded; file tools are confined to their configured storage root. User data is processed in the app process and may be sent to Gemini only through enabled optional features; do not submit sensitive data to external providers without appropriate authorization.
- **Current limitations:** CSV only in the Streamlit upload path; local file-backed state; local A2A; model quality depends on data and environment; no sandboxed notebook execution; optional SDKs must be installed to verify their live integrations.

## Suggested demo

1. Start the app and upload `benchmarks/synthetic_classification.csv`.
2. Inspect types, missingness, and preview on Home.
3. Open Agentic Analysis, run data-quality analysis, then request profiling, EDA, model comparison, evaluation, and report generation for target `target`.
4. Review the actual plan/results and trace table; model metrics distinguish CV, training, and held-out evaluation where the model tool returns them.
5. Set `AGENTIC_APPROVAL_TOOLS=train_model` in `.env`, restart, and repeat to see the pending operation and persisted decision. Do not use this environment variable as a substitute for authorization policy in a multi-user deployment.

The demo fixture is generated synthetic data and contains no personal records.
