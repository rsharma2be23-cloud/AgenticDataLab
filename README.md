# Multi-Agent AutoML Data Analyst (MCP + A2A + Gemini Powered)

Automated Profiler → EDA → AutoML → Verifier → Notebook Synthesizer → Gemini Insights

🚀 Live Demo (Render Deployment):
https://multiagent-data-analyst.onrender.com/

Track: Enterprise Agents

Tech: Python, Streamlit, Gemini, MCP Tools, A2A Bus, Multi-Agent Architecture

# Overview

The Multi-Agent Data Analyst is a fully automated, end-to-end data analysis pipeline powered by multiple specialized agents working together.
It uploads a dataset, analyzes it, builds ML models, verifies the results, generates notebooks, and produces final insights — without any manual coding.

This project demonstrates:

✔ Multi-agent systems

✔ A2A (Agent-to-Agent) 

✔ Tool-based agent execution (MCP Tools)

✔ Sessions & memory

✔ Context-aware notebook synthesis

✔ Gemini-powered explanations

✔ Streamlit multi-page application


# Problem Statement

Performing data analysis typically requires switching between tools, writing repetitive code, running models manually, validating outputs, and documenting everything.

For beginners, this is overwhelming.

For analysts, it's time-consuming.

For teams, it’s inconsistent.

# Goal: Build an agentic system that automates the entire workflow — from raw data to verified insights and notebook generation.

# Why Agents?

Agents make the system:

 -> Modular — each agent does one job

 -> Autonomous — actions happen without the user triggering each step

 -> Traceable — every step is observable

 -> Composable — agents communicate using A2A bus

 -> Extensible — new agents (e.g., Gemini Reviewer) can be added anytime

Instead of one giant notebook, the intelligence is distributed:

# Agent Roles

 -> Profiler Agent – inspects dataset, finds issues

 -> EDA Agent – generates charts, summaries, anomalies

 -> Model Agent (AutoML) – builds ML pipelines automatically

 -> Verifier Agent – detects inconsistencies, bad models, missing columns

 -> Notebook Synthesizer Agent – creates a clean notebook combining all outputs

 -> Gemini Agent – explains the ML results in human-friendly language

# Each agent writes outputs to memory → A2A orchestrates → Next agent reacts.

# Architecture

<img width="2452" height="1286" alt="image" src="https://github.com/user-attachments/assets/92641b2f-fceb-493a-b481-345e5e341de4" />


1️⃣ ProfilerAgent

✔ Reads dataset

✔ Detects column types

✔ Finds missing values

✔ Sends message → EDAAgent

2️⃣ EDAAgent

✔ Creates correlations, histograms, outlier analysis

✔ Saves all plots via MCP FileTools

✔ Sends message → ModelAgent

3️⃣ ModelAgent

✔ Auto-detects task type (classification/regression)

✔ Builds full ML pipeline (imputation + scaling + encoding)

✔ Tunes models

✔ Saves best model

✔ Sends message → VerifierAgent

4️⃣ VerifierAgent

✔ Validates model quality

✔ Computes quality tag (“Good”, “Acceptable”, “Weak”)

✔ Sends message → NotebookAgent

5️⃣ NotebookSynthesizerAgent

✔ Builds a full auto-generated Jupyter Notebook

✔ Embeds all results and images

✔ Saves notebook through FileTools

6️⃣ Gemini Integration

✔ Gemini generates:

✔ Model explanations

✔ Recommendations

✔ Summaries

Plain-English explanations for beginners

7️⃣ Streamlit UI

✔ Beautiful dashboard with:

✔ Dataset Explorer

✔ EDA Dashboard

✔ AutoML Dashboard

✔ Verifier & Notebook Builder

✔ A2A Communications Console

# Setup Instructions

Clone Repo

 1. git clone https://github.com/yourusername/multiagent-data-analyst
 
 2. cd multiagent-data-analyst

 3. pip install -r requirements.txt

 4. Add Gemini API Key

 5. Create .env:

 6. Run Streamlit -> streamlit run streamlit_app/app.py

# Demo (Screenshots)

# Dataset Upload

<img width="2928" height="1746" alt="image" src="https://github.com/user-attachments/assets/6e6dddb5-a33b-47be-a2f0-44a6f26a06ec" />

# EDA Dashboard

<img width="2938" height="1760" alt="image" src="https://github.com/user-attachments/assets/ffc64ed4-f32c-49af-ae7d-b5b1de07770c" />

<img width="2284" height="1518" alt="image" src="https://github.com/user-attachments/assets/ecaa796b-8470-4cdc-89f1-a9ee5437e221" />

<img width="2260" height="936" alt="image" src="https://github.com/user-attachments/assets/68182f66-62b3-4dbd-b6aa-e57ba8bb532c" />

# AutoML Results

<img width="2894" height="1566" alt="image" src="https://github.com/user-attachments/assets/214fd60b-055d-4143-bee5-7283aca09528" />

<img width="2940" height="1584" alt="image" src="https://github.com/user-attachments/assets/a27fe305-5c8c-4799-b049-fce96a0a1326" />

# Gemini Explanation

<img width="2326" height="1528" alt="image" src="https://github.com/user-attachments/assets/2f79d85b-d176-4b21-8a45-0c70d5e0e845" />

# A2A Console

<img width="2940" height="1774" alt="image" src="https://github.com/user-attachments/assets/9a0250a0-2415-432f-a3a0-b2c05aea8a7e" />

# Notebook generated

<img width="2354" height="1500" alt="image" src="https://github.com/user-attachments/assets/7d5ab7c6-97e1-436e-885d-798fb015f72f" />

# Profiler Agent Output - 

<img width="2940" height="1528" alt="image" src="https://github.com/user-attachments/assets/228e3f40-883a-41a4-abb9-58597050d94d" />

<img width="2898" height="1560" alt="image" src="https://github.com/user-attachments/assets/3dc7df6d-6486-4019-87c3-c837a85398db" />

# Verifier Agent - 

<img width="2310" height="1342" alt="image" src="https://github.com/user-attachments/assets/21e769b2-bd25-45ef-a54c-c274eaa28262" />


# Tools & Technologies Used

🚀Category	Tools

🚀Multi-Agent	Custom Agents, A2A Bus

🚀LLM	Gemini 1.5 Flash

🚀UI	Streamlit

🚀ML	Scikit-Learn

🚀Storage	Custom MemoryTools

🚀Notebook	nbformat

🚀Deployment	Render

🚀Visualization	Plotly, Matplotlib, Seaborn

🚀 LLM	Gemini 1.5 Flash


# 🗂 Project Structure

# multiagent-data-analyst/

│

├── src/

│   ├── agents/

│   ├── core/

│   ├── tools/

│   │   ├── file_tools.py

│   │   ├── dataset_tools.py

│   │   ├── memory_tools.py

│   │   ├── model_tools.py

│   │   └── notebook_tools.py

│

├── streamlit_app/

│   ├── app.py

│   └── pages/

│       ├── AutoML.py

│       ├── Profiler.py

│       ├── EDA_Dashboard.py

│       ├── Notebook_Report.py

│       ├── Verifier.py

│       └── A2A_Dashboard.py

│

├── streamlit_app_storage/

│   ├── memory/

│   ├── uploads/

│   └── reports/

│

└── README.md

 # Future Improvements

1. Add RAG-based “Data Question Answering Agent”

2. Add deployment on Google Cloud Run using Docker

3. Add Evaluation Agent for model fairness

4. Provide more AutoML models (XGBoost, LightGBM)

5. Add voice-based interaction mode

# Credits

Built by Vaishnavi Sharma as part of
Google x Kaggle – Agents Intensive 

# If you find this useful, ⭐ star the repo!
