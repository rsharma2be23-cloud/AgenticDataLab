# File: src/orchestrator.py
# Updated orchestrator to create A2ABus and wire agents. Use this orchestrator to start the pipeline
import os
import sys
import time

# ensure src is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

# import bus and agents
from core.a2a_bus import A2ABus
from agents.profiler_agent import ProfilerAgent
from agents.eda_agent import EDAAgent
from agents.model_agent import ModelAgent
from agents.verifier_agent import VerifierAgent
from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent

def main():
    print("=== ORCHESTRATOR STARTED ===")

    memory = MemoryTools()
    file_tool = FileTools()

    # instantiate A2A bus with persistence into memory
    a2a = A2ABus(memory=memory, persist=True)

    # register agents
    profiler = ProfilerAgent(a2a_bus=a2a)
    eda = EDAAgent(a2a_bus=a2a)
    model_agent = ModelAgent(a2a_bus=a2a)
    verifier = VerifierAgent(a2a_bus=a2a)
    notebook_agent = NotebookSynthesizerAgent()

    # Example run flow orchestrated manually (but now agents also notify each other)
    # 1) Load CSV from data/sample.csv (or adjust path)
    path = os.path.join(BASE_DIR, "sample.csv")
    if not os.path.exists(path):
        print("Sample CSV not found:", path)
        return

    import pandas as pd
    df = pd.read_csv(path)

    # 2) Run profiler (this will publish a message to eda)
    print("Running profiler...")
    profiler_out = profiler.run(df)
    print("Profiler:", profiler_out.get("status"))

    # 3) EDA agent can poll messages and decide to run. We call poll then run
    print("EDA polling messages and auto-running if needed...")
    eda_poll = eda.poll_messages_and_run()
    # If poll found a profiler.completed message, we run EDA now with the same df
    if eda_poll:
        eda_out = eda.run(df)
    else:
        # you may run EDA manually if no message
        eda_out = eda.run(df)

    print("EDA:", eda_out.get("status"))

    # 4) Model agent will be triggered by EDA (A2A). We'll fetch and run if messages exist
    print("ModelAgent polling for messages (run if EDA completed)...")
    msgs = a2a.fetch("model", consume=True)
    ran_model = None
    for m in msgs:
        if m.get("topic") == "eda.completed":
            # run training - pick a target heuristically or set below
            target_candidates = eda_out.get("numeric_columns", [])
            if target_candidates:
                target = target_candidates[-1]
            else:
                target = df.columns[-1]
            print("Training target chosen:", target)
            ran_model = model_agent.run(df, target_col=target)
            print("Model run status:", ran_model.get("status"))

    if not ran_model:
        # optionally run model manually here:
        print("No EDA->Model message found, running model manually with last column as target...")
        target = df.columns[-1]
        ran_model = model_agent.run(df, target_col=target)

    # 5) Verifier agent polls and verifies
    print("Verifier polling messages and running verification if model trained...")
    verifier_poll = verifier.poll_messages_and_run()
    if not verifier_poll:
        # attempt manual run if we have model_output in memory
        mo = memory.load().get("memory", {}).get("model_output")
        if mo:
            verifier_res = verifier.run(mo)
            print("Verifier manual run:", verifier_res.get("status"))
    else:
        print("Verifier auto-run result:", verifier_poll.get("status"))

    # 6) Notebook generator: when all pieces are available, create notebook
    mem = memory.load().get("memory", {})
    profiler_output = mem.get("profiler_output")
    eda_output = mem.get("eda_output")
    model_output = mem.get("model_output")
    verifier_output = mem.get("verifier_output")

    if profiler_output and eda_output and model_output and verifier_output:
        print("Generating notebook...")
        nb_res = notebook_agent.run(profiler_output, eda_output, model_output, verifier_output)
        print("Notebook result:", nb_res)
    else:
        print("Not all components available; notebook generation skipped.")

    print("=== ORCHESTRATOR FINISHED ===")

if __name__ == "__main__":
    main()
