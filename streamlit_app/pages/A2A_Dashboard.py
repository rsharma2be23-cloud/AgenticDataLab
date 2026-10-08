import streamlit as st
import pandas as pd
import os
import sys

# Import A2A Bus
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)
from tools.a2a_tools import A2A_GLOBAL_BUS as a2a

st.set_page_config(page_title="Agent-to-Agent Communication", layout="wide")
st.title("🛰️ Agent-to-Agent (A2A) Communication Console")


st.markdown("This dashboard shows **all messages sent between agents**, their inbox, and full audit logs.")

# =====================================================================
# SEND A MANUAL MESSAGE
# =====================================================================
st.subheader("📤 Send Manual A2A Message")

col1, col2 = st.columns(2)

with col1:
    sender = st.text_input("Sender Agent", "UI")
with col2:
    receiver = st.text_input("Receiver Agent", "EDAAgent")

topic = st.text_input("Topic", "test-message")
payload = st.text_area("Payload", "Hello from UI")

if st.button("Send Message"):
    result = a2a.send(sender, receiver, topic, payload)
    st.success("Message sent!")
    st.json(result)


# =====================================================================
# VIEW INBOX
# =====================================================================
st.markdown("---")
st.subheader("📥 Inbox (Messages addressed to the UI)")

inbox = a2a.get_inbox("UI")

if inbox:
    st.dataframe(pd.DataFrame(inbox), use_container_width=True)
else:
    st.info("No messages for UI yet.")

# =====================================================================
# VIEW ALL MESSAGES
# =====================================================================
st.markdown("---")
st.subheader("🧾 All Messages Log")

log = a2a.get_audit_log()

if log:
    st.dataframe(pd.DataFrame(log), use_container_width=True)
else:
    st.info("No messages sent yet.")

# =====================================================================
# LIVE REFRESH
# =====================================================================
st.markdown("---")
if st.button("🔄 Refresh Console"):
    st.rerun()
