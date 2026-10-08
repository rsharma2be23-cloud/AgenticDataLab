# src/tools/a2a_tools.py

class A2ABus:

    def __init__(self):
        self.inboxes = {}      # {agent_name: [messages]}
        self.audit = []        # list of all messages

    # --------------------------------------------
    # MAIN SEND METHOD
    # --------------------------------------------
    def send(self, sender, receiver, topic, payload):
        msg = {
            "sender": sender,
            "receiver": receiver,
            "topic": topic,
            "payload": payload
        }

        # Add to audit log
        self.audit.append(msg)

        # Add to receiver inbox
        if receiver not in self.inboxes:
            self.inboxes[receiver] = []
        self.inboxes[receiver].append(msg)

        return {
            "status": "success",
            "message": "Delivered",
            "data": msg
        }

    # --------------------------------------------
    # COMPATIBILITY
    # --------------------------------------------
    def direct_message(self, sender, receiver, topic, payload):
        return self.send(sender, receiver, topic, payload)

    # --------------------------------------------
    # INBOX READER
    # --------------------------------------------
    def get_inbox(self, agent):
        return self.inboxes.get(agent, [])

    # --------------------------------------------
    # AUDIT LOG
    # --------------------------------------------
    def get_audit_log(self):
        return self.audit


# ----------------------------------------------------
# GLOBAL SINGLETON BUS
# ----------------------------------------------------
A2A_GLOBAL_BUS = A2ABus()
