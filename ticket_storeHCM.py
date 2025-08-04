import json
from pathlib import Path

STORE_PATH = Path("ticket_store.json")

def load_store():
    if STORE_PATH.exists():
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_store(data):
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_ticket_info(ticket):
    data = load_store()
    return data.get(ticket)

def save_ticket_info(ticket, visitor_id, conversation_id):
    data = load_store()
    data[ticket] = {
        "visitorId": visitor_id,
        "conversationId": conversation_id
    }
    save_store(data)
