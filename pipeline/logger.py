import json
from datetime import datetime


def log_interaction(log_file, log_entry):
    log_entry["timestamp"] = datetime.utcnow().isoformat()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
