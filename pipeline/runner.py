import sys
import json
import os
from wrappers.history_wrapper import HistoryBasedWrapper
from wrappers.query_budget_wrapper import QueryBudgetWrapper
from models.llm_client import BlackBoxLLM
from wrappers.keyword_wrapper import KeywordFilterWrapper
from wrappers.base import WrapperDecision
from pipeline.logger import log_interaction

# Load config (fallbacks provided)
CONFIG_FILE = os.path.join("config", "config.json")
DEFAULT_CONFIG = {
    "banned_keywords": ["bomb", "kill", "weapon", "hack"],
    "safe_refusal": "I'm sorry, but I can’t help with that request.",
    "log_file": "logs/experiment_001.jsonl",
    "history_file": "logs/history.jsonl"
}

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as cf:
        CONFIG = json.load(cf)
except Exception:
    CONFIG = DEFAULT_CONFIG

SAFE_REFUSAL = CONFIG.get("safe_refusal", DEFAULT_CONFIG["safe_refusal"])
LOG_FILE = CONFIG.get("log_file", DEFAULT_CONFIG["log_file"])
HISTORY_FILE = CONFIG.get("history_file", DEFAULT_CONFIG["history_file"])


def _read_history(history_file):
    if not os.path.exists(history_file):
        return []
    history = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except Exception:
                # ignore malformed lines
                continue
    return history


def _append_history(history_file, entry):
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_pipeline(prompt: str, wrapper_type: str = "keyword"):
    # Load persistent history
    history = _read_history(HISTORY_FILE)

    model = BlackBoxLLM()

    # Select wrapper dynamically based on type and config
    banned = CONFIG.get("banned_keywords", DEFAULT_CONFIG["banned_keywords"])

    if wrapper_type == "baseline":
        wrapper = None
    elif wrapper_type == "keyword":
        wrapper = KeywordFilterWrapper(banned_keywords=banned)
    elif wrapper_type == "history":
        wrapper = HistoryBasedWrapper(banned_keywords=banned, history_limit=3)
    elif wrapper_type == "query_budget":
        wrapper = QueryBudgetWrapper(max_requeries=2, banned_keywords=banned)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    raw_outputs = []
    decisions = []

    # First model call
    raw_output = model.generate(prompt)
    raw_outputs.append(raw_output)

    if wrapper is None:
        decision = WrapperDecision.ALLOW
        payload = None
    else:
        decision, payload = wrapper.decide(
            user_prompt=prompt,
            model_output=raw_output,
            history=history
        )

    decisions.append(decision.value)

    # Handle REQUERY flow: ask model again with revised prompt until a final decision or until blocks
    while decision == WrapperDecision.REQUERY:
        # Expect payload to contain a revised prompt
        revised_prompt = None
        if isinstance(payload, dict):
            revised_prompt = payload.get("revised_prompt")

        if not revised_prompt:
            # If wrapper didn't provide a revised prompt, use a default safe rephrase
            revised_prompt = f"Please provide a safe, non-actionable, and non-harmful response to the following prompt:\n\n{prompt}"

        raw_output = model.generate(revised_prompt)
        raw_outputs.append(raw_output)

        # Re-run decision with the new output
        decision, payload = wrapper.decide(
            user_prompt=prompt,
            model_output=raw_output,
            history=history
        )
        decisions.append(decision.value)

        # loop will exit if decision is ALLOW or BLOCK (or stays REQUERY until exhausted by wrapper)

    # Persist interaction (store the last model output as the effective model output)
    history_entry = {"user": prompt, "model": raw_outputs[-1]}
    _append_history(HISTORY_FILE, history_entry)

    if decision == WrapperDecision.ALLOW:
        final_output = raw_outputs[-1]
    elif decision == WrapperDecision.BLOCK:
        final_output = SAFE_REFUSAL
    else:  # Should not reach here, but default to safe refusal
        final_output = SAFE_REFUSAL

    log_interaction(LOG_FILE, {
        "model": model.model_name,
        "wrapper": "baseline" if wrapper is None else wrapper.name,
        "wrapper_config": {} if wrapper is None else wrapper.config,
        "user_prompt": prompt,
        "raw_outputs": raw_outputs,
        "decisions": decisions,
        "final_output": final_output,
        "model_calls": len(raw_outputs)
    })

    # Return final output and number of model calls (useful for batch metrics)
    return final_output, len(raw_outputs)


# ------------------------
# CLI only code below
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.runner \"your prompt here\" [wrapper_type]")
        sys.exit(1)

    user_prompt = sys.argv[1]
    wrapper_type = sys.argv[2] if len(sys.argv) > 2 else "keyword"

    output, calls = run_pipeline(user_prompt, wrapper_type)
    print("\nFINAL OUTPUT:\n", output)
    print("Model calls:", calls)

