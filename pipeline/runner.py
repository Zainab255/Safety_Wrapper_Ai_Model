import sys
from wrappers.history_wrapper import HistoryBasedWrapper
from wrappers.query_budget_wrapper import QueryBudgetWrapper
from models.llm_client import BlackBoxLLM
from wrappers.keyword_wrapper import KeywordFilterWrapper
from wrappers.base import WrapperDecision
from pipeline.logger import log_interaction

SAFE_REFUSAL = "I'm sorry, but I can’t help with that request."
LOG_FILE = "logs/experiment_001.jsonl"


def run_pipeline(prompt: str, wrapper_type: str = "keyword"):
    history = []

    model = BlackBoxLLM()

    # Select wrapper dynamically
    if wrapper_type == "baseline":
        wrapper = None
    elif wrapper_type == "keyword":
        wrapper = KeywordFilterWrapper(banned_keywords=["bomb", "kill", "weapon"])
    elif wrapper_type == "history":
        wrapper = HistoryBasedWrapper(banned_keywords=["bomb", "kill", "weapon"], history_limit=3)
    elif wrapper_type == "query_budget":
        wrapper = QueryBudgetWrapper(max_requeries=2)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    raw_outputs = []
    decisions = []

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
    history.append({"user": prompt, "model": raw_output})

    if decision == WrapperDecision.ALLOW:
        final_output = raw_output
    elif decision == WrapperDecision.BLOCK:
        final_output = SAFE_REFUSAL
    else:  # REQUERY or any other
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

    return final_output


# ------------------------
# CLI only code
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.runner \"your prompt here\" [wrapper_type]")
        sys.exit(1)

    user_prompt = sys.argv[1]
    wrapper_type = sys.argv[2] if len(sys.argv) > 2 else "keyword"

    output = run_pipeline(user_prompt, wrapper_type)
    print("\nFINAL OUTPUT:\n", output)

