import sys

from models.llm_client import BlackBoxLLM
from wrappers.keyword_wrapper import KeywordFilterWrapper
from wrappers.base import WrapperDecision
from pipeline.logger import log_interaction


SAFE_REFUSAL = "I'm sorry, but I can’t help with that request."
LOG_FILE = "logs/experiment_001.jsonl"


def run_pipeline(prompt: str):
    history = []

    model = BlackBoxLLM()
    wrapper = KeywordFilterWrapper(
        banned_keywords=["bomb", "kill", "weapon"]
    )

    raw_outputs = []
    decisions = []

    raw_output = model.generate(prompt)
    raw_outputs.append(raw_output)

    decision, payload = wrapper.decide(
        user_prompt=prompt,
        model_output=raw_output,
        history=history
    )

    decisions.append(decision.value)

    if decision == WrapperDecision.ALLOW:
        final_output = raw_output
    elif decision == WrapperDecision.BLOCK:
        final_output = SAFE_REFUSAL
    else:
        final_output = SAFE_REFUSAL

    log_interaction(LOG_FILE, {
        "model": model.model_name,
        "wrapper": wrapper.name,
        "wrapper_config": wrapper.config,
        "user_prompt": prompt,
        "raw_outputs": raw_outputs,
        "decisions": decisions,
        "final_output": final_output,
        "model_calls": len(raw_outputs)
    })

    return final_output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.runner \"your prompt here\"")
        sys.exit(1)

    user_prompt = sys.argv[1]
    output = run_pipeline(user_prompt)
    print("\nFINAL OUTPUT:\n", output)
