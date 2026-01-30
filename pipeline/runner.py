import sys
from models.llm_client import BlackBoxLLM
from wrappers.noop_wrapper import NoOpWrapper
from wrappers.base import WrapperDecision


SAFE_REFUSAL = "I'm sorry, but I can’t help with that request."


def run_pipeline(prompt: str):
    history = []

    model = BlackBoxLLM()
    wrapper = NoOpWrapper()

    raw_output = model.generate(prompt)

    decision, payload = wrapper.decide(
        user_prompt=prompt,
        model_output=raw_output,
        history=history
    )

    if decision == WrapperDecision.ALLOW:
        final_output = raw_output
    elif decision == WrapperDecision.BLOCK:
        final_output = SAFE_REFUSAL
    elif decision == WrapperDecision.MODIFY:
        final_output = payload
    elif decision == WrapperDecision.REQUERY:
        final_output = SAFE_REFUSAL
    else:
        raise ValueError("Unknown wrapper decision")

    return final_output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.runner \"your prompt here\"")
        sys.exit(1)

    user_prompt = sys.argv[1]
    result = run_pipeline(user_prompt)
    print("\nFINAL OUTPUT:\n", result)
