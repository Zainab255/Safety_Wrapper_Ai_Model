import json
import os
from wrappers.history_wrapper import HistoryBasedWrapper
from wrappers.query_budget_wrapper import QueryBudgetWrapper
from wrappers.keyword_wrapper import KeywordFilterWrapper
from wrappers.noop_wrapper import NoOpWrapper
from wrappers.base import WrapperDecision
from pipeline.logger import log_interaction
from models.llm_client import BlackBoxLLM

def load_config():
    with open(os.path.join("config", "config.json"), "r") as f:
        return json.load(f)

CONFIG = load_config()

def run_pipeline(prompt, wrapper_type, model_instance, history_file):
    # Load history
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            for line in f:
                history.append(json.loads(line))

    # Initialize Wrapper
    banned = CONFIG["banned_keywords"]
    if wrapper_type == "baseline":
        wrapper = NoOpWrapper()
    elif wrapper_type == "keyword":
        wrapper = KeywordFilterWrapper(banned_keywords=banned)
    elif wrapper_type == "history":
        wrapper = HistoryBasedWrapper(banned_keywords=banned, history_limit=CONFIG["history_limit"])
    elif wrapper_type == "query_budget":
        wrapper = QueryBudgetWrapper(max_requeries=CONFIG["max_requeries"], banned_keywords=banned)
    else:
        raise ValueError("Unknown wrapper")

    raw_outputs = []
    decisions = []
    
    current_prompt = prompt
    # 1. First Generation
    raw_output = model_instance.generate(current_prompt)
    raw_outputs.append(raw_output)

    # 2. Wrapper Check
    decision, payload = wrapper.decide(prompt, raw_output, history)
    decisions.append(decision.value)

    # 3. REQUERY LOOP
    while decision == WrapperDecision.REQUERY:
        revised_prompt = payload["revised_prompt"] if payload else prompt
        raw_output = model_instance.generate(revised_prompt)
        raw_outputs.append(raw_output)
        
        decision, payload = wrapper.decide(prompt, raw_output, history)
        decisions.append(decision.value)

    # 4. Final Output
    if decision == WrapperDecision.BLOCK:
        final_output = CONFIG["safe_refusal"]
    else:
        final_output = raw_outputs[-1]

    # 5. Save History & Log
    new_entry = {"user": prompt, "model": raw_outputs[-1]}
    with open(history_file, "a") as f:
        f.write(json.dumps(new_entry) + "\n")

    log_entry = {
        "model": model_instance.model_name,
        "wrapper": wrapper_type,
        "prompt": prompt,
        "final_output": final_output,
        "calls": len(raw_outputs),
        "decisions": decisions
    }
    log_interaction(CONFIG["log_file"], log_entry)

    return final_output, len(raw_outputs)