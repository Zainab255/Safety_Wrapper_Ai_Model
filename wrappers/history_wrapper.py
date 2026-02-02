from wrappers.base import SafetyWrapper, WrapperDecision

class HistoryBasedWrapper(SafetyWrapper):
    """
    Uses recent conversation history to make safety decisions.
    If any recent model output contains a banned keyword, BLOCK.
    History is expected to be a list of dicts: {"user": ..., "model": ...}
    """
    def __init__(self, banned_keywords=None, history_limit=5):
        super().__init__(
            name="history_based",
            config={"banned_keywords": banned_keywords or [],
                    "history_limit": history_limit}
        )

    def decide(self, user_prompt, model_output, history):
        # Keep history entries as dicts for consistency
        history.append({"user": user_prompt, "model": model_output})

        # Trim history in-place to preserve the same list object
        limit = self.config["history_limit"]
        if len(history) > limit:
            history[:] = history[-limit:]

        for past_entry in history:
            past_output = past_entry.get("model", "")
            for keyword in self.config["banned_keywords"]:
                if keyword.lower() in past_output.lower():
                    return WrapperDecision.BLOCK, None

        return WrapperDecision.ALLOW, None
