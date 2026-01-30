from wrappers.base import SafetyWrapper, WrapperDecision

class HistoryBasedWrapper(SafetyWrapper):
    """
    Uses recent conversation history to make safety decisions.
    If any recent model output contains a banned keyword, BLOCK.
    """
    def __init__(self, banned_keywords=None, history_limit=5):
        super().__init__(
            name="history_based",
            config={"banned_keywords": banned_keywords or [],
                    "history_limit": history_limit}
        )

    def decide(self, user_prompt, model_output, history):
        history.append(model_output)
        if len(history) > self.config["history_limit"]:
            history = history[-self.config["history_limit"]:]

        for past_output in history:
            for keyword in self.config["banned_keywords"]:
                if keyword.lower() in past_output.lower():
                    return WrapperDecision.BLOCK, None

        return WrapperDecision.ALLOW, None
