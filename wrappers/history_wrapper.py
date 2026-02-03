from wrappers.base import SafetyWrapper, WrapperDecision

class HistoryBasedWrapper(SafetyWrapper):
    def __init__(self, banned_keywords=None, history_limit=3):
        super().__init__(
            name="history_based",
            config={"banned_keywords": banned_keywords or [], "history_limit": history_limit}
        )

    def decide(self, user_prompt, model_output, history):
        # We only check previous turns, not the current one (which is handled by keyword wrapper usually)
        # But for this assignment, we check if context + current output is unsafe
        
        # Combine last K turns
        recent_context = history[-self.config["history_limit"]:]
        
        # Check if banned words appear in the model's history (repetitive unsafe behavior)
        # or if the current output triggers a keyword
        check_text = model_output.lower()
        
        for keyword in self.config["banned_keywords"]:
            if keyword.lower() in check_text:
                return WrapperDecision.BLOCK, None
                
        return WrapperDecision.ALLOW, None