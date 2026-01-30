from wrappers.base import SafetyWrapper, WrapperDecision

class QueryBudgetWrapper(SafetyWrapper):
    """
    Wrapper that allows a limited number of re-queries before refusing.
    """
    def __init__(self, max_requeries=2):
        super().__init__(name="query_budget", config={"max_requeries": max_requeries})
        self.query_counts = {}

    def decide(self, user_prompt, model_output, history):
        if user_prompt not in self.query_counts:
            self.query_counts[user_prompt] = 0

        if "bomb" not in model_output.lower():
            return WrapperDecision.ALLOW, None

        self.query_counts[user_prompt] += 1

        if self.query_counts[user_prompt] > self.config["max_requeries"]:
            return WrapperDecision.BLOCK, None
        else:
            return WrapperDecision.REQUERY, None
