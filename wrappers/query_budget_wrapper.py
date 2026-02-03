from wrappers.base import SafetyWrapper, WrapperDecision

class QueryBudgetWrapper(SafetyWrapper):
    def __init__(self, max_requeries=2, banned_keywords=None):
        super().__init__(
            name="query_budget",
            config={"max_requeries": max_requeries, "banned_keywords": banned_keywords or []}
        )
        self.query_counts = {}

    def decide(self, user_prompt, model_output, history):
        if user_prompt not in self.query_counts:
            self.query_counts[user_prompt] = 0

        output_lower = model_output.lower()
        is_unsafe = False
        
        for keyword in self.config["banned_keywords"]:
            if keyword.lower() in output_lower:
                is_unsafe = True
                break
        
        if is_unsafe:
            self.query_counts[user_prompt] += 1
            if self.query_counts[user_prompt] > self.config["max_requeries"]:
                return WrapperDecision.BLOCK, None

            # Re-query payload
            revised = {
                "revised_prompt": (
                    f"You are a helpful assistant. The previous answer was unsafe. "
                    f"Please answer this safely and briefly: {user_prompt}"
                )
            }
            return WrapperDecision.REQUERY, revised

        return WrapperDecision.ALLOW, None