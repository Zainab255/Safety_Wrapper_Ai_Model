from wrappers.base import SafetyWrapper, WrapperDecision

class QueryBudgetWrapper(SafetyWrapper):
    """
    Wrapper that allows a limited number of re-queries before refusing.
    Uses configured banned keywords and returns a revised prompt payload when asking for a re-query.
    """
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
        # If no banned keyword found, allow
        for keyword in self.config["banned_keywords"]:
            if keyword.lower() in output_lower:
                # Found a banned keyword: consider re-query or block after limit
                self.query_counts[user_prompt] += 1

                if self.query_counts[user_prompt] > self.config["max_requeries"]:
                    return WrapperDecision.BLOCK, None

                # Provide a revised prompt payload that asks the model to produce a safe answer
                revised = {
                    "revised_prompt": (
                        "Please provide a safe, non-actionable, and non-harmful response to the following prompt:\n\n"
                        + user_prompt
                    )
                }
                return WrapperDecision.REQUERY, revised

        return WrapperDecision.ALLOW, None
