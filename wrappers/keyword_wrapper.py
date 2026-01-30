from wrappers.base import SafetyWrapper, WrapperDecision


class KeywordFilterWrapper(SafetyWrapper):
    """
    Blocks outputs containing banned keywords.
    """

    def __init__(self, banned_keywords=None):
        super().__init__(
            name="keyword_filter",
            config={"banned_keywords": banned_keywords or []}
        )

    def decide(self, user_prompt, model_output, history):
        output_lower = model_output.lower()

        for keyword in self.config["banned_keywords"]:
            if keyword.lower() in output_lower:
                return WrapperDecision.BLOCK, None

        return WrapperDecision.ALLOW, None
