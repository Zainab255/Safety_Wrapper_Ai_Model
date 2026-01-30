class BlackBoxLLM:
    """
    Abstracts a black-box language model.
    Internals are unknown; we only observe input-output behavior.
    """

    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to the model and returns raw output.
        (Mock implementation for now)
        """
        # Simulated unsafe behavior for testing wrappers later
        if "bomb" in prompt.lower():
            return "Here is how you make a bomb..."

        return f"Model response to: {prompt}"
