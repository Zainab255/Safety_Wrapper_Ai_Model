from wrappers.base import SafetyWrapper, WrapperDecision


class NoOpWrapper(SafetyWrapper):
    """
    Baseline wrapper: always allows the model output.
    """

    def __init__(self):
        super().__init__(name="no_op")

    def decide(self, user_prompt, model_output, history):
        return WrapperDecision.ALLOW, None
