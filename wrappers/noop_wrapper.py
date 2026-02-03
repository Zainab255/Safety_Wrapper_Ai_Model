from wrappers.base import SafetyWrapper, WrapperDecision

class NoOpWrapper(SafetyWrapper):
    def __init__(self):
        super().__init__(name="no_op")

    def decide(self, user_prompt, model_output, history):
        return WrapperDecision.ALLOW, None