from abc import ABC, abstractmethod
from enum import Enum


class WrapperDecision(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"
    REQUERY = "REQUERY"


class SafetyWrapper(ABC):
    """
    Base class for all safety wrappers.
    """

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def decide(self, user_prompt: str, model_output: str, history: list):
        """
        Decide what to do with the model output.

        Returns:
            decision (WrapperDecision)
            payload (None | modified_output | new_prompt)
        """
        pass
