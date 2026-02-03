from abc import ABC, abstractmethod
from enum import Enum

class WrapperDecision(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"
    REQUERY = "REQUERY"

class SafetyWrapper(ABC):
    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def decide(self, user_prompt: str, model_output: str, history: list):
        pass