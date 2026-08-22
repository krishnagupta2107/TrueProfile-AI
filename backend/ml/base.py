from abc import ABC, abstractmethod

class BaseModelInterface(ABC):
    @abstractmethod
    def predict(self, features: dict) -> float:
        """
        Takes a dictionary of features and returns a risk score between 0.0 and 1.0.
        """
        pass
