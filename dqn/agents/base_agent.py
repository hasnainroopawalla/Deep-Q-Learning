from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass
