from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, render: bool = False) -> None:
        pass

    @abstractmethod
    def simulate(self):
        pass
