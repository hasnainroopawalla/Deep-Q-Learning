import random
from dataclasses import dataclass, field
from typing import List
from torch import Tensor


@dataclass
class SampleBatch:
    obs: List[Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    next_obs: List[Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


@dataclass
class Sample:
    obs: Tensor
    action: int
    next_obs: Tensor
    reward: float
    done: bool


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, sample: Sample):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> SampleBatch:
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs torch.Tensor, action, next_obs, reward)
        """
        batch = SampleBatch()
        for sample in random.sample(self.memory, batch_size):
            batch.obs.append(sample.obs)
            batch.actions.append(sample.action)
            batch.next_obs.append(sample.next_obs)
            batch.rewards.append(sample.reward)
            batch.dones.append(sample.done)
        return batch
