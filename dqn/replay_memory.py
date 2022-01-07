import random
from dataclasses import dataclass
from typing import Tuple
from torch import Tensor


@dataclass
class SampleBatch:
    obs: Tuple[Tensor]
    actions: Tuple[int]
    next_obs: Tuple[Tensor]
    rewards: Tuple[float]
    dones: Tuple[bool]


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> SampleBatch:
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs torch.Tensor, action, next_obs, reward)
        """
        sample = tuple(zip(*random.sample(self.memory, batch_size)))
        return SampleBatch(
            obs=sample[0],
            actions=sample[1],
            next_obs=sample[2],
            rewards=sample[3],
            dones=sample[4],
        )
