import random
from dataclasses import dataclass, field
from typing import List
from torch import Tensor


@dataclass
class Sample:
    """A sample class stored in the replay memory.
    """

    obs: Tensor
    action: int
    next_obs: Tensor
    reward: float
    done: bool


@dataclass
class Batch:
    """A batch sampled from the replay memory.
    """

    obs: List[Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    next_obs: List[Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


class ReplayMemory:
    def __init__(self, capacity: int):
        """Initialize the Replay Memory.

        Args:
            capacity (int): The maximum size of the buffer determined by config.memory_size.
        """
        self.capacity = capacity
        self.memory: List[Sample] = []

    def push(self, sample: Sample):
        """Adds a sample to the memory.

        Args:
            sample (Sample): The sample (observation, action, next observation, reward and a flag if the episode is done or not) to be added to the memory.

        Raises:
            OverflowError: This error is raised if the maximum size of the Replay Memory is exceeded.
        """
        if len(self.memory) > self.capacity:
            raise OverflowError(f"Max Replay Memory size exceeded {self.capacity}")
        self.memory.append(sample)

    def sample(self, batch_size: int) -> Batch:
        """Creates a batch of random samples from the memory.

        Args:
            batch_size (int): The number of samples in each batch.

        Returns:
            Batch: A batch of samples from the memory.
        """
        batch = Batch()
        for sample in random.sample(self.memory, batch_size):
            batch.obs.append(sample.obs)
            batch.actions.append(sample.action)
            batch.next_obs.append(sample.next_obs)
            batch.rewards.append(sample.reward)
            batch.dones.append(sample.done)
        return batch
