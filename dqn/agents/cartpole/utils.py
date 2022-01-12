from typing import Tuple
import torch
from torch import Tensor

from dqn.replay_memory import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_observation(obs):
    """Performs necessary observation preprocessing."""
    return torch.tensor(obs, device=device).float()


def preprocess_sampled_batch(
    batch: Batch
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pre-processes a batch of samples from the memory.

    Args:
        batch (Batch): A batch of raw samples.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A batch of pre-processed samples;
    """
    obs = torch.stack(batch.obs)
    next_obs = torch.stack(batch.next_obs)
    actions = torch.Tensor(batch.actions).long().unsqueeze(1)
    rewards = torch.Tensor(batch.rewards).long().unsqueeze(1)
    dones = torch.Tensor(batch.dones).long().unsqueeze(1)
    return obs, next_obs, actions, rewards, dones
