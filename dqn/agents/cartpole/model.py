import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from dqn.agents.cartpole.config import CartPoleConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, cfg: CartPoleConfig):
        super(DQN, self).__init__()

        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.eps_start = cfg.train.eps_start
        self.eps_end = cfg.train.eps_end
        self.anneal_length = cfg.train.anneal_length
        self.num_actions = cfg.train.num_actions

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.num_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass of the network.

        Args:
            x (Tensor): The input to the network.

        Returns:
            Tensor: The output of the final layer of the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def act(self, observation: Tensor) -> int:
        """Selects an action with an epsilon-greedy exploration strategy.

        Args:
            observation (Tensor): The current observation.

        Returns:
            int: The action taken by the DQN based on the observation.
        """
        prediction = self(observation.squeeze())
        if np.random.uniform(low=0.0, high=1.0) <= self.eps_start:
            action = np.random.randint(low=0.0, high=self.num_actions)  # Random action
        else:
            action = torch.argmax(prediction).item()
        return action  # 0 Push cart to the left. 1 Push cart to the right.
