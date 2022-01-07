import torch
import torch.nn as nn
import numpy as np

from dqn.agents.cartpole.config import CartPoleConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, cfg: CartPoleConfig):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.eps_start = cfg.train.eps_start
        self.eps_end = cfg.train.eps_end
        self.anneal_length = cfg.train.anneal_length
        self.n_actions = cfg.train.n_actions

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def act(self, observation):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        prediction = self(observation.squeeze())
        if np.random.uniform(low=0.0, high=1.0) <= self.eps_start:
            action = np.random.randint(low=0.0, high=self.n_actions)  # Random action
        else:
            action = torch.argmax(prediction).item()
        return action  # 0 Push cart to the left. 1 Push cart to the right.
