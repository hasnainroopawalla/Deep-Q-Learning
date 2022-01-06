import random

import gym
from gym.core import ObservationWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

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

    def act(self, observation, exploit=False):
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


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!

    sample = memory.sample(dqn.batch_size)
    obs = sample[0]
    actions = torch.Tensor(sample[1]).long().unsqueeze(1)
    next_obs = sample[2]
    rewards = torch.Tensor(sample[3]).long().unsqueeze(1)
    dones = torch.Tensor(sample[4]).long().unsqueeze(1)

    obs = torch.stack(obs)
    next_obs = torch.stack(next_obs)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values_expected = dqn(obs).gather(1, actions)

    next_q_values = target_dqn(next_obs).detach().max(1)[0].unsqueeze(1)

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    q_value_targets = rewards + (dqn.gamma * next_q_values * (1 - dones))

    loss = F.mse_loss(q_values_expected, q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
