import gym
import torch

import torch.nn.functional as F

from dqn.agents.cartpole.model import DQN
from dqn.replay_memory import ReplayMemory
from dqn.agents.cartpole.config import CartPoleConfig
from dqn.agents.base_agent import BaseAgent
from dqn.agents.cartpole.utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleAgent(BaseAgent):
    def __init__(self) -> None:
        # Initialize the agent configuration.
        self.cfg = CartPoleConfig()
        # Initialize the gym environment.
        self.env = gym.make(self.cfg.env)
        # Initialize the candidate deep Q-network.
        self.dqn = DQN(cfg=self.cfg).to(device)
        # Initialize target deep Q-network.
        self.target_dqn = DQN(cfg=self.cfg).to(device)
        # Create the replay memory.
        self.memory = ReplayMemory(self.cfg.train.memory_size)
        # Initialize optimizer used for training the DQN.
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.cfg.train.lr)

    def train(self) -> None:
        # Keep track of best evaluation mean return achieved so far.
        best_mean_return = -float("Inf")

        for episode in range(self.cfg.train.episodes):
            done = False
            obs = preprocess(self.env.reset())
            steps = 0
            while not done:
                # Get an action from the DQN.
                action = self.dqn.act(obs)

                # Act in the true environment.
                next_obs, reward, done, info = self.env.step(action)

                # Preprocess the incoming observation.
                next_obs = preprocess(next_obs)

                # Add the transition to the replay memory.
                self.memory.push(obs, action, next_obs, reward, done)

                # Optimize the DQN every cfg.train.frequency steps.
                if steps % self.cfg.train.frequency == 0:
                    self.optimize()

                # Update the target DQN with the candidate DQN every cfg.train.target_update_frequency steps.
                if steps % self.cfg.train.target_update_frequency == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())

                steps += 1
                obs = next_obs

            # Evaluate the current agent.
            if episode % self.cfg.evaluate.frequency == 0:
                mean_return = self.evaluate()

                print(
                    f"Episode {episode}/{self.cfg.train.episodes}, Mean Return: {mean_return}"
                )

                # Save current agent if it has the best performance.
                if mean_return >= best_mean_return:
                    best_mean_return = mean_return

                    print("Best performance so far, Saving model.")
                    torch.save(self.dqn, self.cfg.model_path)

            # Update the epsilon value.
            self.dqn.eps_start = max(self.dqn.eps_end, 0.99 * self.dqn.eps_start)

        self.env.close()

    def optimize(self):
        """This function samples a batch from the replay buffer and optimizes the Q-network."""
        # Check if enough transitions are available before optimizing.
        if len(self.memory) < self.dqn.batch_size:
            return

        # TODO: Sample a batch from the replay memory and concatenate so that there are
        #       four tensors in total: observations, actions, next observations and rewards.
        #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
        #       Note that special care is needed for terminal transitions!

        # Sample a batch from the replay memory
        sample = self.memory.sample(self.dqn.batch_size)
        obs = torch.stack(sample.obs)
        next_obs = torch.stack(sample.next_obs)
        actions = torch.Tensor(sample.actions).long().unsqueeze(1)
        rewards = torch.Tensor(sample.rewards).long().unsqueeze(1)
        dones = torch.Tensor(sample.dones).long().unsqueeze(1)

        # TODO: Compute the current estimates of the Q-values for each state-action
        #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
        #       corresponding to the chosen actions.
        q_values_expected = self.dqn(obs).gather(1, actions)

        next_q_values = self.target_dqn(next_obs).detach().max(1)[0].unsqueeze(1)

        # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
        q_value_targets = rewards + (self.dqn.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values_expected, q_value_targets)

        # Perform gradient descent.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, render=False):
        """Runs {n_episodes} episodes to evaluate current policy."""
        total_return = 0
        for i in range(self.cfg.evaluate.episodes):
            obs = preprocess(self.env.reset()).unsqueeze(0)

            done = False
            episode_return = 0

            while not done:
                if render:
                    self.env.render()

                action = self.dqn.act(obs)

                obs, reward, done, info = self.env.step(action)
                obs = preprocess(obs).unsqueeze(0)

                episode_return += reward

            total_return += episode_return

        return total_return / self.cfg.evaluate.episodes

    def simulate(self):
        self.dqn = torch.load(self.cfg.model_path, map_location=device)
        self.cfg.evaluate.episodes = 1
        mean_return = self.evaluate(render=True)
        print(f"Simulation Complete. Mean Return: {mean_return}")
