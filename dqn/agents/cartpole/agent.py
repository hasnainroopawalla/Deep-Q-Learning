import gym
import torch

import torch.nn.functional as F

from dqn.agents.cartpole.model import DQN
from dqn.replay_memory import ReplayMemory, Sample
from dqn.agents.cartpole.config import CartPoleConfig
from dqn.agents.base_agent import BaseAgent
from dqn.agents.cartpole.utils import preprocess


class CartPoleAgent(BaseAgent):
    """The CartPole agent.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the agent configuration.
        self.cfg = CartPoleConfig()
        # Initialize the gym environment.
        self.env = gym.make(self.cfg.env)
        # Initialize the candidate deep Q-network.
        self.dqn = DQN(cfg=self.cfg).to(self.device)
        # Initialize target deep Q-network.
        self.target_dqn = DQN(cfg=self.cfg).to(self.device)
        # Create the replay memory.
        self.memory = ReplayMemory(self.cfg.train.memory_size)
        # Initialize optimizer used for training the DQN.
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.cfg.train.lr)

    def train(self) -> None:
        # Keep track of best evaluation mean return achieved so far.
        best_mean_return = float("-inf")

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
                sample = Sample(obs, action, next_obs, reward, done)
                self.memory.push(sample)

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
        # Check if enough transitions are availabl in the replay buffer before optimizing.
        if len(self.memory) < self.dqn.batch_size:
            return

        # Sample a batch from the replay memory.
        batch = self.memory.sample(self.dqn.batch_size)
        obs = torch.stack(batch.obs)
        next_obs = torch.stack(batch.next_obs)
        actions = torch.Tensor(batch.actions).long().unsqueeze(1)
        rewards = torch.Tensor(batch.rewards).long().unsqueeze(1)
        dones = torch.Tensor(batch.dones).long().unsqueeze(1)

        # TODO: Compute the current estimates of the Q-values for each state-action
        #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
        #       corresponding to the chosen actions.
        q_values_expected = self.dqn(obs).gather(1, actions)

        next_q_values = self.target_dqn(next_obs).detach().max(1)[0].unsqueeze(1)

        # Compute the Q-value targets ONLY for non-terminal transitions
        # If it is a terminal transition, (1 - dones) will evaluate to 0.
        q_value_targets = rewards + (self.dqn.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values_expected, q_value_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, render: bool = False) -> float:
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

    def simulate(self) -> None:
        self.dqn = torch.load(self.cfg.model_path, map_location=self.device)
        self.cfg.evaluate.episodes = 1
        mean_return = self.evaluate(render=True)
        print(f"Simulation Complete. Mean Return: {mean_return}")
