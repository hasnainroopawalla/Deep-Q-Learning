import gym
import torch

from dqn.agents.cartpole.utils import preprocess
import torch.nn.functional as F

from dqn.agents.cartpole.model import DQN
from dqn.replay_memory import ReplayMemory
from dqn.agents.cartpole.config import CartPoleConfig
from dqn.agents.base_agent import BaseAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleAgent(BaseAgent):
    def __init__(self) -> None:
        # Initialize the agent configuration.
        self.cfg = CartPoleConfig()
        # Initialize the gym environment.
        self.env = gym.make(self.cfg.env)
        # Initialize the deep Q-network.
        self.dqn = DQN(cfg=self.cfg).to(device)
        # Create and initialize target Q-network.
        self.target_dqn = DQN(cfg=self.cfg).to(device)
        # Create replay memory.
        self.memory = ReplayMemory(self.cfg.train.memory_size)
        # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.cfg.train.lr)

    def train(self) -> None:
        # Keep track of best evaluation mean return achieved so far.
        best_mean_return = -float("Inf")

        for episode in range(self.cfg.train.episodes):
            done = False
            obs = preprocess(self.env.reset())
            steps = 0
            while not done:
                # TODO: Get action from DQN.
                action = self.dqn.act(obs)

                next_obs, reward, done, info = self.env.step(
                    action
                )  # Act in the true environment.

                # if not done:
                next_obs = preprocess(next_obs)  # Preprocess incoming observation.

                # TODO: Add the transition to the replay memory. Remember to convert everything to PyTorch tensors!
                self.memory.push(obs, action, next_obs, reward, done)

                # TODO: Run DQN.optimize() every cfg.train.frequency steps.
                if steps % self.cfg.train.frequency == 0:
                    self.optimize()

                # TODO: Update the target network every cfg.train.target_update_frequency steps.
                if steps % self.cfg.train.target_update_frequency == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())

                steps += 1
                obs = next_obs

            # Evaluate the current agent.
            if episode % self.cfg.evaluate.frequency == 0:
                mean_return = self.evaluate()

                print(f"Episode {episode}/{self.cfg.train.episodes}: {mean_return}")

                # Save current agent if it has the best performance so far.
                if mean_return >= best_mean_return:
                    best_mean_return = mean_return

                    print("Best performance so far! Saving model.")
                    torch.save(self.dqn, self.cfg.model_path)

            # Update epsilon here ?? reduce by 1% after an episode
            self.dqn.eps_start = max(self.dqn.eps_end, 0.99 * self.dqn.eps_start)

        # Close environment after training is completed.
        self.env.close()

    def optimize(self):
        """This function samples a batch from the replay buffer and optimizes the Q-network."""
        # If we don't have enough transitions stored yet, we don't train.
        if len(self.memory) < self.dqn.batch_size:
            return

        # TODO: Sample a batch from the replay memory and concatenate so that there are
        #       four tensors in total: observations, actions, next observations and rewards.
        #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
        #       Note that special care is needed for terminal transitions!

        sample = self.memory.sample(self.dqn.batch_size)
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

    def evaluate(self, render=False, verbose=False):
        """Runs {n_episodes} episodes to evaluate current policy."""
        total_return = 0
        for i in range(self.cfg.evaluate.episodes):
            obs = preprocess(self.env.reset()).unsqueeze(0)

            done = False
            episode_return = 0

            while not done:
                if render:
                    self.env.render()

                action = self.dqn.act(obs, exploit=True)

                obs, reward, done, info = self.env.step(action)
                obs = preprocess(obs).unsqueeze(0)

                episode_return += reward

            total_return += episode_return

            if verbose:
                print(
                    f"Finished episode {i+1} with a total return of {episode_return}."
                )

        return total_return / self.cfg.evaluate.episodes
