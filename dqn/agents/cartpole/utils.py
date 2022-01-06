import torch

from dqn.agents.cartpole.config import CartPoleConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs):
    """Performs necessary observation preprocessing."""
    return torch.tensor(obs, device=device).float()


def load_model():
    return torch.load(CartPoleConfig().model_path, map_location=torch.device("cpu"))
