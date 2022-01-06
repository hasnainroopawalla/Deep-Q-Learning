from typing import Dict
from dqn.env.base_env import BaseEnv
from dqn.env.cartpole import CartPole

env_map: Dict[str, BaseEnv] = {"cartpole": CartPole}
