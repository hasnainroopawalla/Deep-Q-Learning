from typing import Dict
from dqn.env.base_agent import BaseAgent
from dqn.env.cartpole import CartPoleAgent

env_agent_map: Dict[str, BaseAgent] = {"cartpole": CartPoleAgent}
