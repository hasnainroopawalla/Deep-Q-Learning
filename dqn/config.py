from typing import Dict
from dqn.agents.base_agent import BaseAgent
from dqn.agents.cartpole import CartPoleAgent

agent_map: Dict[str, BaseAgent] = {"cartpole": CartPoleAgent}
