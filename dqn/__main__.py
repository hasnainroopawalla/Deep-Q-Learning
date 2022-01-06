import argparse
from dqn.config import agent_map
from dqn.agents.cartpole.utils import load_model


def run(args: argparse.Namespace):
    print(args)
    agent = agent_map[args.agent]()
    if args.mode == "train":
        agent.train()
    elif args.mode == "evaluate":
        dqn = load_model()
        mean_return = agent.evaluate(render=True, verbose=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        help="The Atari Game.",
        choices=["cartpole", "pong"],
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Train or Evaluate (render) the RL agent.",
        choices=["train", "evaluate"],
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
