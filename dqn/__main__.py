import argparse
from dqn.config import agent_map


def run(args: argparse.Namespace) -> None:
    agent = agent_map[args.agent]
    if args.mode == "train":
        agent.train()
    elif args.mode == "simulate":
        agent.simulate()


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
        help="Train or Simulate (render) the RL agent.",
        choices=["train", "simulate"],
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
