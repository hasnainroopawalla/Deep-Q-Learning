import argparse
from dqn.config import env_agent_map
from dqn.env.cartpole.utils import load_model


def run(args: argparse.Namespace):
    print(args)
    agent = env_agent_map[args.env]()
    if args.mode == "train":
        agent.train()
    elif args.mode == "evaluate":
        dqn = load_model()
        mean_return = agent.evaluate(render=True, verbose=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        help="The Atari game evironment.",
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
