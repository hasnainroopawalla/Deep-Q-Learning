import argparse
from dqn.config import env_map


def run(args: argparse.Namespace):
    env = env_map[args.env]
    print(env)


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

    # parser.add_argument(
    #     "--evaluate_freq",
    #     type=int,
    #     default=25,
    #     help="How often to run evaluation.",
    #     nargs="?",
    # )

    # parser.add_argument(
    #     "--evaluation_episodes",
    #     type=int,
    #     default=5,
    #     help="Number of evaluation episodes.",
    #     nargs="?",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
