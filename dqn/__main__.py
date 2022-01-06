import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["CartPole-v0", "CartPole-v1"])
    parser.add_argument(
        "--evaluate_freq",
        type=int,
        default=25,
        help="How often to run evaluation.",
        nargs="?",
    )
    parser.add_argument(
        "--evaluation_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
        nargs="?",
    )