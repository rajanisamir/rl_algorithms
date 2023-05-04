import logging
import time
import keyboard
import os
import sys
import datetime
import argparse
from pathlib import Path

import gymnasium as gym

from dqn import DQN

logger = logging.getLogger("train")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model")

    # Environment
    parser.add_argument(
        "--env-name",
        type=str,
        default="LunarLander-v2",
        help="name of the gymnasium environment for training",
    )

    # Saving
    parser.add_argument(
        "--save-path",
        type=Path,
        default="save",
        help="path at which to save checkpoints, metrics, figures, etc.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="path from which to load a checkpoint",
    )

    # Model
    parser.add_argument(
        "--replay-memory-size",
        type=int,
        default=50_000,
        help="the size of the DQN replay memory buffer",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="the starting exploration rate of the policy network",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="the final exploration rate of the policy network",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=75_000,
        help="the number of steps across which to decay epsilon",
    )
    parser.add_argument(
        "--epsilon-eval",
        type=int,
        default=0.01,
        help="the value of epsilon for evaluation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="the learning rate for training the policy network",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="the batch size for training the DQN",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=250,
        help="the number of timesteps between DQN target network updates",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="the dimension of the hidden layer of the MLP policy network",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="the number of timesteps for which to train the RL agent",
    )

    return parser.parse_args()


class InteractiveEnvironment:
    def __init__(self, env_name):
        self.paused = False
        self.exited = False
        self.env = gym.make(env_name, render_mode="human")

        keyboard.on_press_key("enter", self.close_and_exit)
        keyboard.on_press_key("p", self.toggle_pause)

    def close_and_exit(self, _):
        self.exited = True
        self.paused = False

    def toggle_pause(self, _):
        if self.paused:
            logger.info("resuming environment. press [p] to pause")
            self.paused = False
        else:
            logger.info("pausing environment. press [p] to resume")
            self.paused = True

    def simulate(self):
        observation, info = self.env.reset()

        while True:
            if self.exited:
                logger.info("closing environment and exiting")
                self.env.close()
                sys.exit()

            if self.paused:
                time.sleep(0.5)

            while self.paused:
                if keyboard.is_pressed("n"):
                    logger.info(
                        "predicted maximum q-value:"
                        f" {dqn.predict_value(observation):.2f}"
                    )
                    break

            action = dqn.predict(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset()


if __name__ == "__main__":
    args = parse_arguments()

    current_time = datetime.datetime.now().strftime("%B-%d-%y_%I-%M-%S")
    args.save_path = os.path.join(args.save_path, current_time)
    os.makedirs(args.save_path, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.save_path, "logfile.log")),
        ],
    )

    logger.info(f"save path is {args.save_path}")
    logger.info(
        "arguments:\n\t" + "\n\t".join(f"{k}: {v}" for k, v in vars(args).items())
    )

    env = gym.make(args.env_name)
    logger.info(f"created environment {args.env_name}")

    dqn = DQN(
        env,
        replay_memory_size=args.replay_memory_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_eval=args.epsilon_eval,
        gamma=args.gamma,
        lr=args.lr,
        bs=args.bs,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        save_path=args.save_path,
    )
    logger.info("initialized dqn model")

    if args.checkpoint_path is not None:
        if not args.checkpoint_path.is_file():
            logger.error(f"checkpoint does not exist: {args.checkpoint_path}")
        dqn.load_checkpoint(args.checkpoint_path)

    logger.info("training dqn model")
    dqn.learn(total_timesteps=args.total_timesteps, verbose=True)
    env.close()

    checkpoint_path = dqn.save_checkpoint()

    dqn = DQN(
        env,
        replay_memory_size=args.replay_memory_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_eval=args.epsilon_eval,
        gamma=args.gamma,
        lr=args.lr,
        bs=args.bs,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        save_path=args.save_path,
        eval=True,
    )
    dqn.load_checkpoint(checkpoint_path)

    logger.info(f"running trained model in environment {args.env_name}")
    logger.info(f"press the [p] key to pause/resume at any time")
    logger.info(f"press the [enter] key to exit at any time")

    InteractiveEnvironment(args.env_name).simulate()
