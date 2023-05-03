import logging
import os
import sys
import datetime

import gymnasium as gym

from dqn import DQN

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("environment")

env_name = "LunarLander-v2"
time = datetime.datetime.now().strftime("%B-%d-%y_%I-%M-%S")
save_path = os.path.join("save", time)

logger.info(f"setting save path to {save_path}")

logger.info(f"creating environment {env_name}")
env = gym.make(env_name)

logger.info("initializing dqn model")
dqn = DQN(
    env,
    capacity=100_000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=65_000,
    gamma=0.99,
    lr=1e-4,
    bs=32,
    tau=1000,
    hidden_dim=128,
    save_path=save_path,
)
logger.info("training dqn model")
dqn.learn(total_timesteps=2_000, verbose=True)

logger.info("saving checkpoint")
dqn.save_checkpoint()

logger.info(f"running trained model in environment {env_name}")
env = gym.make(env_name, render_mode="human")
observation, info = env.reset()

while True:
    action = dqn.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
