import gymnasium as gym

from dqn import DQN

env = gym.make("LunarLander-v2")

dqn = DQN(env, capacity=10_000, epsilon_start=1, epsilon_end=0.1, gamma=0.99, lr=0.0001)
dqn.learn(total_timesteps=300_000, progress_bar=True)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

while True:
    action = dqn.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()