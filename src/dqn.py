import time
import logging
import random
import os
import copy
from collections import namedtuple, deque

import torch
from torch import nn
import numpy as np
from alive_progress import alive_bar

from metrics import MetricTracker

logger = logging.getLogger(__name__)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class DQN:
    def __init__(
        self,
        env,
        replay_memory_size,
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,
        epsilon_eval,
        gamma,
        lr,
        bs,
        tau,
        hidden_dim,
        save_path,
        eval=False,
    ):
        self.env = env
        self.save_path = save_path

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma
        self.bs = bs
        self.tau = tau

        self.replay_memory = deque(maxlen=replay_memory_size)

        self.policy_network = MLP(env, hidden_dim)
        self.target_network = MLP(env, hidden_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        policy_network_str = "\n\t" + str(self.policy_network).replace("\n", "\n\t")
        logger.info(f"policy network looks like: {policy_network_str}")

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.metric_tracker = MetricTracker(save_path)

        self.timestep = 0

        if eval:
            self.epsilon = epsilon_eval

    @torch.no_grad()
    def predict(self, observation):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.policy_network(torch.tensor(observation, dtype=torch.float32))
        return torch.argmax(q_values).item()

    @torch.no_grad()
    def predict_value(self, observation):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.policy_network(torch.tensor(observation, dtype=torch.float32))
        return torch.max(q_values).item()

    def train(self):
        """Sample and train on a random minibatch of transitions from replay memory."""
        if len(self.replay_memory) < self.bs:
            return
        minibatch = random.sample(self.replay_memory, self.bs)
        inputs = torch.tensor(
            np.array([transition.state for transition in minibatch]),
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                transition.reward
                + (
                    self.gamma
                    * torch.max(
                        self.target_network(
                            torch.tensor(transition.next_state, dtype=torch.float32)
                        )
                    )
                    if transition.next_state is not None
                    else 0
                )
                for transition in minibatch
            ],
            dtype=torch.float32,
        )
        actions = torch.tensor([transition.action for transition in minibatch])

        preds = torch.gather(
            self.policy_network(inputs), 1, actions.unsqueeze(1)
        ).flatten()
        loss = self.criterion(preds, labels)
        self.metric_tracker.update_metric("loss", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, transition):
        self.replay_memory.append(transition)

    def learn(self, total_timesteps, verbose=True, log_freq_time=60):
        observation, info = self.env.reset()

        episode_number = 0
        episode_timesteps = 0
        episode_reward = 0

        start_time = last_logging = time.time()
        with alive_bar(total_timesteps - self.timestep, enrich_print=False) as bar:
            for t in range(self.timestep, total_timesteps):
                self.timestep = t
                current_time = time.time()
                if verbose and current_time - last_logging > log_freq_time:
                    stats = {
                        "time": int(current_time - start_time),
                        "episode": episode_number,
                        "timestep": t,
                    }
                    self.metric_tracker.print_metrics(stats)
                    last_logging = current_time

                if t % self.tau == 0:
                    self.target_network.load_state_dict(
                        self.policy_network.state_dict()
                    )

                self.epsilon = self.epsilon_start + (
                    self.epsilon_end - self.epsilon_start
                ) * min(1, t / self.epsilon_decay_steps)
                action = self.predict(observation)

                prev_observation = copy.deepcopy(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_timesteps += 1

                if terminated or truncated:
                    transition = Transition(prev_observation, action, reward, None)
                else:
                    transition = Transition(
                        prev_observation, action, reward, observation
                    )

                self.store_transition(transition)
                self.train()

                if terminated or truncated:
                    self.metric_tracker.update_metric("reward", episode_reward)
                    self.metric_tracker.update_metric("ep_length", episode_timesteps)
                    self.metric_tracker.update_metric("epsilon", self.epsilon)

                    episode_number += 1
                    episode_timesteps = 0
                    episode_reward = 0

                    observation, _ = self.env.reset()

                bar()

        self.metric_tracker.save_metrics()

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"loading from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.policy_network.load_state_dict(checkpoint["model"])
        self.target_network.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.replay_memory = checkpoint["replay_memory"]
        self.timestep = checkpoint["timestep"]

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        checkpoint = {
            "model": self.policy_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_memory": self.replay_memory,
            "timestep": self.timestep + 1,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"saved checkpoint at {checkpoint_path}")
        return checkpoint_path


class MLP(nn.Module):
    def __init__(self, env, hidden_dim):
        super().__init__()
        input_dim = np.prod(env.observation_space.shape)
        output_dim = env.action_space.n

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = nn.functional.relu(self.input_layer(x))
        out = nn.functional.relu(self.hidden_layer(out))
        out = self.output_layer(out)
        return out
