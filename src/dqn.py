from collections import namedtuple, deque

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

class DQN:
    def __init__(self, env, capacity, epsilon_start, epsilon_end, gamma, lr):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.model = MLP(env)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.replay_memory = deque(maxlen=capacity)
        self.minibatch_size = 32

    def predict(self, observation):
        self.model.eval()
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        observation = torch.tensor(observation)
        q_values = self.model(observation)
        return torch.argmax(q_values).item()

    def train(self):
        """Sample and train on a random minibatch of transitions from replay memory."""
        self.model.train()
        if len(self.replay_memory) < self.minibatch_size:
            return
        sampled_indices = set(np.random.choice(len(self.replay_memory), self.minibatch_size, replace=False))
        minibatch = [transition for i, transition in enumerate(self.replay_memory) if i in sampled_indices]
        inputs = torch.tensor(np.array([transition.state for transition in minibatch]))
        labels = torch.tensor([transition.reward + (self.gamma * torch.max(self.model(torch.tensor(transition.next_state))) if transition.next_state is not None else 0) for transition in minibatch])
        actions = torch.tensor([transition.action for transition in minibatch])

        preds = torch.gather(self.model(inputs), 1, actions.unsqueeze(1)).flatten()
        loss = self.loss(preds, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, transition):
        if len(self.replay_memory) == self.replay_memory.maxlen:
            self.replay_memory.popleft()
        self.replay_memory.append(transition)
    
    def learn(self, total_timesteps, progress_bar=False):
        observation, info = self.env.reset()

        total_timesteps_iter = range(total_timesteps)
        if progress_bar:
            total_timesteps_iter = tqdm(total_timesteps_iter)

        for t in total_timesteps_iter:
            self.epsilon = self.epsilon_start - (t * (self.epsilon_start - self.epsilon_end) / total_timesteps)
            action = self.predict(observation)
            
            prev_observation = observation
            observation, reward, terminated, truncated, info = self.env.step(action)
            transition = Transition(prev_observation, action, reward, observation)
            self.store_transition(transition)
            
            self.train()

            if terminated or truncated:
                observation, info = self.env.reset()

class MLP(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_dim = np.prod(env.observation_space.shape)
        self.hidden_dim = 256
        self.output_dim = env.action_space.n

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer(out))
        out = self.output_layer(out)
        return out