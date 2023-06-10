import torch
import torch.nn as nn
import math
import torch.optim as optim
from collections import namedtuple
import random


class DQNAgent(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN():
    def __init__(self, input_shape, num_actions, gamma, epsilon_start, epsilon_end, epsilon_decay, lr, capacity,
                 batch_size):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size

        self.target_net = DQNAgent(input_shape, num_actions)
        self.policy_net = DQNAgent(input_shape, num_actions)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(self.capacity)

        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

