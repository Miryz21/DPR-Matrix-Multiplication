import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle

# Импорт датасета с тензорами
with open('./tensor_holder/tensors', 'rb') as f:
    multiply_tensors = pickle.load(f)


# Определение актора (Actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        action = self.action_bound * x
        return action


# Определение критика (Critic)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value


# Определение DDPG агента
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma  # коэффициент дисконтирования оценки
        self.tau = tau

        # Инициализация актора и критика
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        # Инициализация оптимизаторов
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Инициализация весов target-сетей
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    def train(self, replay_buffer, batch_size):
        # Получение мини-пакета из replay buffer
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # Обновление критика
        target_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic(next_state_batch, target_actions)
        target_q_values = reward_batch + self.gamma * (1 - done_batch) * target_q_values

        q_values = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Обновление актора
        actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Обновление весов target-сетей
        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer)


class Environment:
    def __init__(self, shape):
        self.shape = shape
        self.m = self.shape[0]
        self.n = self.shape[1]
        self.d = self.shape[2]
        self.state = None

    def reset(self):
        self.state = torch.clone(multiply_tensors[self.shape])
        return self.state

    def step(self, action):
        # Perform action on the environment
        # Update the state based on the action
        # Compute the reward and done flag
        print(f'multiply {torch.tensor(action[0])} and {torch.tensor(action[1])}')
        summ = torch.outer(torch.tensor(action[0]), torch.tensor(action[1]))
        for j in range(4):
            self.state[j] -= summ * np.array(action[2])

        # Example: Compute reward and done flag
        reward = torch.sum(self.state)
        done = torch.zeros(self.m * self.n, self.n * self.d, self.m * self.d)

        return self.state, reward, done


# Example usage
env_shape = (2, 2, 2)  # Shape of the environment tensor
env = Environment(env_shape)

# Reset the environment
state = env.reset()
print("Initial state:")
print(state)

# Take some actions in the environment
actions = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
for action in actions:
    state, reward, done = env.step(action)
    print("Action:", action)
    print("State:")
    print(state)
    print("Reward:", reward)
    print("Done:", done)
    print()

# Reset the environment
state = env.reset()
print("State after reset:")
print(state)
