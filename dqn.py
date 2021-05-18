import network

import gym
import torch
from torch.nn import functional as F
import numpy as np
import random
import tqdm

from copy import deepcopy
from collections import deque

import matplotlib.pyplot as plt

class DQNAgent():
    def __init__(self, 
            envname, 
            network_ = network.MLP, 
            optimizer_ = torch.optim.Adam, 
            epsilon_init = 1.0, 
            epsilon_final = 0.1, 
            epsilon_decay = 0.9999999,
            batch_size = 4,
            replay_buffer_size = 10000,
            update_freq = 1,
            tau = 0.01):

        self.envname = envname
        self.env = gym.make(envname)
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.n

        self.history = []

        self.DISCOUNT = 0.99
        self.exploration = True

        self.Q_dynamic = network.MLP(self.state_shape, self.action_shape, 256, depth=3)
        self.Q_static = deepcopy(self.Q_dynamic)
        self.optimizer = torch.optim.Adam(self.Q_dynamic.parameters(), lr=1e-5)

        self.batch_size = batch_size
        self.update_freq = update_freq 
        self.tau = tau

        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.experiences = deque(maxlen = replay_buffer_size)

    def act(self, state):
        if self.exploration and random.random() < self.EPSILON:
            return torch.LongTensor([self.env.action_space.sample()]).view(1,1)
        else:
            return self.Q_dynamic(state).argmax().view(1, 1)

    def step(self, state):

        action = self.act(state)
        nextState, reward, done, info = self.env.step(action.item())

        nextState = torch.from_numpy(nextState).unsqueeze(0).float()

        return (state, action, reward, nextState), (done, info)

    def train(self):
        batch = random.sample(self.experiences, self.batch_size) 
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        batch_states = torch.cat(batch_states)
        batch_actions = torch.cat(batch_actions)
        batch_rewards = torch.FloatTensor(batch_rewards)
        batch_next_states = torch.cat(batch_next_states)

        q_values = self.Q_dynamic(batch_states).gather(1, batch_actions).reshape(-1)
        with torch.no_grad():
            targets = batch_rewards + self.DISCOUNT * self.Q_static(batch_next_states).max(1)[0]

        loss = F.mse_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.update_freq == 0:
            for static_param, dynamic_param in zip(self.Q_static.parameters(), self.Q_dynamic.parameters()):
                static_param.data = dynamic_param * self.tau + static_param * 1-self.tau
            # self.Q_static = deepcopy(self.Q_dynamic)


    def learn(self, num_simulation = 0, max_steps = 10000):
        best_model, best_score = None, None
        lookback = -15

        self.total_steps = 0

        self.EPSILON = self.epsilon_init

        pbar = tqdm.trange(num_simulation)
        for i in pbar:

            state = torch.from_numpy(self.env.reset()).unsqueeze(0).float()
            done = False
            step_count = total_reward = 0

            while not done and step_count < max_steps:

                (state, action, reward, nextState), (done, info) = self.step(state)

                self.experiences.append((state, action,  reward, nextState))

                state = nextState
                step_count += 1
                self.total_steps += 1
                total_reward += reward

                if len(self.experiences) > self.batch_size:
                    self.train()

                self.EPSILON = max(self.epsilon_final, self.EPSILON * self.epsilon_decay)

            self.history.append(total_reward)

            if best_score is None or sum(self.history[lookback:]) > best_score:
                best_model = deepcopy(self.Q_dynamic)
                best_score = sum(self.history[lookback:])

            pbar.set_description(f'Last Episode length: {step_count:3d}')

            plt.figure(2, clear=True)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.plot(np.asarray(self.history))
            if len(self.history) > -lookback:
                plt.plot(np.convolve(np.asarray(self.history), np.ones(-lookback), 'valid') / -lookback)
            plt.pause(0.001)

        self.Q_dynamic = best_model
        return np.asarray(self.history)

    def test(self, num_simulation = 10000, render=False):
        self.exploration = False
        history = []
        for simulation in tqdm.trange(num_simulation):
            state = self.env.reset()
            done = False
            reward_to_go = 0

            while not done:
                state = torch.from_numpy(state).unsqueeze(0).float()
                action = self.act(state).item()
                state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()
                reward_to_go += reward

            history.append(reward_to_go)
        return np.asarray(history)

    def __str__(self):
        return 'DQN'

    def save(self):
        torch.save(self.Q_static.state_dict(), f'models/{self.envname}_{str(self)}_qstatic.pickle')
        torch.save(self.Q_dynamic.state_dict(), f'models/{self.envname}_{str(self)}_qdynamic.pickle')

    def load(self):
        self.Q_static.load_state_dict(torch.load('models/' + self.envname + '_' + str(self) + '_qstatic.pickle'))
        self.Q_dynamic.load_state_dict(torch.load('models/' + self.envname + '_' + str(self) + '_qdynamic.pickle'))
