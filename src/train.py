from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from evaluate import evaluate_HIV

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
"""class ProjectAgent1:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 24
        
        model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, n_action)).to(device)
        
        self.gamma = 0.95
        self.batch_size = 20
        self.nb_actions = n_action
        self.memory = ReplayBuffer(1000000, device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 1000
        self.epsilon_delay = 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        episode_step = 0
        while episode < max_episode:
            if step % 50 == 0:
                print(episode, step)
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            episode_step += 1
            if done or episode_step >= env._max_episode_steps:
                episode_step = 0
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:10.2e}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
            
            if step % 1000 == 0:
                torch.save(self.model.state_dict(), f"model_step_{step}.pth")
                

        return episode_return
    
    def act(self, observation, use_random=False):
        state = torch.Tensor(observation).unsqueeze(0)
        if use_random:
            return np.random.randint(self.nb_actions)
        action = greedy_action(self.model, state)
        return action

    def save(self, path):
        pass

    def load(self):
        self.model.load_state_dict(torch.load("model_step_9000.pth"))
        pass"""
    

class ProjectAgent:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        state_dim = env.observation_space.shape[0]
        state_dim = self.state_to_smart_state(1+np.zeros(state_dim)).shape[0]
        n_action = env.action_space.n
        nb_neurons = 512
        
        model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, n_action)).to(device)
        
        self.state_dim = state_dim
        self.nb_actions = n_action
        self.gamma = 0.99
        self.batch_size = 1024
        buffer_size = 1000000
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = 1
        self.epsilon_min = 0.005
        self.epsilon_stop = 10000
        self.epsilon_delay = 400
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps =  2
        self.update_target_strategy = 'replace'
        self.update_target_freq = 600
        self.update_target_tau = 0.001
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def state_to_smart_state(self, state):
        #maxs = np.array([800000, 50000, 1400, 80, 250000, 400])
        return state
        #return np.array([state[1]/(state[1]+state[0]), state[3]/(state[3]+state[2])])#, state[4]/250000, state[5]/600])
            
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        smart_state = self.state_to_smart_state(state)
        epsilon = self.epsilon_max
        step = 0
        episode_step = 0
        states_taken = np.zeros((env._max_episode_steps, self.state_dim))
        while episode < max_episode:
            states_taken[episode_step] = smart_state
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                smart_state = self.state_to_smart_state(state)
                action = greedy_action(self.model, smart_state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            next_smart_state = self.state_to_smart_state(next_state)
            self.memory.append(smart_state, action, reward, next_smart_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
                
            # update target network if needed
            if step % self.update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
                    
            # next transition
            episode_step += 1
            step += 1
            if done or trunc: # or episode_step >= env._max_episode_steps:
                # if episode % 10 == 0:
                #     for k in range(self.state_dim):
                #         plt.figure(figsize=(15, 5))
                #         plt.hist(states_taken[:episode_step, k])
                #         plt.savefig(f"histogram_{episode}_{k}.png")
                #         plt.show()
                episode_step = 0
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.4f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:10.2e}'.format(episode_cum_reward),
                      sep='')
                # if episode % 5 == 0:
                #     print("Agent score ", '{:10.2e}'.format(evaluate_HIV(agent=self, nb_episode=5)))
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
            
            if step % 5000 == 0:
                torch.save(self.model.state_dict(), f"model_target_step_{step}.pth")
            
        plt.figure(figsize=(15, 5))
        plt.semilogy(episode_return)
        plt.savefig("episode_return_0001.png")
        plt.show()
        
        return episode_return
    
    def act(self, observation, use_random=False):
        state = torch.Tensor(observation).unsqueeze(0)
        if use_random:
            return np.random.randint(self.nb_actions)
        action = greedy_action(self.model, state)
        return action

    def save(self, path):
        pass

    def load(self):
        self.model.load_state_dict(torch.load("dqn_target_lr0001_eps0001_step_40000.pth", map_location=self.device, weights_only=True))
        pass


if __name__ == "__main__":
    print("Starting...")
    # Train agent
    agent = ProjectAgent()
    agent.load()
    scores = agent.train(env, 200)
    print(scores)