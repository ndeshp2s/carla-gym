import os
import torch
import torch.optim as optim

from experiments.config import Config
from rl_agents.DQN.replay_buffer import ReplayBuffer
from neural_networks.fc64_fc64 import NeuralNetwork

class DDQNAgent:
    def __init__(self, config: Config):

        # Parameter initialization
        self.state_size = config.state_dim
        self.action_size = config.action_dim

        self.hyperparameters = config.hyperparameters

        # self.step_number = 0

        if config.use_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        else:
            self.device = "cpu"

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.hyperparameters["learning_rate"])

        # Initialise replay memory
        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"])

    def add(self, state, reward, action, next_state, done):
        self.memory.add_experience(state, reward, action, next_state, done)

    def learn(self, experiences = None):

        if experiences is None:
            states, actions, rewards, next_states, dones  = self.memory.sample()

        else:
            v = experiences


    # def step(self):

    # def learn(self, experiences = None):
    #     if experiences is None:
    #         states, actions, rewards, next_states, dones = self.memory.sample()

    #     else:
    #         states, actions, rewards, next_states, dones = experiences

    def save_model(self, directory = None, tag = ''):
        if directory is None:
            return

        if not os.path.isdir(directory):
            os.makedirs(directory)

        torch.save(self.local_network.state_dict(), '%s/model%s.pkl' % (directory, ('-' + tag)))

        

