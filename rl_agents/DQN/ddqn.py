import os
import numpy as np
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
        self.criterion = torch.nn.MSELoss()

        # Initialise replay memory
        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"])

    def add(self, state, reward, action, next_state, done):
        self.memory.add_experience(state, reward, action, next_state, done)

    def learn(self, batch_size = 32, experiences = None, step = 0):

        if experiences is None:
            #states, actions, rewards, next_states, dones  = self.memory.sample(batch_size)
            experiences  = self.memory.sample(batch_size)

        # else:
        #     states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)


        # Get the q values for all actions from local network
        q_predicted_all = self.local_network(states)
        # Get teh q value corresponding to the action executed
        q_predicted = q_predicted_all.gather(1, actions.long())
        # Get q values for all the actions of next state
        q_next_predicted_all = self.local_network(next_states)
        # Find the index of action with maximum q values (from next state)
        max_action_index = q_next_predicted_all.detach().argmax(1)

        # get q values for the actions of next state from target netwrok
        q_next_target = self.target_network(next_states)
        # get q value of action with same index as that of the action with maximum q values (from local network)
        q_max_next = q_next_target.gather(1, max_action_index.unsqueeze(1))
        # Find target q value using Bellmann's equation
        q_target = rewards + (self.hyperparameters["discount_rate"] * q_max_next * (1 - dones))


        loss = self.criterion(q_predicted, q_target)

        # make previous grad zero
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # update params
        self.optimizer.step()

        if step % self.hyperparameters["update_every_n_steps"] == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())

        return loss.item()



    def save_model(self, directory = None, tag = ''):
        if directory is None:
            return

        if not os.path.isdir(directory):
            os.makedirs(directory)

        torch.save(self.local_network.state_dict(), '%s/model%s.pkl' % (directory, ('-' + tag)))

    def pick_action(self, state, epsilon):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Query the network
        action_values = self.local_network.forward(state_tensor)

        if np.random.uniform() > epsilon:
            action = action_values.max(1)[1].item()

        else:
            action = np.random.randint(0, action_values.shape[1])

        return action



        

