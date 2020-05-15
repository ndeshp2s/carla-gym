import os
import numpy as np
import torch
import torch.optim as optim

from experiments.config import Config
from rl_agents.DQN.replay_buffer import ReplayBuffer
from neural_networks.cnn_fc import NeuralNetwork

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
        self.memory.add_experience(state = state, reward = reward, action = action, next_state = next_state, done = done)

    def learn(self, batch_size = 32, experiences = None, step = 0):

        batch  = self.memory.sample(batch_size = batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []

        for e in batch:                
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)


        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

        # Get the q values for all actions from local network
        q_predicted_all = self.local_network.forward(states, batch_size = batch_size)
        #Get the q value corresponding to the action executed
        q_predicted = q_predicted_all.gather(dim = 1, index = actions.unsqueeze(dim = 1)).squeeze(dim = 1)
        # Get q values for all the actions of next state
        q_next_predicted_all = self.local_network.forward(next_states, batch_size = batch_size)
        
        # get q values for the actions of next state from target netwrok
        q_next_target_all = self.target_network.forward(next_states, batch_size = batch_size)
        # get q value of action with same index as that of the action with maximum q values (from local network)
        q_next_target = q_next_target_all.gather(1, q_next_predicted_all.max(1)[1].unsqueeze(1)).squeeze(1)
        # Find target q value using Bellmann's equation
        q_target = rewards + (self.hyperparameters["discount_rate"] * q_next_target)


        # if experiences is None:
        #     #states, actions, rewards, next_states, dones  = self.memory.sample(batch_size)
        #     experiences  = self.memory.sample(batch_size)

        # # else:
        # #     states, actions, rewards, next_states, dones = experiences

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)


        # # Get the q values for all actions from local network
        # q_predicted_all = self.local_network(states)
        # # Get teh q value corresponding to the action executed
        # q_predicted = q_predicted_all.gather(1, actions.long())
        # # Get q values for all the actions of next state
        # q_next_predicted_all = self.local_network(next_states)
        # # Find the index of action with maximum q values (from next state)
        # max_action_index = q_next_predicted_all.detach().argmax(1)

        # # get q values for the actions of next state from target netwrok
        # q_next_target = self.target_network(next_states)
        # # get q value of action with same index as that of the action with maximum q values (from local network)
        # q_max_next = q_next_target.gather(1, max_action_index.unsqueeze(1))
        # # Find target q value using Bellmann's equation
        # q_target = rewards + (self.hyperparameters["discount_rate"] * q_max_next * (1 - dones))


        loss = self.criterion(q_predicted, q_target)

        # make previous grad zero
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # Gradient clipping
        for param in self.local_network.parameters():
            param.grad.data.clamp_(-1, 1)

        # update params
        self.optimizer.step()

        if step % self.hyperparameters["update_every_n_steps"] == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())
        #self.soft_update(self.local_network, self.target_network, self.hyperparameters["tau"])

        return loss.item()


    def soft_update(self, local_network, target_network, tau):

        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1 - tau)*target_param.data)

    def save_model(self, directory = None, tag = ''):
        if directory is None:
            return

        if not os.path.isdir(directory):
            os.makedirs(directory)

        torch.save(self.local_network.state_dict(), '%s/model%s.pkl' % (directory, ('-' + tag)))

    def pick_action(self, state, epsilon, steps = 0):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Query the network
        action_values = self.local_network.forward(state_tensor)

        if np.random.uniform() > epsilon:
            action = action_values.max(1)[1].item()

        else:
            if steps < 10000:
                action = np.random.choice(np.arange(0, 3), p = [0.5, 0.25, 0.25])
            else:
                action = np.random.randint(0, 3)

        return action



        

