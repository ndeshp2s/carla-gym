import random
import numpy as np
import torch
import torch.optim as optim

from experiments.config import Config
from rl_agents.DQN.replay_buffer import ReplayBuffer
from neural_networks.cnn_lstm import NeuralNetwork


class DDQNCNNLSTMAgent:
    def __init__(self, config: Config):

        # Parameter initialization
        self.state_size = config.state_dim
        self.action_size = config.action_dim

        self.hyperparameters = config.hyperparameters

        # if config.use_cuda:
        #     self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # else:
        self.device = "cpu"

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.state_size, self.action_size, self.device).to(self.device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size, self.device).to(self.device)
        self.target_network.load_state_dict(self.local_network.state_dict())
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.hyperparameters["learning_rate"])
        self.criterion = torch.nn.MSELoss()

        # Initialise replay memory
        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"])


    def add(self, episode):
        self.memory.add_episode(episode)


    def learn(self, batch_size, time_step, experiences = None, step = 0, soft_update = True):


        hidden_batch1, cell_batch1 = self.local_network.init_hidden_states(batch_size = batch_size, lstm_memory = 256)
        hidden_batch2, cell_batch2 = self.local_network.init_hidden_states(batch_size = batch_size, lstm_memory = 128)

        batch  = self.memory.get_batch(batch_size = batch_size, time_step = time_step)

        states1 = []
        states2 = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []

        for b in batch:
            s1, s2, a, r, ns1, ns2 = [], [], [], [], [], []
            for e in b:
                s1.append(e[0][0])
                s2.append(e[0][1])
                a.append(e[1])
                r.append(e[2])
                ns1.append(e[3][0])
                ns2.append(e[3][1])
                
            states1.append(s1)
            states2.append(s2)
            actions.append(a)
            rewards.append(r)
            next_states1.append(ns1)
            next_states2.append(ns2)


        states1 = torch.from_numpy(np.array(states1)).float().to(self.device)
        states2 = torch.from_numpy(np.array(states2)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states1 = torch.from_numpy(np.array(next_states1)).float().to(self.device)
        next_states2 = torch.from_numpy(np.array(next_states2)).float().to(self.device)

        # Get the q values for all actions from local network
        q_predicted_all, _, _ = self.local_network.forward(x1 = states1, x2 = states2, batch_size = batch_size, time_step = time_step, hidden_state1 = hidden_batch1, cell_state1 = cell_batch1, hidden_state2 = hidden_batch2, cell_state2 = cell_batch2)
        #Get the q value corresponding to the action executed
        q_predicted = q_predicted_all.gather(dim = 1, index = actions[:, time_step - 1].unsqueeze(dim = 1)).squeeze(dim = 1)
        # Get q values for all the actions of next state
        q_next_predicted_all, _, _ = self.local_network.forward(x1 = next_states1, x2 = next_states2, batch_size = batch_size, time_step = time_step, hidden_state1 = hidden_batch1, cell_state1 = cell_batch1, hidden_state2 = hidden_batch2, cell_state2 = cell_batch2)
        
        # get q values for the actions of next state from target netwrok
        q_next_target_all, _, _ = self.target_network.forward(x1 = next_states1, x2 = next_states2, batch_size = batch_size, time_step = time_step, hidden_state1 = hidden_batch1, cell_state1 = cell_batch1, hidden_state2 = hidden_batch2, cell_state2 = cell_batch2)
        # get q value of action with same index as that of the action with maximum q values (from local network)
        q_next_target = q_next_target_all.gather(1, q_next_predicted_all.max(1)[1].unsqueeze(1)).squeeze(1)
        # Find target q value using Bellmann's equation
        q_target = rewards[:, time_step - 1] + (self.hyperparameters["discount_rate"] * q_next_target)


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

        # Update target network
        if soft_update:
            self.soft_update(self.local_network, self.target_network, self.hyperparameters["tau"])

        else:
            self.hard_update(step)
            

        return loss.item()


    def soft_update(self, local_network, target_network, tau):

        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1 - tau)*target_param.data)


    def hard_update(self, step = 0):
        if step % self.hyperparameters["update_every_n_steps"] == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())


    def pick_action(self, state, batch_size, time_step, hidden_state1, cell_state1, hidden_state2, cell_state2, epsilon, learning = True):
        state_tensor1 = torch.from_numpy(state[0]).float().unsqueeze(0).to(self.device)
        state_tensor2 = torch.from_numpy(state[1]).float().unsqueeze(0).to(self.device)

        # Query the network
        model_output = self.local_network.forward(x1 = state_tensor1, x2 = state_tensor2, batch_size = 1, time_step = 1, hidden_state1 = hidden_state1, cell_state1 = cell_state1, hidden_state2 = hidden_state2, cell_state2 = cell_state2)
        hidden_state1 = model_output[1][0]
        cell_state1 = model_output[1][1]
        hidden_state2 = model_output[2][0]
        cell_state2 = model_output[2][1]

        #if np.random.uniform() > epsilon:
        if random.random() > epsilon:
            #print('Q values: ', model_output[0])
            action = int(torch.argmax(model_output[0]))

        else:
            #action = np.random.randint(0, 4)
            #action = random.randrange(3)
            # if learning is False:
            #     action = np.random.choice(np.arange(0, 4), p = [0.5, 0.25, 0.0, 0.25])                
            # else:
            action = np.random.randint(0, 4)

        return action, hidden_state1, cell_state1, hidden_state2, cell_state2, model_output[0].squeeze(0)