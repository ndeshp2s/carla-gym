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

        if config.use_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        else:
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


    def learn(self, batch_size = 1, time_step = 1, experiences = None, step = 0):

        hidden_batch, cell_batch = self.local_network.init_hidden_states(batch_size = batch_size)

        batch  = self.memory.get_batch(batch_size = batch_size, time_step = time_step)
        
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        states = []
        actions = []
        rewards = []
        next_states = []

        for b in batch:
            s, a, r, ns = [], [], [], []
            for e in b:
                s.append(e[0])
                a.append(e[1])
                r.append(e[2])
                ns.append(e[3])
                
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

        # Get the q values for all actions from local network
        q_predicted_all, _ = self.local_network.forward(states, batch_size = batch_size, time_step = time_step, hidden_state = hidden_batch, cell_state = cell_batch)
        # Get the q value corresponding to the action executed
        q_predicted = q_predicted_all.gather(dim = 1, index = actions[:, time_step - 1].unsqueeze(dim = 1)).squeeze(dim = 1)
        # Get q values for all the actions of next state
        q_next_predicted_all, _ = self.local_network.forward(next_states, batch_size = batch_size, time_step = time_step, hidden_state = hidden_batch, cell_state = cell_batch)
        
        # get q values for the actions of next state from target netwrok
        q_next_target_all, _ = self.target_network.forward(next_states, batch_size = batch_size, time_step = time_step, hidden_state = hidden_batch, cell_state = cell_batch)
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

        if step % self.hyperparameters["update_every_n_steps"] == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())

        
        return loss.item()


    def pick_action(self, state, batch_size, time_step, hidden_state, cell_state, epsilon):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Query the network
        model_output = self.local_network.forward(state_tensor, batch_size = 1, time_step = 1, hidden_state = hidden_state, cell_state = cell_state)
        hidden_state = model_output[1][0]
        cell_state = model_output[1][1]

        if np.random.uniform() > epsilon:
            action = int(torch.argmax(model_output[0]))

        else:
            action = np.random.randint(0, model_output[0].shape[1])

        return action