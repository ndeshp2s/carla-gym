import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import Adam
from experiments.config import Config
from rl_agents.DQN.replay_buffer import ReplayBuffer
from neural_networks.fc32_fc32 import NeuralNetwork
import random
from torch import nn
import torch.nn.functional as F


class DDQNAgent:
    def __init__(self, config: Config):
        self.config = config

        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        # self.model = DQN(self.config.state_dim, self.config.action_dim).cuda()
        # self.target_model = DQN(self.config.state_dim, self.config.action_dim).cuda()
        # self.target_model.load_state_dict(self.model.state_dict())
        # self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # if self.config.use_cuda:
        #     self.model.cuda()

        # # Parameter initialization
        # self.state_size = config.state_dim
        # self.action_size = config.action_dim

        self.hyperparameters = config.hyperparameters

        # # self.step_number = 0

        if config.use_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        else:
            self.device = "cpu"

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.config.state_dim, self.config.action_dim).cuda()
        self.target_network = NeuralNetwork(self.config.state_dim, self.config.action_dim).cuda()
        self.target_network.load_state_dict(self.local_network.state_dict())
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.config.learning_rate, eps=1e-4)
        self.criterion = torch.nn.MSELoss()

        if self.config.use_cuda:
            self.local_network.cuda()

        # # Initialise replay memory
        # self.memory = ReplayBuffer(self.hyperparameters["buffer_size"])

    def add(self, state, action, reward, next_state, done):
        self.memory.add_experience(state, action, reward, next_state, done) 



    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        q_values = self.local_network(s0).cuda()
        next_q_values = self.local_network(s1).cuda()
        next_q_state_values = self.target_network(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())

        return loss.item()

    def learn(self, step = 0):

        # if experiences is None:
        #     #states, actions, rewards, next_states, dones  = self.memory.sample(batch_size)
        #     
        experiences  = self.buffer.sample(self.hyperparameters["batch_size"])

        # else:
        #     states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)


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

        # s0, a, r, s1, done = self.buffer.sample(batch_size)

        # states = torch.from_numpy(np.vstack([state for state in s0 if state is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([action for action in a if action is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([reward for reward in r if reward is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([next_state for next_state in s1 if next_state is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([d for d in done if d is not None]).astype(np.uint8)).float().to(self.device)

        #target_values = rewards[:,8-1]
        #print(torch_rewards[:, 7])


        # s0 = torch.tensor(s0, dtype=torch.float)
        # s1 = torch.tensor(s1, dtype=torch.float)
        # a = torch.tensor(a, dtype=torch.long)
        # r = torch.tensor(r, dtype=torch.float)
        # done = torch.tensor(done, dtype=torch.float)

        # s0 = s0.cuda()
        # s1 = s1.cuda()
        # a = a.cuda()
        # r = r.cuda()
        # done = done.cuda()

        # q_values = self.local_network(s0).cuda()
        # next_q_values = self.local_network(s1).cuda()
        # next_q_state_values = self.target_network(s1).cuda()


        # q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        # next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # expected_q_value = r + 0.99 * next_q_value * (1 - done)
        # #print(expected_q_value.detach())
        # # Notice that detach the expected_q_value
        # loss = (q_value - expected_q_value.detach()).pow(2).mean()


        q_local_predicted_all = self.local_network(states)
        q_next_Local_predicted_all = self.local_network(next_states)
        q_next_target_predicted_all = self.target_network(next_states)

        max_action_indexes = self.local_network(next_states).detach().argmax(1)
        q_next_max = self.target_network(next_states).gather(1, max_action_indexes.unsqueeze(1))
        #q_next_target_predicted_all.detach()#gather(1, q_next_Local_predicted_all.detach().max(1)[0].unsqueeze(1))

        q_predicted = q_local_predicted_all.gather(1, actions.long())
        q_target = rewards + (self.hyperparameters["discount_rate"] * q_next_max * (1 - dones))

        # with torch.no_grad():
        #     Q_targets = self.compute_q_targets(next_states, rewards, dones)
        # Q_expected = self.compute_expected_q_values(states, actions)

        # loss = F.mse_loss(Q_expected, Q_targets)

        # self.take_optimisation_step(self.optimizer, self.local_network, loss, self.hyperparameters["gradient_clipping_norm"])



        




        # q_predicted_all = self.local_network(states)
        # q_predicted = q_predicted_all.gather(1, actions.long())
        # q_next_predicted_all = self.local_network(next_states)
        # max_action_index = q_next_predicted_all.detach().argmax(1)
        # q_next_target = self.target_network(next_states)
        # #q_max_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # q_max_next = q_next_target.gather(1, max_action_index.unsqueeze(1))
        # q_target = rewards + (self.hyperparameters["discount_rate"] * q_max_next.detach() * (1 - dones))

        # #self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # loss = self.criterion(q_predicted, q_target)#(q_predicted- q_target.detach()).pow(2).mean()

        #print(expected_q_value, q_target)

        # Q_expected = self.compute_expected_q_values(states, actions)

        # with torch.no_grad():
        #     Q_targets = self.compute_q_targets(next_states, rewards, dones)
        #print(Q_targets)

        #print(q_values)
        #print(q_predicted_all)

        # print(q_value, q_predicted)
        # print(next_q_values, q_next_predicted_all)
        # print(next_q_state_values, q_next_target)
        #print(next_q_value, q_max_next)
        #print(loss, loss_m)
        # #
        #print(r, rewards)
        #print(expected_q_value, q_target)

        #print(torch.eq(loss, loss_m))
        # print('---------------------')





        loss = self.criterion(q_predicted, q_target)
        #loss = (q_predicted - q_target).pow(2).mean()

        #print(torch.eq(loss, lossm))

        # make previous grad zero
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # # Gradient clipping
        # for param in self.local_network.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # update params
        self.optimizer.step()

        if step % self.config.update_tar_interval == 0:
            self.target_network.load_state_dict(self.local_network.state_dict())

        return loss.item()

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients


    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.local_network(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.local_network(next_states).detach().argmax(1)
        Q_targets_next = self.target_network(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next 

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

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

        rnd_num = np.random.uniform()
        if rnd_num > epsilon:
            action = action_values.max(1)[1].item()

        else:
            action = np.random.randint(0, action_values.shape[1])

        return action


    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action



        

