import torch
import torch.nn as nn
import numpy as np
from experiments.config import Config


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0, device = "cpu", lstm_memory = 256):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_memory = lstm_memory

        self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 32, kernel_size = 4, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 1)

        self.fc1 = nn.Linear(in_features = 128, out_features = 32)
        self.fc2 = nn.Linear(in_features = 32, out_features = self.output_size)
        
        self.relu = nn.ReLU()

        self.device = device


    def forward(self, x1, batch_size, time_step):
        # (N, C, H, W) batch size, input channel, input height, input width
        # batch, channel, time 
        x1 = x1.view(batch_size, 2, time_step)

        x1 = self.conv1(x1)
        x1 = self.relu(x1)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)

        #x2 = torch.flatten(x2)
        n_features = np.prod(x1.size()[1:])
        output = x1.view(-1, n_features)

        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output


        # x1 = self.conv1(x1)
        # x1 = self.relu(x1)
        # #print(x1.size())

        # x1 = self.conv2(x1)
        # x1 = self.relu(x1)
        # #print(x1.size())

        # x1 = self.conv3(x1)
        # x1 = self.relu(x1)
        # #print(x1.size())

        # n_features = np.prod(x1.size()[1:])
        # #print(n_features)

        # x1 = x1.view(batch_size, time_step, n_features)

        # lstm_out = self.lstm_layer(x1, (hidden_state, cell_state))
        # output1 = lstm_out[0][:, time_step - 1, :]
        # h_n = lstm_out[1][0]
        # c_n = lstm_out[1][1]
        # x2 = x2.view(batch_size, time_step, 2)
        # output2 = x2[:, time_step - 1, :]

        # output = torch.cat((output2, output1), dim = 1)

        # output = self.fc1(output)
        # output = self.relu(output)
        # output = self.fc2(output)

        # return output, (h_n, c_n)

        return None


    def init_hidden_states(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        c = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        
        return h, c

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size = 0, output_size = 0, device = "cpu", lstm_memory = 256):
#         super(NeuralNetwork, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.lstm_memory = lstm_memory

#         self.conv1 = nn.Conv2d(in_channels = input_size[2], out_channels = 32, kernel_size = 4, stride = 2)
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2)
#         self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2)
#         self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = (7, 3), stride = 1)

#         self.lstm_layer = nn.LSTM(input_size = 256, hidden_size = self.lstm_memory, num_layers = 1, batch_first = True)

#         self.fc1 = nn.Linear(self.lstm_memory + 2, self.output_size)
        
#         self.relu = nn.ReLU()

#         self.device = device


#     def forward(self, x1, x2, batch_size, time_step, hidden_state, cell_state):
#         # (N, C, H, W) batch size, input channel, input height, input width
#         x1 = x1.view(batch_size*time_step, self.input_size[2], self.input_size[0], self.input_size[1])
#         conv_out = self.conv1(x1)
#         conv_out = self.relu(conv_out)
#         conv_out = self.conv2(conv_out)
#         conv_out = self.relu(conv_out)
#         conv_out = self.conv3(conv_out)
#         conv_out = self.relu(conv_out)
#         conv_out = self.conv4(conv_out)
#         conv_out = self.relu(conv_out)
#         n_features = np.prod(conv_out.size()[1:])
#         conv_out = conv_out.view(batch_size, time_step, n_features)

#         lstm_out = self.lstm_layer(conv_out, (hidden_state, cell_state))
#         output1 = lstm_out[0][:,time_step-1,:]
#         h_n = lstm_out[1][0]
#         c_n = lstm_out[1][1]

#         x2 = x2.view(batch_size, time_step, 2)
#         output2 = x2[:, time_step - 1, :]

#         output = torch.cat((output2, output1), dim = 1)
#         output = self.fc1(output)

#         return output, (h_n, c_n)


#     def init_hidden_states(self, batch_size):
#         h = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
#         c = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        
#         return h, c



DEBUG = 0

if DEBUG:
    import random

    INPUT_IMAGE_DIM = 84
    OUT_SIZE = 4
    BATCH_SIZE = 1
    TIME_STEP = 1

    # Dummy data
    x = np.zeros([32,32,4])
    batch = []
    for i in range(TIME_STEP*BATCH_SIZE):
        batch.append(x)

    ev_data = np.zeros([1, 1])


    model = NeuralNetwork(input_size=x.shape, output_size=OUT_SIZE, lstm_memory=128)
    hidden_state, cell_state = model.init_hidden_states(batch_size=1)
    target = NeuralNetwork(input_size=x.shape, output_size=OUT_SIZE, lstm_memory=128)
    print(model)


    # Dummy data
    x = []
    x.append(np.zeros([32,32,4]))
    x.append(np.zeros([2]))

    batch = []
    for i in range(TIME_STEP*BATCH_SIZE):
        batch.append(x)

    # x = np.array(batch)

    # torch_x = torch.from_numpy(x).float()
    # torch_ev_data = torch.from_numpy(ev_data).float()
    # output = model.forward(torch_x, torch_ev_data, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)
    # print(output[0])

    states1 = []
    states2 = []
    actions = []
    rewards = []
    next_states = []

    s1, s2, a, r, ns = [], [], [], [], []

    s1.append(x[0])
    s2.append(x[1])
    a.append(1)
    r.append(0)
    ns.append(x[0])

    states1.append(s1)
    states2.append(s2)
    actions.append(a)
    rewards.append(r)
    next_states.append(ns)

    states1 = torch.from_numpy(np.array(states1)).float()
    states2 = torch.from_numpy(np.array(states2)).float()
    actions = torch.from_numpy(np.array(actions)).long()
    rewards = torch.from_numpy(np.array(rewards)).float()
    next_states = torch.from_numpy(np.array(next_states)).float()

    states2 = torch.randn(1, 2, 1)

    q_predicted_all, _ = model.forward(states1, states2, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)

    # print(actions[:,TIME_STEP-1].unsqueeze(dim=1))

    # q_predicted = q_predicted_all.gather(dim=1,index=actions[:,TIME_STEP-1].unsqueeze(dim=1)).squeeze(dim=1)
    # print(q_predicted)

    # next_q_values, _ = model.forward(next_states, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)
    # next_q_state_values, _ = target.forward(next_states, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)

    # print(next_q_state_values)
    # print(next_q_values)

    # next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)

    # print(next_q_value)

    # expected_q_value = rewards[:,TIME_STEP-1] + 0.99 * next_q_value
    # print(expected_q_value)
    # print(next_q_values.max(1)[1].unsqueeze(1))

