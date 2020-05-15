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

        self.conv1 = nn.Conv2d(in_channels = input_size[2], out_channels = 16, kernel_size = (8, 4), stride = 1, padding = (1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (4, 2), stride = 1, padding = (1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (2, 1), stride = 1, padding = (1, 1))
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 2), stride = 1, padding = (1, 1))

        self.lstm_layer1 = nn.LSTM(input_size = 256, hidden_size = self.lstm_memory, num_layers = 1, batch_first = True)

        #self.lstm_layer2 = nn.LSTM(input_size = 2, hidden_size = self.lstm_memory, num_layers = 1, batch_first = True)

        self.fc1 = nn.Linear(in_features = self.lstm_memory, out_features = 64)

        self.fc2 = nn.Linear(in_features = 64, out_features = self.output_size)
        
        self.relu = nn.ReLU()

        self.device = device


    def forward(self, x1, x2, batch_size, time_step, hidden_state, cell_state):
        # (N, C, H, W) batch size, input channel, input height, input width
        x1 = x1.view(batch_size*time_step, self.input_size[2], self.input_size[0], self.input_size[1])

        x1 = self.conv1(x1)
        #print(x1.size())
        x1 = self.max_pool1(x1)
        print(x1.size())

        x1 = self.conv2(x1)
        #print(x1.size())
        x1 = self.max_pool2(x1)
        print(x1.size())

        x1 = self.conv3(x1)
        x1 = self.max_pool3(x1)
        print(x1.size())
        x1 = self.conv4(x1)
        x1 = self.max_pool3(x1)
        print(x1.size())

        #n_features = np.prod(x1.size()[1:])

        x1 = x1.view(batch_size, time_step, 256)
        print(x1.size())

        lstm1_out = self.lstm_layer1(x1, (hidden_state, cell_state))
        output1 = lstm1_out[0][:, time_step - 1, :]
        h_n1 = lstm1_out[1][0]
        c_n1 = lstm1_out[1][1]

        # x2 = x2.view(batch_size, time_step, 2)
        # lstm2_out = self.lstm_layer2(x2, (hidden_state, cell_state))
        # output2 = lstm2_out[0][:, time_step - 1, :]
        # h_n2 = lstm2_out[1][0]
        # c_n2 = lstm2_out[1][1]

        # output = torch.cat((output2, output1), dim = 1)
        # print(output.size())

        output = self.fc1(output1)
        output = self.relu(output)
        output = self.fc2(output)
        print(output)

        return output, (h_n1, c_n1), (None, None) #, (h_n2, c_n2)


    def init_hidden_states(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        c = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        
        return h, c




import random

INPUT_IMAGE_DIM = 84
OUT_SIZE = 4
BATCH_SIZE = 1
TIME_STEP = 1

# Dummy data
x = np.zeros([60,30,3])
batch = []
for i in range(TIME_STEP*BATCH_SIZE):
    batch.append(x)

ev_data = np.zeros([1, 1])


model = NeuralNetwork(input_size=x.shape, output_size=OUT_SIZE, lstm_memory=256)
hidden_state, cell_state = model.init_hidden_states(batch_size=1)
target = NeuralNetwork(input_size=x.shape, output_size=OUT_SIZE, lstm_memory=256)
print(model)


 # Dummy data
x = []
x.append(np.zeros([60,30,3]))
x.append(np.zeros([2]))

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

model_output = model.forward(states1, states2, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)

model_output[0]
model_output[1][0]
model_output[1][1]
model_output[2][0]
model_output[2][1]