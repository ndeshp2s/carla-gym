import torch
import torch.nn as nn
import numpy as np
from experiments.config import Config

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0, device = "cpu"):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels = input_size[2], out_channels = 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 512, kernel_size = 7, stride = 1)

        self.lstm_layer = nn.LSTM(input_size = 512, hidden_size = 512, num_layers = 1, batch_first = True)

        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, self.output_size)

        self.relu = nn.ReLU()

        self.device = device


    def forward(self, x, batch_size, time_step, hidden_state, cell_state):
        # (N, C, H, W) batch size, input channel, input height, input width
        x = x.view(1, self.input_size[2], self.input_size[0], self.input_size[1])
        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)        
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)        
        conv_out = self.conv3(conv_out)
        conv_out = self.relu(conv_out)        
        conv_out = self.conv4(conv_out)
        conv_out = self.relu(conv_out)
        print(conv_out.size())

        conv_out = conv_out.view(batch_size,time_step, 512)
        lstm_out = self.lstm_layer(conv_out, (hidden_state, cell_state))
        # print(len(lstm_out))

        # out = lstm_out[0][:,time_step-1,:]
        # h_n = lstm_out[1][0]
        # c_n = lstm_out[1][1]

        # print(len(out))
        # print(len(h_n))
        # print(len(c_n))

        print(lstm_out[0])
        # print(h)
        # print(n)


    def init_hidden_states(self,batch_size):
        h = torch.zeros(1, batch_size, 512).float().to(self.device)
        c = torch.zeros(1, batch_size, 512).float().to(self.device)
        
        return h, c