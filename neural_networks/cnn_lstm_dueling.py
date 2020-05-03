import torch
import torch.nn as nn
import numpy as np
from experiments.config import Config

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0, device = "cpu", lstm_memory = 512):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_memory = lstm_memory

        self.conv1 = nn.Conv2d(in_channels = input_size[2], out_channels = 32, kernel_size = 4, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 512, kernel_size = (7, 3), stride = 1)

        self.lstm_layer = nn.LSTM(input_size = 512, hidden_size = self.lstm_memory, num_layers = 1, batch_first = True)

        self.adv = nn.Linear(in_features = self.lstm_memory + 2, out_features = self.output_size)
        self.val = nn.Linear(in_features = self.lstm_memory + 2, out_features = 1)
        
        self.relu = nn.ReLU()

        self.device = device


    def forward(self, x1, x2, batch_size, time_step, hidden_state, cell_state):
        # (N, C, H, W) batch size, input channel, input height, input width
        x1 = x1.view(batch_size*time_step, self.input_size[2], self.input_size[0], self.input_size[1])
        conv_out = self.conv1(x1)
        conv_out = self.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = self.relu(conv_out)
        n_features = np.prod(conv_out.size()[1:])
        conv_out = conv_out.view(batch_size, time_step, n_features)

        lstm_out = self.lstm_layer(conv_out, (hidden_state, cell_state))
        output1 = lstm_out[0][:,time_step-1,:]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        x2 = x2.view(batch_size, time_step, 2)
        output2 = x2[:, time_step - 1, :]

        output = torch.cat((output2, output1), dim = 1)
        
        adv_out = self.adv(output)
        val_out = self.val(output)

        q_output = val_out.expand(batch_size, self.output_size) + (adv_out - adv_out.mean(dim = 1).unsqueeze(dim = 1).expand(batch_size, self.output_size))

        return q_output, (h_n, c_n)


    def init_hidden_states(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        c = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        
        return h, c

DEBUG = 0

if DEBUG:
    # Dummy data
    x = np.zeros([60,30,3])

    model = NeuralNetwork(input_size=x.shape, output_size=4)

    print(model)

