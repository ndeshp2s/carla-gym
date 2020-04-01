import torch
import torch.nn as nn
import numpy as np
from experiments.config import Config

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0, device = "cpu", lstm_memory = 768):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_memory = lstm_memory

        self.conv1 = nn.Conv2d(in_channels = input_size[2], out_channels = 32, kernel_size = 4, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 512, kernel_size = 7, stride = 1)

        self.lstm_layer = nn.LSTM(input_size = 768, hidden_size = self.lstm_memory, num_layers = 1, batch_first = True)

        self.fc1 = nn.Linear(self.lstm_memory, self.output_size)
        
        self.relu = nn.ReLU()

        self.device = device


    def forward(self, x, batch_size, time_step, hidden_state, cell_state):
        # (N, C, H, W) batch size, input channel, input height, input width
        x = x.view(batch_size*time_step, self.input_size[2], self.input_size[1], self.input_size[0])
        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.relu(conv_out)
        n_features = np.prod(conv_out.size()[1:])
        conv_out = conv_out.view(batch_size, time_step, n_features)

        lstm_out = self.lstm_layer(conv_out, (hidden_state, cell_state))
        o = lstm_out[0][:,time_step-1,:]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        output = self.fc1(o)

        return output, h_n, c_n


    def init_hidden_states(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        c = torch.zeros(1, batch_size, self.lstm_memory).float().to(self.device)
        
        return h, c



DEBUG = 1

if DEBUG:
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


	model = NeuralNetwork(input_size=x.shape, output_size=OUT_SIZE, lstm_memory=512)
	hidden_state, cell_state = model.init_hidden_states(batch_size=1)

	states = np.array(batch)

	torch_x = torch.from_numpy(states).float()
	output = model.forward(torch_x, batch_size = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_state, cell_state = cell_state)
	print(output[0])