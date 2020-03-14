import torch
import torch.nn as nn
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels = self.input_size[2], out_channels = 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1)
        self.adapt = nn.AdaptiveMaxPool2d(output_size = (4, 4))

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x, batch_size = 1, time_step = 1):

        # (N, C, H, W) batch size, input channel, input height, input width
        x = x.view(1, self.input_size[2], self.input_size[0], self.input_size[1])

        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)        
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)        
        conv_out = self.conv3(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.adapt(conv_out)
        conv_out = self.relu(conv_out)

        n_features = np.prod(conv_out.size()[1:])
        output = conv_out.view(-1, n_features)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output