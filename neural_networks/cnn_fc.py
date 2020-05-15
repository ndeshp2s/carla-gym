import torch
import torch.nn as nn
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels = self.input_size[2], out_channels = 32, kernel_size = 4, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 1)
        self.adapt = nn.AdaptiveMaxPool2d(output_size = (4, 4))

        self.fc1 = nn.Linear(in_features = 64 + 1, out_features = 32, bias = True)
        self.fc2 = nn.Linear(in_features = 32, out_features = self.output_size, bias = True)

        self.relu = nn.ReLU()

        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x1, x2, batch_size = 1):

        # (N, C, H, W) batch size, input channel, input height, input width
        x1 = x1.view(batch_size, self.input_size[2], self.input_size[0], self.input_size[1])

        conv_out = self.conv1(x1)
        conv_out = self.relu(conv_out) 
        #conv_out = self.max_pool1(conv_out)
        # print(conv_out.size())
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)
        #conv_out = self.max_pool2(conv_out)
        # print(conv_out.size()) 
        conv_out = self.conv3(conv_out)
        conv_out = self.relu(conv_out)
        #conv_out = self.max_pool3(conv_out)
        # print(conv_out.size()) 

        n_features = np.prod(conv_out.size()[1:])
        output1 = conv_out.view(-1, n_features)
        output2 = x2.view(batch_size, 1)

        output = torch.cat((output2, output1), dim = 1)
        output = self.fc1(output)
        # print(output.size())
        output = self.relu(output)
        output = self.fc2(output)
        # print(output.size())

        return output



# dummy1 = torch.randn([32, 32, 3])
# dummy2 = torch.randn([1])
# dummy = []
# # dummy.appedn(dummy1)
# # dummy.append(dummy2)
# # print(dummy.shape)
# model = NeuralNetwork(input_size = dummy1.shape, output_size = 3)

# model.forward(x1 = dummy1, x2 = dummy2, batch_size = 1)