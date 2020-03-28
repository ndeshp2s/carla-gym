import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(NeuralNetwork, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)
        
# class NeuralNetwork(nn.Module):

#     """
#     #################################################
#     Initialize neural network model 
#     Initialize parameters and build model.
#     """
#     def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
#         """
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(NeuralNetwork, self).__init__()

#         self.input_features = state_size
#         # self.input_features = self.input_features.view(-1, self.input_features)
#         # self.output_features = action_size
       
#         self.fc1 = nn.Linear(self.input_features, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)


#     """
#     ###################################################
#     Build a network that maps state -> action values.
#     """
#     def forward(self, state):
#         x = state.view(-1, self.input_features)
        
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x


# inpt1 = torch.randn([1, 30, 20, 3]) #np.zeros([self.state_y, self.state_x, self.channel])
# model = NeuralNetwork((30, 20, 3), 4)

# out = model(inpt1)
# print(out)
# print(out.max(1)[1].item())
# #action_values.max(1)[1].item()