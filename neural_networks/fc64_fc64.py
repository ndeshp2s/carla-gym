import torch
import torch.nn as nn
#import torch.nn.Functional as F

class NeuralNetwork(nn.Module):

    """
    #################################################
    Initialize neural network model 
    Initialize parameters and build model.
    """
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NeuralNetwork, self).__init__()
       
        self.fc1 = nn.Linear(state_size[0], fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)


    """
    ###################################################
    Build a network that maps state -> action values.
    """
    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
