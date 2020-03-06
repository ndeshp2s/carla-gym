import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 0, output_size = 0):
        super(NeuralNetwork, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8, stride = 4)
        # self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        # self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)

        # self.lstm = nn.LSTM(input_size = , hidden_size = 786, num_layers = 1,batch_first = True)
        # self.out = nn.Linear(in_features = 786, out_features = self.output_size)

    def forward(self, x):
        # x = self.conv1(x)
        # x = torch.sigmoid(x)
        # x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # x = x.view(-1, 12*12*20)
        # x = self.fc1(x)
        # x = torch.sigmoid(x)

        # x = self.out(x)
        return x

class LSTMDQN(nn.Module):
    def __init__(self, n_action):
        super(LSTMDQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=1)  # (In Channel, Out Channel, ...)
        print(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(16, 128, 1)  # (Input, Hidden, Num Layers)

        self.affine1 = nn.Linear(128 * 64, 512)
        # self.affine2 = nn.Linear(2048, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x, hidden_state, cell_state):
        # CNN
        x = x.view(1*1,1,84,84)
        print(x.size())

        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        print(h.size())

        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
        print(h.size())

        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
        print(h.size())

        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))
        print(h.size())

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        print(h.size())
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        #print(h.size())
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)
        #print(h.size())

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state

    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 64, 128))
        cell_state = Variable(torch.zeros(1, 64, 128))
        return hidden_state, cell_state



class Network(nn.Module):
    
    def __init__(self,input_size,out_size):
        super(Network,self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        
        self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=8,stride=4) # potential check - in_channels
        self.conv_layer2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64,out_channels=512,kernel_size=7,stride=1)
        self.lstm_layer = nn.LSTM(input_size=512,hidden_size=512,num_layers=1,batch_first=True)
        self.adv = nn.Linear(in_features=512,out_features=self.out_size)
        self.val = nn.Linear(in_features=512,out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self,x,bsize,time_step,hidden_state,cell_state):
        #print(x.size())
        x = x.view(bsize*time_step,1,self.input_size,self.input_size)
        print(x.size())
        
        conv_out = self.conv_layer1(x)
        conv_out = self.relu(conv_out)
        print(conv_out.size())
        conv_out = self.conv_layer2(conv_out)
        conv_out = self.relu(conv_out)
        print(conv_out.size())
        conv_out = self.conv_layer3(conv_out)
        conv_out = self.relu(conv_out)
        print(conv_out.size())
        conv_out = self.conv_layer4(conv_out)
        conv_out = self.relu(conv_out)
        print(conv_out.size())
        
        conv_out = conv_out.view(bsize,time_step,512)
        print(conv_out.size())
        
        lstm_out = self.lstm_layer(conv_out,(hidden_state,cell_state))
        out = lstm_out[0][:,time_step-1,:]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]
        
        adv_out = self.adv(out)
        val_out = self.val(out)
        
        qout = val_out.expand(bsize,self.out_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize,self.out_size))
        
        return qout, (h_n,c_n)
    
    def init_hidden_states(self, bsize):
        h = torch.zeros(1,bsize,512).float()
        c = torch.zeros(1,bsize,512).float()
        
        return h,c


lstm_dqn = LSTMDQN(4)
train_hidden_state, train_cell_state = lstm_dqn.init_states()

o = np.zeros([84, 84])
torch_current_states = torch.from_numpy(o).float()

lstm_dqn.forward(torch_current_states, train_hidden_state, train_cell_state)

print("-----------------------------------------")

lstm = Network(84, 4)

h, c = lstm.init_hidden_states(1)

state = np.zeros([84, 84])
torch_x = torch.from_numpy(state).float()

lstm.forward(torch_x, bsize = 1, time_step = 1, hidden_state = h, cell_state = c)