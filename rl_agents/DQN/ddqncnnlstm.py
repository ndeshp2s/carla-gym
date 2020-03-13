import torch
from rl_agents.DQN.ddqn import DDQNAgent
from experiments.config import Config
from neural_networks.cnn_lstm import NeuralNetwork


class DDQNCNNLSTMAgent(DDQNAgent):
    def __init__(self, config: Config):
        DDQNAgent.__init__(self, config)

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.state_size, self.action_size, self.device).to(self.device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size, self.device).to(self.device)


    def pick_action(self, state, hidden_state, cell_state, epsilon):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Query the network
        action_values = self.local_network.forward(state_tensor, batch_size = 1, time_step = 1, hidden_state = hidden_state, cell_state = cell_state)

        # if np.random.uniform() > epsilon:
        #     action = action_values.max(1)[1].item()

        # else:
        #     action = np.random.randint(0, action_values.shape[1])

        # return action