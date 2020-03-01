
from utils.config import Config

class DQNAgent:
    def __init__(self, config: Config):

        # Parameter initialization
        self.state_size = config.state_dim
        self.action_ize = config.action_dim

        self.hyperparameters = config.hyperparameters

        self.seed = random.seed(config.ssed)

        self.step_number = 0

        if config.use_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        else:
            self.device = "cpu"

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.hyperparameters["learning_rate"])

        # Initialise replay memory
        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], self.seed)
