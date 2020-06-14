import os
import math
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
#from experiments.util import EpsilonTracker, DataVisualization

DEBUG = 0
class Trainer:
    def __init__(self, env, agent, spawner, config: Config):
        self.env = env
        self.agent = agent
        self.config = config

        self.epsilon_decay = lambda frame_idx: self.config.hyperparameters["epsilon_end"] + (self.config.hyperparameters["epsilon_start"]\
                                    - self.config.hyperparameters["epsilon_end"]) * math.exp(-1. * frame_idx * self.config.hyperparameters["epsilon_decay"])
        if not os.path.isdir(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        self.writer = SummaryWriter(log_dir = self.config.log_dir)

        # Parameters for re-training
        self.previous_episode = 0
        self.epsilon = 0

        self.start_learning = False
        self.reset_epsilon = True

        self.spawner = spawner


    def train(self, previous_episode = -1, total_steps = 0):
        losses = []
        rewards = []
        episode_reward = 0
        episode_steps = 0
        episode_number = previous_episode + 1
        total_steps = total_steps
        learning = False

        state = self.env.reset()

        local_memory = []

        hidden_state, cell_state = self.agent.local_network.init_hidden_states(batch_size = 1)

        for i in range(100):

            action = input('Enter to continue: ')
            action = int(action)

                
            # Execute action for 4 times
            next_state, reward, done, info = self.env.step(action)

            local_memory.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            print(action, self.env.get_ego_speed(), reward, self.env.planner.local_planner.get_target_speed(), i)


        self.close()


        for i in range(32):
            self.agent.add(local_memory)

        # learn

        for i in range(10000):
            loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"], time_step = self.config.hyperparameters["sequence_length"], step = total_steps)

            checkpoint = {'state_dict': self.agent.local_network.state_dict(),
                                'optimizer': self.agent.optimizer.state_dict(),
                                'episode': 0,
                                'epsilon': 0,
                                'total_steps': 0}
            torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')

            # Print details of the episode
            print("----------------------------------------------------------")
            print("Loss: %4f, Steps: %d" % (loss, i))
            print("----------------------------------------------------------")





    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        print("Loading old network parameters")
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.previous_episode = checkpoint['episode']
        self.config.hyperparameters["epsilon_start"] = checkpoint['epsilon']
        self.config.hyperparameters["epsilon_before_learning"] = checkpoint['epsilon']
        self.steps = checkpoint['total_steps']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode, self.steps)