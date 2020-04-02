import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
from experiments.util import EpsilonTracker, DataVisualization

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

        self.spawner = spawner


    def train(self, previous_episode = 0):
        losses = []
        rewards = []
        episode_reward = 0
        episode_number = 0
        total_steps = 0

        #data_vis = DataVisualization(x_min = 0, x_max = 30, y_min = -5, y_max = 25)
        
        epsilon = self.config.hyperparameters["epsilon_start"]

        #for ep_num in range(previous_episode + 1, self.config.number_of_episodes + 1):
        while total_steps < self.config.total_steps:

            # Reset the environment and variables for new episode
            episode_reward = 0
            state = self.env.reset()
            self.spawner.reset()

            local_memory = []

            hidden_state, cell_state = self.agent.local_network.init_hidden_states(batch_size = 1)


            for step_num in range(self.config.steps_per_episode):
                #data_vis.display(state)
                
                # Select action
                action = self.agent.pick_action(state = state, batch_size = 1, time_step = 1,\
                                                hidden_state = hidden_state, cell_state = cell_state, epsilon = epsilon)

                if DEBUG:
                    action = input('Enter to continue: ')
                    action = int(action)

                
                # Execute action for 10 times
                next_state, reward, done, info = self.env.step(action)
                for i in range(9):
                    next_state, reward, done, info = self.env.step(3)

                # Add experience to memory of local network
                local_memory.append((state, action, reward, next_state, done))
            #     #self.agent.add(state, action, reward, next_state, done)

                # Update parameters
                state = next_state
                episode_reward += reward
                total_steps += 1

                # Execute spwner step
                self.spawner.run_step()


                # Performing learning if minumum required experiences gathered
                loss = 0

                if total_steps > self.config.hyperparameters["min_steps_before_learning"] and total_steps % self.config.learing_frequency == 0 \
                and self.agent.memory.__len__() > self.config.hyperparameters["batch_size"]:
                    loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"], time_step = self.config.hyperparameters["sequence_length"], step = total_steps)

                    self.writer.add_scalar('Loss per step', loss, total_steps)


                if done:
                    self.spawner.destroy_all()
                    break

                # epsilon update
                # Only after a few initial steps
                if total_steps > self.config.hyperparameters["min_steps_before_learning"]:
                    epsilon = self.epsilon_decay(total_steps - self.config.hyperparameters["min_steps_before_learning"])

                self.writer.add_scalar('Epsilon decay', epsilon, total_steps)


            # Save the episode
            self.agent.add(local_memory)

            # Print details of the episode
            print("----------------------------------------------------------")
            print("Episode: %d, Reward: %5f, Loss: %4f, Total Step: %d, Info: %s, Epsilon: %4f" % (episode_number, episode_reward, loss, total_steps, info, epsilon))
            print("----------------------------------------------------------")

            # Save episode reward
            self.writer.add_scalar('Reward per episode', episode_reward, episode_number)

            # # # Save weights
            # # self.agent.save_model(self.config.model_dir)

            # Save checkpoints
            # if not os.path.isdir(self.config.checkpoint_dir):
            #     os.makedirs(self.config.checkpoint_dir)

            if self.config.checkpoint and episode_number % self.config.checkpoint_interval == 0:
                checkpoint = {'state_dict': self.agent.local_network.state_dict(),
                            'optimizer': self.agent.optimizer.state_dict(),
                            'episode': episode_number,
                            'epsilon': epsilon,
                            'total_steps': total_steps}
                torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')
            
            # # # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
            # # #     self.agent.save_checkpoint()

            # Update epsiode count
            episode_number = episode_number + 1


        # Once done, close environment
        self.close()


    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.previous_episode = checkpoint['episode']
        self.config.hyperparameters["epsilon_start"] = checkpoint['epsilon']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode)







