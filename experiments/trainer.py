import os
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from collections import namedtuple
from experiments.config import Config
#from experiments.util import EpsilonTracker, DataVisualization
from utils.miscellaneous import plot_grid

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


    def add_experience(self):
        data = {}
        data["experiences"] = []
        experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        
        experiences = []

        for i in range(4):
            

            state = self.env.reset()
            self.spawner.reset()

            for step_num in range(200):
                action = input('Enter to continue: ')
                action = int(action)

                self.spawner.run_step()
                next_state, reward, done, info = self.env.step(action)

                e = experience(state = state, action = action, reward = reward, next_state = next_state, done = done)
                experiences.append(e)

                state = next_state

        self.close()

        return experiences



    def train(self, previous_episode = -1, total_steps = 0):
        losses = []
        rewards = []
        episode_reward = 0
        episode_number = previous_episode + 1
        total_steps = total_steps
        learning = False
        episode_steps = 0

    #     #data_vis = DataVisualization(x_min = 0, x_max = 60, y_min = -5, y_max = 25)
        
        epsilon = self.config.hyperparameters["epsilon_before_learning"]

    #     #for ep_num in range(previous_episode + 1, self.config.number_of_episodes + 1):
        
        while total_steps < self.config.total_steps:

            # Reset the environment and variables for new episode
            episode_reward = 0
            episode_steps = 0
            state = self.env.reset()


            if self.config.spawner:
                self.spawner.reset()

            local_memory = []

            for step_num in range(self.config.steps_per_episode):
                #data_vis.display(state[0])
                if self.agent.memory.__len__() >= self.config.hyperparameters["min_steps_before_learning"] and self.start_learning == False:
                    self.start_learning = True
                    episode_number = previous_episode + 1
                    total_steps = 0

                    epsilon = self.config.hyperparameters["epsilon_start"]

                
                # Select action
                action = self.agent.pick_action(state, epsilon, steps = total_steps)

                # Execute spwner step
                if self.config.spawner:
                    self.spawner.run_step(step_num = total_steps)

                if DEBUG:
                    action = input('Enter to continue: ')
                    action = int(action)
                    #input('Enter to continue: ')

                
                # Execute action for 4 times
                for i in range(1):
                    next_state, reward, done, info = self.env.step(action)

                # Add experience to memory of local network
                self.agent.add(state = state, action = action, reward = reward, next_state = next_state, done = done)

                # Update parameters
                state = next_state
                episode_reward += reward
                episode_steps += 1
                #if self.start_learning: 
                total_steps += 1

                # Performing learning if minumum required experiences gathered
                loss = 0

                if self.start_learning:
                    loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"], step = total_steps)

                    self.writer.add_scalar('Loss per step', loss, total_steps)



                if done:
                    if self.config.spawner:
                        self.spawner.destroy_all()
                    break

                # epsilon update
                # Only after a few initial steps
                if self.start_learning:
                    epsilon = self.epsilon_decay(total_steps)

                    self.writer.add_scalar('Epsilon decay', epsilon, total_steps)


            # Print details of the episode
            print("----------------------------------------------------------")
            print("Episode: %d, Reward: %5f, Steps: %d, Loss: %4f, Total Step: %d, Info: %s, Epsilon: %4f" % (episode_number, episode_reward, episode_steps, loss, total_steps, info, epsilon))
            print("----------------------------------------------------------")

            # Save episode reward
            if self.start_learning:
                self.writer.add_scalar('Reward per episode', episode_reward, episode_number)
                self.writer.add_scalar('Step per episode', episode_steps, episode_number)

            # # # Save weights
            # # self.agent.save_model(self.config.model_dir)

            # Save checkpoints
            # if not os.path.isdir(self.config.checkpoint_dir):
            #     os.makedirs(self.config.checkpoint_dir)

            if self.config.checkpoint and episode_number % self.config.checkpoint_interval == 0 and self.start_learning:
                checkpoint = {'state_dict': self.agent.local_network.state_dict(),
                            'optimizer': self.agent.optimizer.state_dict(),
                            'episode': episode_number,
                            'epsilon': epsilon,
                            'total_steps': total_steps}
                torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')
            
            # # # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
            # # #     self.agent.save_checkpoint()

            # Update epsiode count
            #if self.start_learning:
            episode_number = episode_number + 1


        # Once done, close environment
        self.close()


    def train_on_experiences(self, experiences):
        for e in experiences:
            state = e.state
            action = e.action
            reward = e.reward
            done = e.done
            next_state = e.next_state
            if action == 22:
                action = 2
            self.agent.add(state = state, action = action, reward = reward, next_state = next_state, done = done)


        for i in range(10000):
            loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"], step = i)

            self.writer.add_scalar('Loss per step', loss, i)

            if i%100 == 0:
                print("Loss: %4f, Total Step: %d," % (loss, i))


    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        print("Loading old network parameters")
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.previous_episode = 0#checkpoint['episode']
        #self.config.hyperparameters["epsilon_start"] = checkpoint['epsilon']
        #self.config.hyperparameters["epsilon_before_learning"] = checkpoint['epsilon']
        self.steps = 0#checkpoint['total_steps']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode, self.steps)







