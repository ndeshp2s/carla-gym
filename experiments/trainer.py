import os
import math
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

        #data_vis = DataVisualization(x_min = 0, x_max = 60, y_min = -5, y_max = 25)
        
        epsilon = self.config.hyperparameters["epsilon_before_learning"]

        for ep_num in range(previous_episode + 1, self.config.number_of_episodes + 1 + self.config.hyperparameters["batch_size"]):
        
        #while total_steps < self.config.total_steps:

            # Reset the environment and variables for new episode
            episode_steps = 0
            episode_reward = 0
            state = self.env.reset()
            exploration = True

            # if episode_number%2 == 0:
            #     self.config.spawner = True
            # else:
            #     self.config.spawner = False


            if self.config.spawner:
                self.spawner.reset()

            local_memory = []

            hidden_state1, cell_state1 = self.agent.local_network.init_hidden_states(batch_size = 1, lstm_memory = 512)
            hidden_state2, cell_state2 = self.agent.local_network.init_hidden_states(batch_size = 1, lstm_memory = 128)

            #print(self.agent.memory_collision.__len__(), self.agent.memory.__len__())

            for step_num in range(self.config.steps_per_episode):
                #data_vis.display(state[0])
                
                size_ = int(self.config.hyperparameters["batch_size"]/2)
                if self.agent.memory_collision.__len__() >= size_ and self.agent.memory.__len__() >= size_ and self.start_learning == False:
                    self.start_learning = True
                    episode_number = previous_episode + 1
                    ep_num = previous_episode + 1
                    total_steps = 0

                    epsilon = self.config.hyperparameters["epsilon_start"]

                    # if self.reset_epsilon:
                    #     epsilon = self.config.hyperparameters["epsilon_start"]
                    #     self.reset_epsilon = False
                    print('Pre training Memory filled. ep_num, episode_number', ep_num, episode_number)


                if episode_number < ((self.config.number_of_episodes) / 4):
                    exploration = True
                else:
                    exploration = False
                # Select action
                action, hidden_state1, cell_state1, hidden_state2, cell_state2, q_values = self.agent.pick_action(state = state, batch_size = 1, time_step = 1, hidden_state1 = hidden_state1, cell_state1 = cell_state1, hidden_state2 = hidden_state2, cell_state2 = cell_state2, epsilon = epsilon, learning = exploration, pre_train = not self.start_learning)
                if DEBUG:
                    action = input('Enter to continue: ')
                    action = int(action)
                   # input('Enter to continue: ')

                
                # Execute action for 4 times
                next_state, reward, done, info = self.env.step(action)
                # for i in range(3):
                #     next_state, reward, done, info = self.env.step(3)
                if DEBUG:
                    print(action, self.env.get_ego_speed(), reward, self.env.planner.local_planner.get_target_speed())

                # Add experience to memory of local network
                local_memory.append((state, action, reward, next_state, done))
            #     #self.agent.add(state, action, reward, next_state, done)

                # Update parameters
                state = next_state
                episode_reward += reward
                if self.start_learning: 
                    total_steps += 1
                episode_steps += 1

                # Execute spwner step
                if self.config.spawner:
                    if ep_num%2 == 0 and self.start_learning is True:
                        self.spawner.run_step(crossing = True)
                    else:
                        self.spawner.run_step(crossing = True)


                # Performing learning if minumum required experiences gathered
                loss = 0

                if self.start_learning:
                    for i in range(1):
                        loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"], time_step = self.config.hyperparameters["sequence_length"], step = total_steps)

                    #self.writer.add_scalar('Loss per step', loss, total_steps)



                if done:
                    if self.config.spawner:
                        self.spawner.destroy_all()
                    break

                # epsilon update
                # Only after a few initial steps
                # if self.start_learning:
                #     epsilon = self.epsilon_decay(total_steps)

                #     self.writer.add_scalar('Epsilon decay', epsilon, total_steps)


            # Save the episode
            if len(local_memory) >= (self.config.hyperparameters["sequence_length"]):
                if info == 'Collision':
                    self.agent.add(local_memory, collision = True)
                else:
                    self.agent.add(local_memory, collision = False)

                # if info == 'Collision' and self.start_learning is True:
                #     for i in range(3):
                #         self.agent.add(local_memory)
                    #self.agent.add(local_memory)


            # Print details of the episode
            print("----------------------------------------------------------")
            print("Episode: %d, Reward: %5f, Steps: %d, Loss: %4f, Total Step: %d, Info: %s, Epsilon: %4f" % (episode_number, episode_reward, episode_steps, loss, total_steps, info, epsilon))
            print("----------------------------------------------------------")

            # Save episode reward and loss
            if self.start_learning:
                self.writer.add_scalar('Reward per episode', episode_reward, episode_number)
                self.writer.add_scalar('Steps per episode', episode_steps, episode_number)
                self.writer.add_scalar('Loss per episode', loss, episode_number)

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
            if self.start_learning == False:
                ep_num = previous_episode + 1

            episode_number = episode_number + 1

            # Epsilon decay
            if self.start_learning:
                if epsilon > self.config.hyperparameters["epsilon_end"]:
                    epsilon -= (self.config.hyperparameters["epsilon_start"] - self.config.hyperparameters["epsilon_end"])/(self.config.number_of_episodes - 20)

                self.writer.add_scalar('Epsilon decay', epsilon, episode_number)


        # Once done, close environment
        self.close()


    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        print("Loading old network parameters")
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        #self.previous_episode = checkpoint['episode']
        #self.config.hyperparameters["epsilon_start"] = 0.2#checkpoint['epsilon']
        #self.config.hyperparameters["epsilon_before_learning"] = 0.2#checkpoint['epsilon']
        self.steps = 0#checkpoint['total_steps']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode, self.steps)







