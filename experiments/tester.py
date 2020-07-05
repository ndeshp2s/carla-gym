import os
import math
import torch

from experiments.config import Config
from agents.tools.misc import get_speed
from experiments.util import EpsilonTracker, DataVisualization

DEBUG = 0
class Tester:
    def __init__(self, env, agent, spawner, config: Config):
        self.env = env
        self.agent = agent
        self.config = config
        self.spawner = spawner


    def test(self):

        episode_number = 0
        episode_reward = 0

        epsilon = 0

        average_speed = 0.0
        steps_for_speed = 0


        for i in range(0, 10):

            state = self.env.reset()
            self.spawner.reset()

            hidden_state1, cell_state1 = self.agent.local_network.init_hidden_states(batch_size = 1, lstm_memory = 512)
            hidden_state2, cell_state2 = self.agent.local_network.init_hidden_states(batch_size = 1, lstm_memory = 128)

            #input('Enter to continue: ')
            #data_vis = DataVisualization(x_min = 0, x_max = 32, y_min = -16, y_max = 16)

            for step_num in range(self.config.steps_per_episode*4):
                #data_vis.display(state[0])
                # Select action
                action, hidden_state1, cell_state1, hidden_state2, cell_state2, q_values = self.agent.pick_action(state = state, batch_size = 1, time_step = 1, hidden_state1 = hidden_state1, cell_state1 = cell_state1, hidden_state2 = hidden_state2, cell_state2 = cell_state2, epsilon = epsilon)

                if DEBUG:
                    input('Enter to continue: ')
                

                # Execute action for 10 times
                ev_speed = self.env.get_ego_speed()
                next_state, reward, done, info = self.env.step(action, model_output = q_values, speed = ev_speed)
                #for i in range(3):
                #   next_state, reward, done, info = self.env.step(action)
                print(action, self.env.get_ego_speed(), reward, self.env.planner.local_planner.get_target_speed(), step_num)

                # calculate average speed
                if self.env.get_ego_speed() > 0.5:
                    average_speed = average_speed + self.env.get_ego_speed()
                    steps_for_speed = steps_for_speed + 1 
                # Update parameters
                state = next_state
                episode_reward += reward

                # Execute spwner step
                if self.config.spawner and step_num > 0:
                    self.spawner.run_step(crossing = True)


                # if (130 > step_num > 100) or (230 > step_num > 200):
                #     self.spawner.set_factors(0.0, 0.0)

                # else: 
                #     self.spawner.set_factors(0.7, 0.2)
                #     self.spawner.run_step()

                if done:
                    self.spawner.destroy_all()
                    break

            # Print details of the episode
            print("----------------------------------------------------------")
            print("Episode: %d, Reward: %5f, Info: %s" % (episode_number, episode_reward, info))
            print("Speed:", average_speed, steps_for_speed)
            print("----------------------------------------------------------")

            # Update epsiode count
            episode_number = episode_number + 1

                    


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])

        self.agent.local_network.eval()
        self.agent.target_network.eval()


    def close(self):
        self.env.close()
