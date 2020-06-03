import os
import math
import torch

from experiments.config import Config
from agents.tools.misc import get_speed

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

        for i in range(0, 10):

            state = self.env.reset()
            self.spawner.reset()

            hidden_state, cell_state = self.agent.local_network.init_hidden_states(batch_size = 1)

            input('Enter to continue: ')

            for step_num in range(self.config.steps_per_episode*5):
                # Select action
                action, hidden_state, cell_state, model_output = self.agent.pick_action(state = state, batch_size = 1, time_step = 1, \
                                                                                                        hidden_state = hidden_state, cell_state = cell_state, epsilon = epsilon)

                if DEBUG:
                    input('Enter to continue: ')
                

                # Execute action for 10 times
                ev_speed = self.env.get_ego_speed()
                next_state, reward, done, info = self.env.step(action, model_output = model_output, speed = ev_speed)
                # for i in range(3):
                #     next_state, reward, done, info = self.env.step(action)
                print(action, self.env.get_ego_speed(), reward, self.env.planner.local_planner.get_target_speed())
                # Update parameters
                state = next_state
                episode_reward += reward

                # Execute spwner step
                if self.config.spawner:
                    self.spawner.run_step()

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
