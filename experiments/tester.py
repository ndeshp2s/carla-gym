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


            for step_num in range(self.config.steps_per_episode):
                # Select action
                action = self.agent.pick_action(state, epsilon, steps = step_num)
                print(action, self.env.get_ego_speed())

                if DEBUG:
                    input('Enter to continue: ')
                
                                # Execute spwner step
                if self.config.spawner:
                    self.spawner.run_step(step_num = step_num)

                # Execute action for 10 times
                next_state, reward, done, info = self.env.step(action)
                # for i in range(4):
                #     next_state, reward, done, info = self.env.step(action)
                print(reward, done, info)
                print(action, self.env.get_ego_speed(), reward, self.env.planner.local_planner.get_target_speed())
                # Update parameters
                state = next_state
                episode_reward += reward




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
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file, map_location='cpu')

        # Load network weights and biases
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])


        self.agent.local_network.eval()
        self.agent.target_network.eval()


    def close(self):
        self.env.close()
