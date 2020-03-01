import argparse
import torch

import gym
from gym import error, spaces

import environments
from utils.config import Config
from agents.DQN.dqn import DQNAgent

def load_config(observation_space, action_space):

    config = Config()
    config.seed = 1

    config.action_dim = action_space
    
    config.state_dim = observation_space
    
    config.use_cuda = True

    config.num_of_episodes = 1000
    config.steps_per_episode = 1000
    config.current_episode = 0

    config.hyperparameters = {
        "learning_rate": 0.1,
        "batch_size": 32,
        "buffer_size": int(1e5),
        "update_every_n_steps": 1,
        "min_steps_before_learning": 1000,
        "epsilon_start": 1,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.996,
        "discount_rate": 0.99,
        "tau": 0.01
    }

    config.config_path = "/home/niranjan/carla-gym/experiments/test1/config/"
    config.model_path = config.path = "/home/niranjan/carla-gym/experiments/test1/model/"

    return config


def main(args):

	# Initialize the environment
	env = gym.make('Urban-v0')

	state_size = env.observation_space
	action_size = env.action_space.n

	config = load_config(env.observation_space, env.action_space.n)

	print(state_size)
	print(action_size)

	# Initialize the agent
	agent = DQNAgent(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='Urban-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()

    main(parser.parse_args())