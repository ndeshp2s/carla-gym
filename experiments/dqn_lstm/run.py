import argparse
import torch

import gym
from gym import error, spaces


from experiments.trainer import Trainer
from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv                                      
from experiments.config import Config
from rl_agents.DQN.ddqn import DDQNAgent



def main(args):

    # Load configuration
    config = Config()

    config.env = args.env

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
    
    config.use_cuda = True

    config.number_of_episodes = 1000
    config.steps_per_episode = 1000
    config.previous_episode = 0


    # Initialize the environment
    env = gym.make('Urban-v0')
    config.state_dim = env.observation_space
    config.action_dim = env.action_space.n


    # Initialize the agent
    agent = DDQNAgent(config)

    if args.train:
        trainer = Trainer(env, agent, config)

    elif args.test:
        None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--config_path', type=str, help='file with training parameters')
    parser.add_argument('--env', default='Urban-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()

    main(parser.parse_args())