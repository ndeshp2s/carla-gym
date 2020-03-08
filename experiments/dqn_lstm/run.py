import argparse
import torch
import os, sys

import gym
from gym import error, spaces


from experiments.trainer import Trainer
from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv                                      
from experiments.config import Config
from rl_agents.DQN.ddqn import DDQNAgent



def main(args):

    # Directory of current experiment
    experiment_dir = 'experiments/dqn_lstm/test1'

    # Load configuration
    config = Config()

    config.env = args.env

    config.hyperparameters = {
        "learning_rate": 0.1,
        "batch_size": 32,
        "sequence_length": 1,
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
    config.total_steps = 10000

    config.checkpoint = True
    config.checkpoint_interval = 1
    config.checkpoint_dir = experiment_dir + '/checkpoints'

    config.log_dir = experiment_dir + '/logs'

    config.model_dir = experiment_dir + '/model'


    # Initialize the environment
    env = gym.make('Urban-v0')
    config.state_dim = env.observation_space.shape
    config.action_dim = env.action_space.n


    # Initialize the agent
    agent = DDQNAgent(config)

    if args.train:
        trainer = Trainer(env, agent, config)

        try:
            trainer.train()

        except KeyboardInterrupt:
            try:
                trainer.close()
                sys.exit(0)
            except SystemExit:
                trainer.close()
                os._exit(0)

    elif args.retrain:
        if args.checkpoint_file is None:
            print(' ---------- Please mention checkoutpoint file name ----------')
            return
        trainer = Trainer(env, agent, config)
        trainer.load_checkpoint(args.checkpoint_file)

    elif args.test:
        None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--config_path', type=str, help='file with training parameters')
    parser.add_argument('--env', default='Urban-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    parser.add_argument('--checkpoint_file', default = None, type=str, help='if retrain, import the model and previous parameters')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='re-train model')
    args = parser.parse_args()

    main(parser.parse_args())