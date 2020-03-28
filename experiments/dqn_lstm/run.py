import argparse
import torch
import os, sys

import gym
from gym import error, spaces


from experiments.trainer import Trainer
from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv                                      
from experiments.config import Config
from rl_agents.DQN.ddqn import DDQNAgent
from environments.urban_environment.spawner import Spawner



def main(args):

    # Directory of current experiment
    experiment_dir = 'experiments/dqn_lstm/test1'

    # Load configuration
    config = Config()

    config.env = args.env

    config.hyperparameters = {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "sequence_length": 1,
        "buffer_size": 1000,
        "update_every_n_steps": 100,
        "min_steps_before_learning": 0,
        "epsilon_start": 1,
        "epsilon_end": 0.01,
        "epsilon_decay": 500,
        "discount_rate": 0.99,
        "tau": 0.01,
        "gradient_clipping_norm": 1.0,
    }
    
    config.use_cuda = True

    config.number_of_episodes = 300
    config.steps_per_episode = 500
    config.previous_episode = 0
    config.total_steps = 160000
    config.pre_train_steps = 100
    config.learing_frequency = 1

    config.checkpoint = True
    config.checkpoint_interval = 1
    config.checkpoint_dir = experiment_dir + '/checkpoints'

    config.log_dir = experiment_dir + '/logs'

    config.model_dir = experiment_dir + '/model'


    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 500
    config.frames = 160000
    config.use_cuda = True
    config.learning_rate = 1e-3
    config.max_buff = 1000
    config.update_tar_interval = 100
    config.batch_size = 128
    config.print_interval = 200
    config.log_interval = 200
    config.win_reward = 198     # CartPole-v0
    config.win_break = True


    # Initialize the environment
    env = gym.make('CartPole-v0')
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n

    # Initialize the agent
    agent = DDQNAgent(config)

    # Initialize spawner
    spawner = Spawner()

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
        
        trainer = Trainer(env, agent, spawner, config)
        trainer.load_checkpoint(args.checkpoint_file)

        try:
            trainer.retrain()

        except KeyboardInterrupt:
            try:
                #trainer.close()
                sys.exit(0)
            except SystemExit:
                #trainer.close()
                os._exit(0)

    elif args.test:
        tester = Tester(episodes, steps)
        tester.load_checkpoint(args.checkpoint_file)

        try:
            tester.retrain()

        except KeyboardInterrupt:
            try:
                tester.close()
                sys.exit(0)
            except SystemExit:
                tester.close()
                os._exit(0)


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