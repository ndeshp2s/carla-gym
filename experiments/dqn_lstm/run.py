import argparse
# #import torch
# import os, sys

# import gym
# from gym import error, spaces


# from experiments.trainer import Trainer
# from experiments.tester import Tester
# from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv                                      
from experiments.config import Config
# from rl_agents.DQN.ddqncnnlstm import DDQNCNNLSTMAgent
# from environments.urban_environment.spawner import Spawner

import os, sys
import argparse
import gym

from experiments.config import Config
from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv
from rl_agents.DQN.ddqn import DDQNAgent 
from environments.urban_environment.spawner import Spawner
from experiments.trainer import Trainer
from experiments.tester import Tester
from collections import namedtuple


Experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])   
Site_Product=namedtuple("Site_Product", ["state", "action", "reward", "next_state", "done"])

def main(args):

    # Directory of current experiment
    experiment_dir = 'experiments/dqn_lstm/test15'

    # Load configuration
    config = Config()

    config.env = args.env

    config.hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "sequence_length": 10,
        "buffer_size": 10000,
        "update_every_n_steps": 1000,
        "min_steps_before_learning": 2000,
        "epsilon_start": 1,
        "epsilon_end": 0.1,
        "epsilon_before_learning": 1.0,
        "epsilon_decay": 5e-5,
        "discount_rate": 0.99,
        "tau": 0.001,
    }
    
    config.use_cuda = True

    config.number_of_episodes = 100
    config.steps_per_episode = 300
    config.previous_episode = 0
    config.total_steps = 50000
    config.pre_train_steps = 1
    config.learing_frequency = 1

    config.checkpoint = True
    config.checkpoint_interval = 1
    config.checkpoint_dir = experiment_dir + '/checkpoints'

    config.log_dir = experiment_dir + '/logs'

    config.model_dir = experiment_dir + '/model'
    config.spawner = True


    # Initialize the environment
    env = gym.make('Urban-v0')
    config.state_dim = env.observation_space.shape
    config.action_dim = env.action_space.n


    # Initialize the agent
    agent = DDQNAgent(config)

    # Initialize spawner
    spawner = Spawner()

    if args.train:
        trainer = Trainer(env, agent, spawner, config)
        #trainer.load_checkpoint(args.checkpoint_file)

        try:
            trainer.train()

        except KeyboardInterrupt:
            try:
                trainer.close()
                sys.exit(0)
            except SystemExit:
                trainer.close()
                os._exit(0)

    elif args.train_on_experiences:
        trainer = Trainer(env = None, agent = agent, spawner = None, config = config)

        # Get saved episodes
        import pandas as pd 
        PIK = "data.dat"
        data = pd.read_pickle(r"data.dat")

        try:
            trainer.train_on_experiences(data)

        except KeyboardInterrupt:
            try:
                trainer.close()
                sys.exit(0)
            except SystemExit:
                trainer.close()
                os._exit(0)


    elif args.experiences:
        trainer = Trainer(env, agent, spawner, config)

        try:
            experiences = trainer.add_experience()

            # store episodes
            data = []
            for e in experiences:
                e = Experience(state = e.state, action = e.action, reward = e.reward, next_state = e.next_state, done = e.done)
                data.append(e)

            import pickle
            PIK = "data.dat"
            with open(PIK, "wb") as f:
                pickle.dump(data, f)


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
                trainer.close()
                sys.exit(0)
            except SystemExit:
                trainer.close()
                os._exit(0)

    elif args.test:
        tester = Tester(env, agent, spawner, config)
        tester.load_checkpoint(args.checkpoint_file)

        try:
            tester.test()
            

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
    parser.add_argument('--train_on_experiences', dest='train_on_experiences', action='store_true', help='train model on experiences')
    parser.add_argument('--experiences', dest='experiences', action='store_true', help='experience model')
    parser.add_argument('--config_path', type=str, help='file with training parameters')
    parser.add_argument('--env', default='Urban-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    parser.add_argument('--checkpoint_file', default = None, type=str, help='if retrain, import the model and previous parameters')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='re-train model')
    args = parser.parse_args()

    main(parser.parse_args())