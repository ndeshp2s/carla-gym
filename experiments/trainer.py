import os
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
from experiments.util import EpsilonTracker


class Trainer:
    def __init__(self, env, agent, config: Config):
        self.env = env
        self.agent = agent
        self.config = config

        self.epsilon_decay = EpsilonTracker(self.config.hyperparameters["epsilon_start"], self.config.hyperparameters["epsilon_end"], 
            self.config.hyperparameters["min_steps_before_learning"], self.config.number_of_episodes * self.config.steps_per_episode)

        if not os.path.isdir(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        self.writer = SummaryWriter(log_dir = self.config.log_dir)

    def train(self, previous_episode = 0):
        losses = []
        rewards = []
        episode_reward = 0
        episode_number = 0
        total_steps = 0
        
        epsilon = self.config.hyperparameters["epsilon_start"]

        for ep_num in range(previous_episode + 1, self.config.number_of_episodes + 1):

            # Reset the environment and variables for new episode
            episode_reward = 0
            state = self.env.reset()

            for step_num in range(50):
                
                # Select action
                action = '2'

                # Execute step
                next_state, reward, done = self.env.step(action)

                # Add experience to memory of local network
                self.agent.add(state, action, reward, next_state, done)

                # Update parameters
                state = next_state
                episode_reward += reward

                # Performing learning if minumum required experiences gathered
                loss = 0

                if done:
                    break

                # epsilon update
                epsilon = self.epsilon_decay.update()


            # Print details of the episode
            print("------------------------------------------------")
            print("Episode: %d, Reward: %5f, Loss: %4f" % (ep_num, episode_reward, loss))
            print("------------------------------------------------")

            # Save episode reward and loss
            self.writer.add_scalar('Reward per episode', episode_reward, ep_num)

            # Save weights
            self.agent.save_model(self.config.model_dir)

            # Save checkpoints
            if not os.path.isdir(self.config.checkpoint_dir):
                os.makedirs(self.config.checkpoint_dir)

            checkpoint = {'state_dict': self.agent.local_network.state_dict(),
                        'optimizer': self.agent.optimizer.state_dict(),
                        'episode': ep_num,
                        'epsilon': epsilon}
            torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')
            
            # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
            #     self.agent.save_checkpoint()



    def close(self):
        self.env.close()




