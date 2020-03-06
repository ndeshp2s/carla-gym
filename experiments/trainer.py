from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
from experiments.util import EpsilonTracker


class Trainer:
    def __init__(self, env, agent, config: Config):
        self.env = env
        self.agent = agent
        self.config = config

        self.epsilon_decay = EpsilonTracker(self.config.hyperparameters["epsilon_start"], self.config.hyperparameters["epsilon_end"], self.config.hyperparameters["min_steps_before_learning"], self.config.number_of_episodes * self.config.steps_per_episode)

        self.writer = SummaryWriter(log_dir = self.config.log_dir)

    def train(self, previous_episode = 0):
        losses = []
        rewards = []
        episode_reward = 0
        episode_number = 0
        total_steps = 0
        
        epsilon = self.config.hyperparameters["epsilon_start"]

        for e in range(previous_episode + 1, self.config.number_of_episodes + 1):
            state = self.env.reset()

            for s in range(self.config.steps_per_episode):
            
                action = '2'
                next_state, reward, done = self.env.step(action)

                self.agent.add(state, action, reward, next_state)

            #   loss = self.agent.step(state, action, reward, next_state, done)

            #   if loss is not None:
            #       losses.append(loss)
            #       self.writer.add_scalar('Loss per step', loss, s) # value, step

            #   if done:
            #       episode_reward = 0
            #       break

            #   state = next_state

            #   episode_reward += reward

            #   # epsilon update
            #   epsilon = self.epsilon_decay(s)


            episode_number += 1

            print("Episode: %4d, Reward: %5f, Loss: %4f" % episode_number, episode_reward, loss)

            self.writer.add_scalar('Reward per episode', episode_reward, episode_number)

            # Save weights
            if self.config.checkpoint and e % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint()

    def close(self):
        self.env.close()




