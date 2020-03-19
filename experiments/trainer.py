import os
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
from experiments.util import EpsilonTracker


class Trainer:
    def __init__(self, env, agent, spawner, config: Config):
        self.env = env
        self.agent = agent
        self.config = config
        self.epsilon_decay = EpsilonTracker(epsilon_start = self.config.hyperparameters["epsilon_start"], epsilon_final = self.config.hyperparameters["epsilon_end"], 
            warmup_steps = self.config.hyperparameters["min_steps_before_learning"], total_steps = self.config.total_steps, epsilon_decay = self.config.hyperparameters["epsilon_decay"])

        if not os.path.isdir(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        self.writer = SummaryWriter(log_dir = self.config.log_dir)

        # Parameters for re-training
        self.previous_episode = 0
        self.epsilon = 0

        self.start_learning = False

        self.spawner = spawner

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
            self.spawner.reset()

            for step_num in range(self.config.steps_per_episode):
                
                # Select action
                action = self.agent.pick_action(state, epsilon)

                # Execute spwner step
                self.spawner.run_step()
                
                # Execute step
                next_state, reward, done = self.env.step(action)

                # Add experience to memory of local network
                self.agent.add(state, action, reward, next_state, done)

                # Update parameters
                state = next_state
                episode_reward += reward
                total_steps += 1

                # Performing learning if minumum required experiences gathered
                loss = 0
                if total_steps > self.config.pre_train_steps and total_steps % self.config.learing_frequency == 0:
                    loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"])

                    self.writer.add_scalar('Loss per step', loss, total_steps)


                if done:
                    break

                # epsilon update
                # Only after a few initial steps
                if total_steps > self.config.pre_train_steps:
                    epsilon = self.epsilon_decay.update(total_steps)


                self.writer.add_scalar('Epsilon decay', epsilon, total_steps)



            # # Print details of the episode
            # print("------------------------------------------------")
            # print("Episode: %d, Reward: %5f, Loss: %4f" % (ep_num, episode_reward, loss))
            # print("------------------------------------------------")

            # # Save episode reward and loss
            # self.writer.add_scalar('Reward per episode', episode_reward, ep_num)

            # # Save weights
            # self.agent.save_model(self.config.model_dir)

            # # Save checkpoints
            # if not os.path.isdir(self.config.checkpoint_dir):
            #     os.makedirs(self.config.checkpoint_dir)

            # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
            #     checkpoint = {'state_dict': self.agent.local_network.state_dict(),
            #                 'optimizer': self.agent.optimizer.state_dict(),
            #                 'episode': ep_num,
            #                 'epsilon': epsilon,
            #                 'total_steps': total_steps}
            #     torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')
            
            # # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
            # #     self.agent.save_checkpoint()



    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load netwrok weights and biases
        self.agent.local_network = checkpoint['model_state_dict']
        self.agent.target_network = checkpoint['model_state_dict']
        self.agent.optimizer = checkpoint['optimizer_state_dict']
        self.previous_episode = checkpoint['episode']
        self.config.hyperparameters["epsilon_start"] = checkpoint['epsilon']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode)







