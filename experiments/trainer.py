import os
import torch
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config
from experiments.util import EpsilonTracker, DataVisualization

DEBUG = 0
class Trainer:
    def __init__(self, env, agent, config: Config):
        self.env = env
        self.agent = agent
        self.config = config
        self.epsilon_decay = EpsilonTracker(epsilon_start = self.config.hyperparameters["epsilon_start"], 
                                            epsilon_final = self.config.hyperparameters["epsilon_end"], 
                                            warmup_steps = self.config.hyperparameters["min_steps_before_learning"], 
                                            total_steps = self.config.total_steps, 
                                            epsilon_decay = self.config.hyperparameters["epsilon_decay"],
                                            total_episodes = self.config.number_of_episodes*self.config.steps_per_episode)

        if not os.path.isdir(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        self.writer = SummaryWriter(log_dir = self.config.log_dir)

        # Parameters for re-training
        self.previous_episode = 0
        self.epsilon = 0

        self.start_learning = False

        # self.epsilon_by_step = lambda step_num: self.config.hyperparameters["epsilon_end"] + (self.config.hyperparameters["epsilon_start"] - self.config.hyperparameters["epsilon_end"]) * math.exp(
        #     -1. * step_num / self.config.hyperparameters["epsilon_decay"])

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_step = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)



    def train(self):

        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset()

        for step in range(1, self.config.frames + 1):
            epsilon = self.epsilon_by_step(step)
            self.writer.add_scalar('Epsilon decay', epsilon, step)

            action = self.agent.pick_action(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add_experience(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0
            if self.agent.buffer.__len__() > self.config.hyperparameters["batch_size"]:
                loss = self.agent.learn(step = step)
                #print(loss)
                losses.append(loss)
                self.writer.add_scalar('Loss per step', loss, step)

            if step % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (step, np.mean(all_rewards[-10:]), loss, ep_num))



            if done:
                # # Save episode reward
                self.writer.add_scalar('Reward per episode', episode_reward, ep_num)
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))



                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    #self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break


        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')





    # def train(self, previous_episode = 0):
    #     losses = []
    #     rewards = []
    #     episode_reward = 0
    #     episode_number = 0
    #     total_steps = 0

    #     data_vis = DataVisualization(x_min = 0, x_max = 30, y_min = -5, y_max = 25)
        
    #     epsilon = self.config.hyperparameters["epsilon_start"]

    #     for ep_num in range(previous_episode + 1, self.config.number_of_episodes + 1):

    #         # Reset the environment and variables for new episode
    #         episode_reward = 0
    #         state = self.env.reset()
    #         self.spawner.reset()

    #         for step_num in range(self.config.steps_per_episode):
    #             #data_vis.display(state)
    #             if DEBUG:
    #                 input('Enter to continue: ')
                
    #             # Select action
    #             action = self.agent.pick_action(state, epsilon)

                
    #             # Execute step
    #             next_state, reward, done = self.env.step(action)

    #             # Add experience to memory of local network
    #             self.agent.add(state, action, reward, next_state, done)

    #             # Update parameters
    #             state = next_state
    #             episode_reward += reward
    #             total_steps += 1

    #             # Execute spwner step
    #             self.spawner.run_step()


    #             # Performing learning if minumum required experiences gathered
    #             loss = 0
    #             if total_steps > self.config.pre_train_steps and total_steps % self.config.learing_frequency == 0:
    #                 loss = self.agent.learn(batch_size = self.config.hyperparameters["batch_size"])

    #                 self.writer.add_scalar('Loss per step', loss, total_steps)


    #             if done:
    #                 self.spawner.destroy_all()
    #                 break

    #             # epsilon update
    #             # Only after a few initial steps
    #             if total_steps > self.config.pre_train_steps:
    #                 #epsilon = self.epsilon_decay.update(total_steps)
    #                 epsilon = self.epsilon_decay.update()

    #             self.writer.add_scalar('Epsilon decay', epsilon, total_steps)


    #         # Print details of the episode
    #         print("------------------------------------------------")
    #         print("Episode: %d, Reward: %5f, Loss: %4f" % (ep_num, episode_reward, loss))
    #         print("------------------------------------------------")

    #         # # Save episode reward
    #         self.writer.add_scalar('Reward per episode', episode_reward, ep_num)

    #         # # Save weights
    #         # self.agent.save_model(self.config.model_dir)

    #         # Save checkpoints
    #         if not os.path.isdir(self.config.checkpoint_dir):
    #             os.makedirs(self.config.checkpoint_dir)

    #         if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
    #             checkpoint = {'state_dict': self.agent.local_network.state_dict(),
    #                         'optimizer': self.agent.optimizer.state_dict(),
    #                         'episode': ep_num,
    #                         'epsilon': epsilon,
    #                         'total_steps': total_steps}
    #             torch.save(checkpoint, self.config.checkpoint_dir + '/model_and_parameters.pth')
            
    #         # # if self.config.checkpoint and ep_num % self.config.checkpoint_interval == 0:
    #         # #     self.agent.save_checkpoint()



    def close(self):
        self.env.close()


    def load_checkpoint(self, file = None, checkpoint_dir = None):
        checkpoint = torch.load(self.config.checkpoint_dir + '/' + file)

        # Load network weights and biases
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.previous_episode = checkpoint['episode']
        self.config.hyperparameters["epsilon_start"] = checkpoint['epsilon']

        self.agent.local_network.train()
        self.agent.target_network.train()

    def retrain(self):
        self.train(self.previous_episode)







