import random
import numpy as np
import torch
from collections import deque, namedtuple



# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = deque(maxlen = capacity)

#     def add_experience(self, state, action, reward, next_state, done):

#         # if len(self.buffer) >= self.capacity:
#         #     self.buffer.pop(0)
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return state, action, reward, next_state, done

#     def __len__(self):
#         return len(self.buffer)

class ReplayBuffer():
    def __init__(self, buffer_size):

        self.buffer = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):

        experiences = random.sample(self.buffer, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return experiences


    # def get_batch(self, batch_size = 0, sequence_length = 0):
    #     experiences = random.sample(self.memory, batch_size)

    #     batch = []
    #     for e in experiences:
    #         point = np.random.randit(0, len(e) + 1 - sequence_length)
    #         batch.append(e[point:point + sequence_length])
    #     return batch

    # def __len__(self):
    #     return len(self.memory)

# class ReplayBuffer(object):
#     """Replay buffer to store past experiences that the agent can then use for training data"""
    
#     def __init__(self, buffer_size, batch_size, seed):

#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     def add_experience(self, states, actions, rewards, next_states, dones):
#         """Adds experience(s) into the replay buffer"""
#         if type(dones) == list:
#             assert type(dones[0]) != list, "A done shouldn't be a list"
#             experiences = [self.experience(state, action, reward, next_state, done)
#                            for state, action, reward, next_state, done in
#                            zip(states, actions, rewards, next_states, dones)]
#             self.memory.extend(experiences)
#         else:
#             experience = self.experience(states, actions, rewards, next_states, dones)
#             self.memory.append(experience)
   
#     def sample(self, num_experiences=None, separate_out_data_types=True):
#         """Draws a random sample of experience from the replay buffer"""
#         experiences = self.pick_experiences(num_experiences)
#         if separate_out_data_types:
#             states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
#             return states, actions, rewards, next_states, dones
#         else:
#             return experiences
            
#     def separate_out_data_types(self, experiences):
#         """Puts the sampled experience into the correct format for a PyTorch neural network"""
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
#         return states, actions, rewards, next_states, dones
    
#     def pick_experiences(self, num_experiences=None):
#         if num_experiences is not None: batch_size = num_experiences
#         else: batch_size = self.batch_size
#         return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.buffer)