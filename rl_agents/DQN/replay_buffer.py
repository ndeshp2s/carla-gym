import random
import numpy as np
import torch
from collections import deque, namedtuple

class ReplayBuffer():
    def __init__(self, buffer_size):

        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)


    def sample(self, batch_size = 32):
        experiences = random.sample(self.memory, k = batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return experiences#(states, actions, rewards, next_states, dones)

        #return experiences

    # def get_batch(self, batch_size = 0, sequence_length = 0):
    #     experiences = random.sample(self.memory, batch_size)

    #     batch = []
    #     for e in experiences:
    #         point = np.random.randit(0, len(e) + 1 - sequence_length)
    #         batch.append(e[point:point + sequence_length])
    #     return batch

    def __len__(self):
        return len(self.memory)