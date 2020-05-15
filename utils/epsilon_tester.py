import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np

epsilon_start = 1.0
epsilon_end = 0.01
total_steps = 40000

decay = 5e-5


steps = 0

epsilon_decay = lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * frame_idx * decay)

writer = SummaryWriter()
epsilon = 1.0
while steps < total_steps:
	epsilon = epsilon_decay(steps)
	#epsilon -= (1.0 - 0.1)/total_steps
	writer.add_scalar('Epsilon decay', epsilon, steps)
	steps += 1

	print(steps)

writer.close()

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# x = range(100)
# for i in x:
#     writer.add_scalar('y=2x', i * 2, i)
# writer.close()