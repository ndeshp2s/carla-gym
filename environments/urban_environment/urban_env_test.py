#!/usr/bin/env python
import sys, os
import random
import time
import numpy as np

import gym
import environments

from utils.pylive import live_plotter

PLOT = True

if PLOT:
    size = 100
    x_vec = np.linspace(0, 1,size + 1)[0:-1]
    y_vec = [0 for _ in range(len(x_vec))] 
    line1 = []
    line2 = []


env = gym.make('Urban-v0')

env.reset()


try:
    while True:
        action = '0'
        #action = random.randint(0, 3)
        action = input('Enter action: ')

#         if rand == 0:
#             action = '0'
#         elif rand == 1:
#             action = '1'
#         elif rand == 2:
#             action = '2'


        env.step(action)
        # env.render()
        # env.tick()

        if PLOT:
            speed = env.get_ego_speed()
            y_vec[-1] = speed
            line1 = live_plotter(x_vec, y_vec, line1)
            y_vec = np.append(y_vec[1:], 0.0)

except KeyboardInterrupt:
    try:
        env.close()
        sys.exit(0)
    except SystemExit:
        env.close()
        os._exit(0)