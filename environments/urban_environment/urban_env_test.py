#!/usr/bin/env python
import sys, os
import random

import gym
import environments

env = gym.make('Urban-v0')

env.reset()


try:
    while True:
    	print('Do nothing')


except KeyboardInterrupt:
    try:
        env.close()
        sys.exit(0)
    except SystemExit:
        env.close()
        os._exit(0)