#!/usr/bin/env python
import sys, os

import gym
import environments

env = gym.make('RuleBased-v0')

env.reset()

try:
    while True:
    	env.step()
    	speed = env.get_ego_speed()
    	print(speed)


except KeyboardInterrupt:
    try:
        env.close()
        sys.exit(0)
    except SystemExit:
        env.close()
        os._exit(0)