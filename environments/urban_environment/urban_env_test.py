#!/usr/bin/env python
import sys, os
import random
import time

import gym
import environments

env = gym.make('Urban-v0')

env.reset()


try:
    while True:
    	env.step(0)
    	time.sleep(1.0)

except KeyboardInterrupt:
    try:
        env.close()
        sys.exit(0)
    except SystemExit:
        env.close()
        os._exit(0)