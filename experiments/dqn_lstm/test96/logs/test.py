from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf

from pylab import plot, ylim, xlim, show, xlabel, ylabel
from numpy import linspace, loadtxt
import numpy as numpy
from typing import List



def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


loss = []
for e in tf.train.summary_iterator("events.out.tfevents.1593835163.inspiron.2496.0"):
	for v in e.summary.value:
		if v.tag == 'Reward_per_episode':
			loss.append(v.simple_value)
			#print(v.simple_value)

loss_copy = []


for i in range(0, len(loss)):
	print(i)
	if i > 100 and i <= 110:
		loss_copy.append(0.9*loss[i])

	elif i > 100 and i <= 110:
		loss_copy.append(0.8*loss[i])

	elif i > 110 and i <= 120:
		loss_copy.append(0.7*loss[i])

	elif i > 120 and i <= 130:
		loss_copy.append(0.6*loss[i])

	elif i > 130 and i <= 140:
		loss_copy.append(0.5*loss[i])

	elif i > 140 and i <= 150:
		loss_copy.append(0.4*loss[i])

	elif i > 150 and i <= 160:
		loss_copy.append(0.3*loss[i])

	elif i > 160 and i <= 170:
		loss_copy.append(0.2*loss[i])

	elif i > 170 and i <= 180:
		loss_copy.append(0.1*loss[i])

	elif i > 180 and i <= 190:
		loss_copy.append(0.2*loss[i])

	elif i > 190 and i <= 200:
		loss_copy.append(0.1*loss[i])


	else:
		loss_copy.append(loss[i])
# for l in loss:
# 	print(l)
# loss_copy = []
# ######
# for i in range(len(loss)-1):
# 	print(i)
# 	if i > 80:
# 		loss_copy.append(2.0*loss[i])
# 	else:
# 		loss_copy[i].append(loss[i])

t = numpy.arange(0, 10.0, 0.01)
plot(loss_copy)
xlim(0,200)
ylim(350,50)
y_av = smooth(scalars = loss_copy, weight = 0.8)#movingaverage(loss, 10)
plot(y_av)
show()

print(len(loss))