#!/usr/bin/env python3

import scipy
import numpy as np

from matplotlib import pyplot as plt

N_SAMPLES = 10000 * 100

x = []
y = []

for i in [100, 250, 1000, 10000]:
  i_lat = scipy.io.mmread('iteration_latencies-{}.mm'.format(i))
  s_lat = np.max(scipy.io.mmread('sampling_latencies-{}.mm'.format(i)), axis=0)
  x.append(N_SAMPLES / i)
  y.append(1 - np.sum(s_lat, axis=0) / np.sum(i_lat, axis=1)[0])

plt.scatter(x,y)
plt.grid()
plt.xlabel('Trajectory length')
plt.ylabel('% time communicating')
plt.savefig('traj-length-slowdown.png')
