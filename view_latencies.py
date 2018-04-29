#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sampling_latencies = scipy.io.mmread('output/sampling_latencies.mm')
    iteration_latencies = scipy.io.mmread('output/iteration_latencies.mm')
    # plt.subplot(211)
    plt.grid()
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Iteration')
    plt.semilogy()
    for i in range(1,sampling_latencies.shape[0]):
      plt.scatter(x=range(sampling_latencies.shape[1]), y=sampling_latencies[i,:])
    legend = []
    for i in range(1, 5):
        legend.append('Worker {}'.format(i))
    plt.legend(legend)
    # plt.subplot(212)
    # plt.grid()
    # plt.title('Time per iteration')
    # plt.xlabel('Time (seconds)')
    # plt.xlabel('Iteration')
    # plt.scatter(x=range(sampling_latencies.shape[1]), y=iteration_latencies[0,:])
    plt.tight_layout()
    plt.savefig('fig-latencies.png')
