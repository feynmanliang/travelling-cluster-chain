#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

if __name__ == '__main__':
    plt.grid()
    plt.semilogy()
    num_workers = len(set(
        x[:16] for x in glob('output/samples-*-*.mm')))
    legend = []
    for i in range(1, num_workers+1):
        # Perplexities here are recorded once per minibatch
        perplexities = scipy.io.mmread('output/perplexities-{}.mm'.format(i))
        plt.plot(perplexities)
        legend.append("Worker {}".format(i))

    plt.legend(legend)
    # plt.xlabel('Trajectory number')
    plt.xlabel('Iteration over dataset')
    plt.ylabel('Perplexity (nats)')
    plt.savefig('fig-perplexities.png'.format(i))
