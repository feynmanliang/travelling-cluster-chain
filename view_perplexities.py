#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.grid()
    # plt.semilogy()
    for i in range(1, 5):
        # Perplexities here are recorded once per minibatch
        perplexities = scipy.io.mmread('./perplexities-{}.mm'.format(i))
        plt.plot(perplexities)


    # plt.xlabel('Trajectory number')
    plt.xlabel('Iteration over dataset')
    plt.ylabel('Perplexity (nats)')
    plt.savefig('fig-perplexities.png'.format(i))
