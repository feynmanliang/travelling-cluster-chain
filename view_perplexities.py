#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    N_docs = 1740
    N_mb = 5
    plt.grid()
    plt.semilogy()
    for i in range(1, 5):
        # Perplexities here are recorded once per minibatch
        perplexities = scipy.io.mmread('./perplexities-{}.mm'.format(i))
        # plt.plot(perplexities)

        # average over entire batch
        pp = []
        for batch in range(int(perplexities.shape[0] / (N_docs / N_mb))):
            pp.append(perplexities[int(N_docs/N_mb*batch):int(N_docs/N_mb*(batch+1))].mean())
        plt.plot(pp)

    # plt.xlabel('Trajectory number')
    plt.xlabel('Iteration over dataset')
    plt.ylabel('Perplexity (nats)')
    plt.savefig('fig-perplexities.png'.format(i))
