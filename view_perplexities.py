#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.grid()
    plt.semilogy()
    for i in range(1, 5):
        perplexities = scipy.io.mmread('./perplexities-{}.mm'.format(i))
        # plt.plot(perplexities)

        # average over entire batch
        pp = []
        for batch in range(int(perplexities.shape[0] * 50 / 1740)):
            pp.append(perplexities[int(1740/50*batch):int(1740/50*(batch+1))].mean())
        print(pp)
        plt.plot(pp)

    # plt.xlabel('Trajectory number')
    plt.xlabel('Iteration over dataset')
    plt.ylabel('Perplexity (nats)')
    plt.savefig('fig-perplexities.png'.format(i))
