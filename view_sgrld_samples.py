#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

if __name__ == '__main__':
    plt.grid()
    plt.semilogy()
    plt.semilogx()
    for i in range(1,5):
        samples = []
        for part in glob('./samples-{}-*.mm'.format(i)):
            samples.append(scipy.io.mmread(part))
        samples = np.hstack(samples)

        # convert from extended mean parameterization to simplex parameters
        samples /= samples.sum(axis=0)

        # plt.subplot(211)
        # plt.plot(samples.T)
        # plt.subplot(212)
        # plt.subplot('22' + str(i))
        plt.scatter(samples[0, :], samples[1, :], alpha=1.0)
    plt.savefig('fig-samples.png')
