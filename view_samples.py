#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

if __name__ == '__main__':
    legend = []
    for i in range(1,4):
        samples = []
        for part in glob('output/samples-{}-*.mm'.format(i)):
            samples.append(scipy.io.mmread(part))
        samples = np.hstack(samples)
        # samples = samples[:, np.arange(0, samples.shape[1]) % 15 == 0]

        # plt.subplot(211)
        # plt.plot(samples[0,:].T)
        # plt.subplot(212)
        # plt.subplot('22' + str(i))
        plt.scatter(samples[0, :], samples[1, :], alpha=0.01)
        # plt.xlim(-1, 2)
        # plt.ylim(-3, 3)

        legend.append("Worker {}".format(i))
    plt.grid()
    plt.legend(legend)
    plt.xlabel("$\\theta_1$")
    plt.ylabel("$\\theta_2$")
    plt.tight_layout()
    plt.savefig('fig-samples.png')
