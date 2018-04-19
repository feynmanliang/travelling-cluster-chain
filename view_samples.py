#!/usr/bin/env python3
#vim:set et sw=4 ts=8:

import scipy.io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for i in range(1,5):
        samples = scipy.io.mmread('./samples-{}.mm'.format(i))

        # plt.subplot(211)
        # plt.plot(samples.T)
        # plt.subplot(212)
        # plt.subplot('22' + str(i))
        plt.scatter(samples[0, :], samples[1, :], alpha=0.01)
    plt.savefig('fig-samples.png'.format(i))
