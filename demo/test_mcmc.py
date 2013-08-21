"""Test the MCMC class.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""


if __name__ == '__main__':
    import fix_path


import numpy as np
import math
from best.random import *
import matplotlib.pyplot as plt


class SampleTarget(LikelihoodFunction):
    """A sample target distribution."""

    def __init__(self):
        super(SampleTarget, self).__init__(1)

    def __call__(self, x):
        k = 3.
        t = 2.
        if x[0] < 0.:
            return -1e99
        else:
            return (k - 1.) * math.log(x[0]) - x[0] / t


if __name__ == '__main__':
    target = SampleTarget()
    x_init = np.ones(1)
    proposal = RandomWalkProposal(dt=5.)
    mcmc = MarkovChainMonteCarlo(target=target, proposal=proposal,
                                 store_samples=True,
                                 verbose=True,
                                 output_frequency=1000)
    mcmc.initialize(x_init)
    mcmc.sample(steps=100000)
    samples = [mcmc.samples[i][0] for i in range(len(mcmc.samples))]
    plt.hist(samples, 100)
    plt.show()