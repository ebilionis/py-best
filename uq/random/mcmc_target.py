"""Defines the class of distributions that can be sampled via MCMC.

Basically, it is only required that the function can be evaluated up to
an arbitrary constant. This is enough to define a general MCMC scheme based
on a random walk. That is:
    p(q) \propto \pi(q).
If information about the gradients of log pi(q) is also available, then
metropolized Langevin dynamics may be used which enjoy accelerated convergence
properties.

All distributions that you wish to use as targets in an MCMC procedure must
inherit from MCMCTarget and overload the following methods accordingly.

Author:
    Ilias Bilionis

Date:
    3/7/2013

"""

import numpy as np


class MCMCTarget(object):
    """
