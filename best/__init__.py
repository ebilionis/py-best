"""
.. module:: best
    :platform: Unix, Windows
    :synopsis: Implementation of various Bayesian algorithms for
    uncertainty propagation and stochastic inverse problems.

.. moduleauthor:: Ilias Bilionis <ebilionis@gmail.com>

The purpose of BEST is to serve as a platform for the development of fully
Bayesian algorithms to be applied in uncertainty propagation and stochastic
inverse problems.

Dependencies
------------
The package depends on the following external libraries:
    * Numpy
    * Scipy
    * mpi4py

Modules
-------
BEST is split in several (mostly) independent submodules:
    * :ref:`Linear Algebra <best.linalg>`
        Some linear algebra routines

"""

from domain import *
import misc
import linalg
import maps
import random
#from solver import Solver
#from binary_tree import BinaryTree
#from random_element import RandomElement
