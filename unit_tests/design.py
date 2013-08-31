"""
Unittests for best.design

Author:
    Ilias Bilionis

Date:
    8/31/2013
"""


if __name__ == '__main__':
    import fix_path


import best.design
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestDesign(unittest.TestCase):

    def test_latinize(self):
        x = np.random.rand(1000, 2)
        x_lhc = best.design.latinize(x)
        print x
        print x_lhc
        plt.plot(x[:, 0], x[:, 1], '+', x_lhc[:, 0], x_lhc[:, 1], 'o')
        plt.show()


if __name__ == '__main__':
    unittest.main()
