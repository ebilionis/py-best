import sys
sys.path.insert(0, '../..')
import numpy as np
import best
import matplotlib.pyplot as plt
import cPickle as pickle


if __name__ == '__main__':
    filename = sys.argv[1]
    var_name = sys.argv[2]
    with open(filename, 'rb') as fd:
        pa = pickle.load(fd)
    best.smc.hist(pa, var_name)
    plt.show()
