import sys
sys.path.insert(0, '../..')
import numpy as np
import best
import matplotlib.pyplot as plt
import cPickle as pickle
import best


if __name__ == '__main__':
    prefix = sys.argv[1]
    num_surrogates = int(sys.argv[2])
    Xt = np.linspace(0, 1., 100).reshape((100, 1))
    Ht = np.ones(Xt.shape)
    for i in xrange(num_surrogates):
        print 'surrogate', i
        with open(prefix + '_%d.pickle' % i, 'rb') as fd:
            s, u = pickle.load(fd)
        m, v = s.get_statistics(Xt, Ht, num_samples=1000)
        plt.plot(v, 'r')
    plt.show()
