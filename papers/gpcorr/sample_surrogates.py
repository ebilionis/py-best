import sys
sys.path.insert(0, '../..')
import numpy as np
import best
import matplotlib.pyplot as plt
import cPickle as pickle
import best


if __name__ == '__main__':
    data_file = sys.argv[1]
    with open(data_file, 'rb') as fd:
        X, H, Y = pickle.load(fd)
    particle_file = sys.argv[2]
    with open(particle_file, 'rb') as fd:
        pa = pickle.load(fd)
    # Get the hyper-parameters (ignore the uncertainty in the particles)
    pa.compute_all_means()
    ell1 = pa.mean['stochastics']['ell1']
    ell2 = pa.mean['stochastics']['ell2']
    g1 = pa.mean['stochastics']['g1']
    g2 = pa.mean['stochastics']['g2']
    hyp = np.hstack([ell1, ell2, g1, g2])
    print hyp
    mgp = best.gp.MultioutputGaussianProcess()
    mgp.set_data(X, H, Y)
    mgp.initialize(hyp)
    # Number of surrogates
    num_surrogate = 10
    # Number of design points to be used in sampling surrogates
    num_design = 1000
    X_design = best.design.latin_center(num_design, X[0].shape[1])
    H_design = np.ones((num_design, 1))
    for i in xrange(num_surrogate):
        s = mgp.sample_surrogate(X_design, H_design)
        out_file = 'ko_surrogate_s=%d_%d.pickle' % (X[0].shape[0], i)
        with open(out_file, 'wb') as fd:
            pickle.dump(s, fd, pickle.HIGHEST_PROTOCOL)
