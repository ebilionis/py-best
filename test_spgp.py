import numpy as np
import math
from uq.gp import *
import mdp
import pickle
from scipy.cluster.vq import *
import matplotlib.pyplot as plt

# load demo data set

#x = np.loadtxt('train_inputs', ndmin=2)
#y = np.loadtxt('train_outputs', ndmin=2)
#Y = y
#xtest = np.loadtxt('test_inputs', ndmin=2)
#print 'reading x'
#xa = np.loadtxt('two_phase_flow_many/pflow_X0.dat')
#print 'done'
#print 'reading y'
#Z = np.loadtxt('two_phase_flow_many/pflow_Y.dat')
#print 'done'
#pflow_data = {}
#pflow_data['X'] = xa
#pflow_data['Y'] = Z
#with open('pflow_data.bin', 'wb') as fd:
#    pickle.dump(pflow_data, fd)
with open('pflow_data.bin', 'rb') as fd:
    pflow_data = pickle.load(fd)
xa = pflow_data['X']
Z = pflow_data['Y']
Z_all = []
for i in range(xa.shape[0]):
    Z_all.append(Z[(i * 20):((i + 1) * 20), :].reshape((1, 20 * 8), order='F'))
Z = np.vstack(Z_all)
#print Z.shape
pca = mdp.nodes.PCANode(output_dim=0.9)
pca.train(Z)
pca.stop_training()
print 'PCA dim:', pca.get_output_dim()
Y = pca(Z)
ko = pca.get_output_dim()

n = 500
y = Y[:n, 0:1]
x = xa[:n, :]
n_all = 40000
nt = 0
xtest = xa[nt:, :]
ytest = Y[nt:, :]

# zero mean the data
me_y = np.mean(y)
y0 = y - me_y

N, dim = xa.shape

# number of zero points
M = 20

#code_book, distortion = kmeans(Y, M)
#code, dist = vq(Y, code_book)
#print code
#fig = plt.figure()
#for i in range(6):
#    for j in range(6):
#        ax = fig.add_subplot(6, 6, i * 6 + j)
#        plt.plot(xa[code == 0, i], xa[code == 0, j], 'k.')
#        plt.plot(xa[code == 1, i], xa[code == 1, j], 'r.')
#        plt.plot(xa[code == 2, i], xa[code == 2, j], 'b.')
#plt.show()
#quit()

# initialize pseudo-inputs to a random subset of training inputs
I = np.argsort(np.random.rand(N))
I = I[:M]

#I = range(M)
#xb_init = xa[I, :]
#xb_init = []
#for i in range(M):
#    idx = np.argmin(dist[code == i])
#    xb_i = xa[code == i, :][idx]
#    xb_init.append(xb_i)
#xb_init = np.vstack(xb_init)
xb_init, distortion = kmeans(xa, M)

def f(w, args):
    lw = spgp_lik(w, args['y'], args['x'], args['M'], d=args['d'],
                    compute_der=False)
    print args['c'], w[0], w[1], math.exp(w[-2]), math.exp(w[-1]), lw
    args['c'] += 1
    return lw

def fp(w, args):
    return spgp_lik(w, args['y'], args['x'], args['M'], d=args['d'],
                    compute_der=True)[1]

w_all = []
me_y_all = []
mu_all = []
for i in range(Y.shape[1]):
    y = Y[:n, i:(i + 1)]
    me_y = np.mean(y)
    y0 = y - me_y
    y_all = Y[:n_all, i:(i + 1)] - me_y
    x_all = xa[:n_all, :]
    data = {}
    data['x'] = x
    data['y'] = y0
    data['M'] = M
    data['d'] = 1e-6
    data['c'] = 0
    # initialize hyperparameters sensibly
    hyp_init = np.hstack(
        [-2. * np.log(((np.max(x, 0) - np.min(x, 0)).T) * 0.5), # log 1 / (length scales)** 2
         math.log(np.var(y)),                                # log size
         math.log(1e-6)                            # log noise
         ])
    print hyp_init
    # optimize hyper-parameters and pseudo-inputs
    w_init = np.hstack([xb_init.flatten(order='F'), hyp_init])
    w = w_init
    w, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = scipy.optimize.fmin_bfgs(
        f, w_init, fprime=fp, args=(data, ), full_output=True, disp=True)
    xb = w[:(M * dim)].reshape((M, dim), order='F')
    hyp = w[(M * dim):]
    [mu0, s2] = spgp_pred(y_all, x_all, xb, xtest, hyp)
    mu = mu0 + me_y
    #plt.plot(mu, ytest[:, i], 'k.')
    #plt.show()
    #yt = Y[nt:, i:(i + 1)]
    #plt.plot(yt, mu, 'k.')
    #plt.show()
    w_all.append(w)
    me_y_all.append(me_y)
    mu_all.append(mu.T)
mu_all = np.vstack(mu_all)

#w = w_init
xb = w[:(M * dim)].reshape((M, dim), order='F')
hyp = w[(M * dim):]

# PREDICTION
[mu0, s2] = spgp_pred(y0, x, xb, xtest, hyp)
# Add the mean back on
mu = mu0 + me_y
# Add noise
s2 += math.exp(hyp[-1])

#plt.plot(x, y, 'm.')

#plt.plot(xtest, mu, 'b')
#plt.plot(xtest, mu + 2. * np.sqrt(s2), 'r')
#plt.plot(xtest, mu - 2. * np.sqrt(s2), 'r')
#plt.show()
#quit()

#plt.plot(xb, -2.75 * np.ones(xb.shape), 'k+', markersize=20)
#plt.show()
#print np.exp(hyp)
#plt.plot(ytest, mu, 'k.')
#plt.show()

fig = plt.figure()
for i in range(6):
    for j in range(6):
        ax = fig.add_subplot(6, 6, i * 6 + j)
        plt.plot(Y[:, i], Y[:, j], 'k.')
        yi = mu_all[i:(i + 1), :].T
        yj = mu_all[j:(j + 1), :].T
        plt.plot(yi, yj, 'ro')
plt.show()

for j in range(n_all, n_all + 20):
    #plt.plot(mu_all[j:(j+1), 0], ytest[j:(j + 1), 0], 'k.')
    z_t = pca.inverse(mu_all[:, j:(j + 1)].T).T
    #z_t = pca.inverse(Y[j:(j + 1), :]).T
    plt.plot(z_t, 'r')
    z_true = np.atleast_2d(Z[nt + j, :])
    plt.plot(z_true[0, :], 'b')
    #plt.plot(pca.inverse(pca(z_true))[0, :], 'g')
    #plt.axis([0., 160., -5., .1])
    plt.show()
plt.show()