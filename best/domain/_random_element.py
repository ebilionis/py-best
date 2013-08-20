"""A class that defines a random element.

Author:
    Ilias Bilionis

Date:
    11/29/2012

"""


__all__ = ['RandomElement']


import numpy as np
import itertools as iter
from ..misc import BinaryTree


class RandomElement(BinaryTree):
    """Define a class that describes a random element.

    """

    # Number o components
    _s = None

    # The domain
    _domain = None

    # Number of components per dimension
    _k_of = None

    # Total number of dimensions
    _k = None

    # Number of output dimensions
    _q = None

    # Number of observations per dimension
    _n_of = None

    # Total number of observations
    _n = None

    # Observed input
    _X = None

    # Observed outputs
    _Y = None

    # Design matrix
    _H = None

    # Spliting component
    _split_comp = None

    # Splitting dimension
    _split_dim = None

    # Splitting point
    _split_pt = None

    # Purge data when splitting or not
    _purge = None

    # The scaled inputs
    _scaled_X = None

    # Scale X or not
    _scale_X = None

    # The probability of an element
    _pdf = None

    @property
    def pdf(self):
        """Get the probability of this element."""
        return self._pdf

    @property
    def scaled_X(self):
        """Get the scaled inputs."""
        return self._scaled_X

    @property
    def scale_X(self):
        """Get scale_X."""
        return self._scale_X

    @scale_X.setter
    def scale_X(self, value):
        """Set scale_X."""
        if not isinstance(value, bool):
            raise TypeError('A boolean must be used.')
        self._scale_X = value

    @property
    def purge(self):
        """Get the purge data property."""
        return self._purge

    @purge.setter
    def purge(self, value):
        """Set the purge data property."""
        if not isinstance(value, bool):
            raise TypeError('purge must be a boolean.')
        self._purge = value

    @property
    def s(self):
        """Get the number of components."""
        return self._s

    @property
    def domain(self):
        """Get the domain of the random element."""
        return self._domain

    @domain.setter
    def domain(self, value):
        """Set the domain of the random element."""
        if not isinstance(value, list):
            value = [value]
        for d in value:
            if not isinstance(d, np.ndarray):
                raise TypeError('All dub domains must be numpy arrays.')
            if not len(d.shape) == 2:
                raise ValueError('All sub domains must be two dimensional.')
            if not d.shape[1] == 2:
                raise ValueError('All sub domains must have two columns.')
            for i in xrange(d.shape[0]):
                if d[i, 0] > d[i, 1]:
                    raise ValueError('Left end point greater than right one.')
        if self.s is not None:
            if not self.s == len(value):
                raise ValueError('domain and data have different components.')
            for k, d in iter.izip(self.k_of, value):
                if not k == d.shape[0]:
                    raise ValueError(
                    'domain and data have different dimensions.')
        self._domain = value

    @property
    def k_of(self):
        """Get the number of dimensions per component."""
        return self._k_of

    @property
    def k(self):
        """Get the total number of dimensions."""
        return self._k

    @property
    def n_of(self):
        """Get the number of samples per dimension."""
        return self._n_of

    @property
    def n(self):
        """Get the total number of samples."""
        return self._n

    @property
    def q(self):
        """Get the total number of outputs."""
        return self._q

    @property
    def X(self):
        """Get the observed inputs."""
        return self._X

    @property
    def Y(self):
        """Get the observed outputs."""
        return self._Y

    @property
    def H(self):
        """Get the design matrix."""
        return self._H

    @property
    def split_comp(self):
        """Get the splitting component."""
        return self._split_comp

    @split_comp.setter
    def split_comp(self, value):
        """Set the splitting component."""
        if not isinstance(value, int):
            raise TypeError('split_comp must be an int.')
        if value < 0 or value >= self.s:
            raise ValueError('split_comp is out of bounds.')
        self._split_comp = value

    @property
    def split_dim(self):
        """Get the splitting dimension."""
        return self._split_dim

    @split_dim.setter
    def split_dim(self, value):
        """Set the splitting dimension.

        Precondition:
            The splitting component has already been set.
        """
        if not isinstance(value, int):
            raise TypeError('split_dim must be an int.')
        if value < 0 or value >= self.k_of[self.split_comp]:
            raise ValueError('split_dim is out of bounds.')
        self._split_dim = value

    @property
    def split_pt(self):
        """Get the splitting point."""
        return self._split_pt

    @split_pt.setter
    def split_pt(self, value):
        """Set the splitting point.

        Precondition:
            + The splitting component has already been set.
            + The splitting dimension has already been set.
        """
        if not isinstance(value, float):
            raise TypeError('split_pt must be a float.')
        dom = self.get_domain(self.split_comp, self.split_dim)
        if value < dom[0] or value > dom[1]:
            raise ValueError('split_pt is out of bounds.')
        self._split_pt = value

    def __init__(self, domain=None, purge=True, scale_X=False):
        """Initialize the object.

        Keyword Arguments:
            domain  ---     The desired domain for the element.
            purge   ---     Do you want to clean up every time you split an
                            element?

        Be careful with this one, because for efficiency reasons we do not copy
        the data. If you want, differnt elements to have different domains,
        then you need to make sure that you copy the data yourself.
        """
        super(RandomElement, self).__init__()
        if domain is not None:
            self.domain = domain
        self.purge = purge
        self.scale_X = scale_X
        self._pdf = 1.

    def _scale_input_comp(self, i, x):
        """Scale the input component i."""
        domain = self.get_domain(i)
        x_s = (x - domain[:, 0]) / (domain[:, 1] - domain[:, 0])
        return x_s

    def _scale_input_data(self, X):
        """Scale the input data."""
        X_s = []
        for i, x in iter.izip(range(self.s), X):
            X_s.append(self._scale_input_comp(i, x))
        return X_s

    def _scale_back_input_comp(self, i, x_s):
        """Scale back the input component i."""
        domain = self.get_domain(i)
        x = x_s * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
        return x

    def _scale_back_input_data(self, X_s):
        """Scale back the input data."""
        X = []
        for i, x_s in iter.izip(range(self.s), X_s):
            X.append(self._scale_back_input_comp(i, x_s))
        return X

    def set_data(self, X, H, Y):
        """Set the data of the element."""
        if isinstance(X, tuple):
            X = list(X)
        if isinstance(H, tuple):
            H = list(H)
        if not isinstance(X, list):
            X = [X]
        if not isinstance(H, list):
            H = [H]
        self._n_of = []
        self._k_of = []
        for x, h in iter.izip(X, H):
            if not isinstance(x, np.ndarray):
                raise TypeError('X must be a numpy array.')
            if not len(x.shape) == 2:
                raise ValueError('X must be two dimensional.')
            if not isinstance(h, np.ndarray):
                raise ValueError('H must be a numpy array.')
            if not len(h.shape) == 2:
                raise TypeError('H must be two dimensional.')
            n = x.shape[0]
            if not n == h.shape[0]:
                raise ValueError('Dimensions of X and H do not agree.')
            self._n_of += [n]
            self._k_of += [x.shape[1]]
        self._n = np.prod(self.n_of)
        self._k = np.prod(self.k_of)
        self._X = X
        self._H = H
        if not isinstance(Y, np.ndarray):
            raise TypeError('Y must be a numpy array.')
        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1), order='F')
        self._Y = Y
        self._q = Y.shape[1]
        self._s = len(self._X)
        if self.domain is None and self.is_root:
            domain = []
            for k in self.k_of:
                dom = np.ndarray((k, 2), order='F')
                dom[:, 0] = 0.
                dom[:, 1] = 1.
                domain += [dom]
            self.domain = domain
        if self.scale_X:
            self._scaled_X = self._scale_input_data(self._X)
        else:
            self._scaled_X = self._X

    def _add_data_helper(self, X, H, Y):
        """A helper for add_data."""
        if self.is_leaf:
            self.X[0] = np.vstack([self.X[0], X[0]])
            self.H[0] = np.vstack([self.H[0], H[0]])
            self._Y = np.vstack([self.Y, Y])
            self.n_of[0] += X[0].shape[0]
            self._n = np.prod(self.n_of)
            if self.scale_X:
                self.scaled_X[0] = np.vstack([self.scaled_X[0],
                            self._scale_input_comp(0, X[0])])
            else:
                self.scaled_X[0] = self.X[0]
            hyp = self.model.rg_to_hyp()
            self.model.set_data(self.scaled_X, self.H, self.Y)
            self.model.initialize(hyp)
            return [self]
        else:
            n_of = []
            (X_comp_left, H_comp_left, Y_left_list,
             X_comp_right, H_comp_right, Y_right_list) = self._split_data(
                     self.split_comp, self.split_dim, self.split_pt,
                     X, H, Y)
            updated = []
            if len(X_comp_left) > 0:
                X_left, H_left, Y_left = self._fix_data(self.split_comp,
                    X, H, X_comp_left, H_comp_left, Y_left_list)
                updated += self.left._add_data_helper(X_left, H_left, Y_left)
            if len(X_comp_right) > 0:
                X_right, H_right, Y_right = self._fix_data(self.split_comp,
                    X, H, X_comp_right, H_comp_right, Y_right_list)
                updated += self.right._add_data_helper(X_right, H_right, Y_right)
            return updated

    def add_data(self, X, H, Y):
        """Add more data to the element.

        If this is not a leaf element, then it adds the data to the right
        children.
        """
        return self._add_data_helper(X, H, Y)

    def _join_comp_to_full(self, data, comp, data_comp):
        """Convert a list of split data to the appropriate type accepted by
        set_data."""
        data_new = []
        data_comp = np.vstack(data_comp)
        for i in xrange(self.s):
            if i == comp:
                data_new += [data_comp]
            else:
                data_new += [data[i]]
        return data_new

    def _fix_data(self, comp, X, H, X_comp, H_comp, Y_list):
        X = self._join_comp_to_full(X, comp, X_comp)
        H = self._join_comp_to_full(H, comp, H_comp)
        Y = np.concatenate(Y_list, axis=comp)
        n = np.prod(Y.shape[:-1])
        Y = Y.reshape((n, self.q))
        return (X, H, Y)

    def _fix_child(self, child, comp, X, H, X_comp, H_comp, Y_list):
        """Fix the data relevant to this child."""
        (X, H, Y) = self._fix_data(comp, X, H, X_comp, H_comp, Y_list)
        child.set_data(X, H, Y)

    def _update_child_data(self, comp, i, indices, X, H, Y, X_comp, H_comp, Y_list):
        """Updates the data related to a child."""
        X_comp.append(X[comp][i, :])
        H_comp.append(H[comp][i, :])
        Y_list.append(Y[indices])

    def _get_domain_left(self, comp, dim):
        """Return the left end point of the element at (comp, dim)."""
        if self.is_root:
            return self.domain[comp][dim][0]
        if self.is_right:
            if comp == self.parent.split_comp and dim == self.parent.split_dim:
                return self.parent.split_pt
        return self.parent._get_domain_left(comp, dim)

    def _get_domain_right(self, comp, dim):
        """Return the right end point of the element at (comp, dim)."""
        if self.is_root:
            return self.domain[comp][dim][1]
        if self.is_left:
            if comp == self.parent.split_comp and dim == self.parent.split_dim:
                return self.parent.split_pt
        return self.parent._get_domain_right(comp, dim)

    def get_domain(self, comp, dim=None):
        """Return a tuple that describes the comp-dim part of the domain."""
        if dim is None:
            dom = np.ndarray((self.k_of[comp], 2), order='F')
            for k in range(self.k_of[comp]):
                dom[k, :] = self.get_domain(comp, dim=k)
            return dom
        else:
            return (self._get_domain_left(comp, dim),
                    self._get_domain_right(comp, dim))

    def _split_data(self, comp, dim, pt, X, H, Y):
        """Splits the data."""
        n_of = []
        for x in X:
            n_of += [x.shape[0]]
        q = Y.shape[1]
        s = len(X)
        Y = Y.reshape(n_of + [q])
        Y_left_list = []
        X_comp_left = []
        H_comp_left = []
        Y_right_list = []
        X_comp_right = []
        H_comp_right = []
        indices = [slice(None, None)] * s
        for i in xrange(n_of[comp]):
            indices[comp] = (i, )
            if X[comp][i, dim] <= pt:
                self._update_child_data(comp, i, indices, X, H, Y, X_comp_left,
                        H_comp_left, Y_left_list)
            if X[comp][i, dim] >= pt:
                self._update_child_data(comp, i, indices, X, H, Y, X_comp_right,
                        H_comp_right, Y_right_list)
        return (X_comp_left, H_comp_left, Y_left_list,
                X_comp_right, H_comp_right, Y_right_list)

    def split(self, comp, dim, loc=0.5):
        """Split the element.

        Arguments:
            dim         ---     The dimension on which to split.
            comp        ---     The component to split.

        Keyword Arguments:
            loc         ---     The location to split.

        Returns:
            True if succesful, False otherwise. The operation will fail if
            there are too few observations in the element.

        """
        self.split_comp = comp
        self.split_dim = dim
        my_dom = self.get_domain(comp, dim)
        self.split_pt = my_dom[0] + loc * (my_dom[1] - my_dom[0])
        self.left = RandomElement(scale_X=self.scale_X)
        self.left._pdf = 0.5 * self.pdf
        self.right = RandomElement(scale_X=self.scale_X)
        self.right._pdf = 0.5 * self.pdf
        (X_comp_left, H_comp_left, Y_left_list,
         X_comp_right, H_comp_right, Y_right_list) = self._split_data(
                comp, dim, self.split_pt, self.X, self.H, self.Y)
        if len(X_comp_left) == 0 or len(X_comp_right) == 0:
            left = self.left
            self._left = None
            left._parent = None
            del left
            right = self.right
            right._parent = None
            self._right = None
            del right
            return False
        self._fix_child(self.left, comp, self.X, self.H, X_comp_left,
                        H_comp_left, Y_left_list)
        self._fix_child(self.right, comp, self.X, self.H, X_comp_right,
                        H_comp_right, Y_right_list)
        self._clean_up_after_splitting()
        return True

    def _clean_up_after_splitting(self):
        """Clean up the element after splitting."""
        if self.purge:
            X = self._X
            X_s = self._scaled_X
            self._X = None
            del X
            del X_s
            Y = self._Y
            self._Y = None
            del Y
            H = self._H
            self._H = None
            del H

    def _str_array_dim(self, data, name):
        """Return the dimensions of an array as a string."""
        s = name + ': '
        if data is None:
            s += str(data)
        elif isinstance(data, list):
            for d in data:
                s += str(d.shape[0]) + 'x' + str(d.shape[1]) + ' '
        else:
            s += str(data.shape[0]) + 'x' + str(data.shape[1]) + ' '
        return s

    def _str_helper(self, padding):
        """A helper function for __str__()."""
        s = padding + self._str_array_dim(self.X, 'X') + '\n'
        s += padding + self._str_array_dim(self.H, 'H') + '\n'
        s += padding + self._str_array_dim(self.Y, 'Y') + '\n'
        s += padding + 'split_comp: ' + str(self.split_comp) + '\n'
        s += padding + 'split_dim: ' + str(self.split_dim) + '\n'
        s += padding + 'split_pt: ' + str(self.split_pt) + '\n'
        if self.has_left:
            s += '\n' + padding + 'Left:\n'
            s += self.left._str_helper(padding + ' ')
        if self.has_right:
            s += '\n' + padding + 'Right:\n'
            s += self.right._str_helper(padding + ' ')
        return s

    def __str__(self):
        """Return a string representation of the object."""
        s = 'Random Element\n'
        s += self._str_helper('')
        return s

    def _uniform_split_helper(self, comp, dim, depth):
        """Split the element depth times along comp and dim.

        Arguments:
            comp    ---     Component to split.
            dim     ---     Dimension to split.
            depth   ---     Depth to split.
        """
        if depth == 0:
            return
        if self.split(comp, dim):
            self.left._uniform_split_helper(comp, dim, depth - 1)
            self.right._uniform_split_helper(comp, dim, depth - 1)

    def uniform_split(self, comp, depth, dim=None):
        """Split the element depth times along comp and dim.

        """
        if dim is None:
            dim = range(self.k_of[comp])
        if isinstance(dim, int):
            dim = [dim]
        for d in dim:
            leaves = self.leaves
            if len(leaves) == 0:
                leaves = [self]
            for elm in leaves:
                elm._uniform_split_helper(comp, d, depth)

    def join_children(self):
        """Join the children of this node.

        All children of the current node are joined recursvively.
        """
        if not self.is_leaf:
            self.left.join_children()
            self.right.join_children()
            X = self.left.X
            X[self.split_comp] = np.vstack(
                                           [self.left.X[self.split_comp],
                                            self.right.X[self.split_comp]]
                                          )
            H = self.left.H
            H[self.split_comp] = np.vstack(
                                           [self.left.H[self.split_comp],
                                            self.right.H[self.split_comp]]
                                          )
            Y_left = self.left.Y.reshape(self.left.n_of + [self.left.q])
            Y_right = self.right.Y.reshape(self.right.n_of + [self.right.q])
            Y = np.concatenate([Y_left, Y_right], axis=self.split_comp)
            n = np.prod(Y.shape[:-1])
            Y = Y.reshape([n, self.q])
            self.set_data(X, H, Y)
            left = self.left
            left._parent = None
            self._left = None
            del left
            right = self.right
            right._parent = None
            self._right = None
            del right
            self._split_comp = None
            self._split_dim = None
            self._split_pt = None

    def _split_data_to_evaluate(self, X):
        """Split the data and return indices to be propagated further down the
        tree."""
        n_of = []
        for x in X:
            n_of += [x.shape[0]]
        s = len(X)
        indices_left = [slice(None, None)] * s
        indices_left[self.split_comp] = []
        indices_right = [slice(None, None)] * s
        indices_right[self.split_comp] = []
        for i in xrange(n_of[self.split_comp]):
            if X[self.split_comp][i, self.split_dim] <= self.split_pt:
                indices_left[self.split_comp] += [i]
            if X[self.split_comp][i, self.split_dim] >= self.split_pt:
                indices_right[self.split_comp] += [i]
        return indices_left, indices_right

    def _evaluate_child(self, child, idx, X, H, Y, V=None):
        """Evalutes a particular child."""
        X_r = X[:]
        X_r[self.split_comp] = X[self.split_comp][idx[self.split_comp], :]
        H_r = H[:]
        H_r[self.split_comp] = H[self.split_comp][idx[self.split_comp], :]
        Y_r = Y[idx + [slice(None, None)]][:]
        V_r = None
        if V is not None:
            V_r = V[idx + [slice(None, None)]][:]
            V_r = V_r.reshape((np.prod(V_r.shape[:-1]), V_r.shape[-1]))
        child(X_r, H_r, Y_r.reshape([np.prod(Y_r.shape[:-1]), Y_r.shape[-1]]),
                V=V_r)
        Y[idx + [slice(None, None)]] = Y_r[:]
        if V is not None:
            V[idx + [slice(None, None)]] = V_r[:]

    def __call__(self, X, H, Y, V=None):
        """Evaluate the model at a particular point.

        Precondition:
            A model has been attached to the element.
        """
        if self.is_leaf:
            X_s = self._scale_input_data(X)
            C = None
            if V is not None:
                C = np.ndarray((Y.shape[0], Y.shape[0]), order='F')
            self.model(X_s, H, Y, C=C)
            if V is not None:
                V[:] = np.tile(np.diag(C), V.shape[1]).reshape((V.shape[0],
                    V.shape[1]), order='F')
                V *= np.diag(self.model.Sigma)
        else:
            Y = Y.reshape([x.shape[0] for x in X] + [Y.shape[1]])
            if V is not None:
                V = V.reshape([x.shape[0] for x in X] + [V.shape[1]])
            idx_left, idx_right = self._split_data_to_evaluate(X)
            #print idx_left, idx_right
            if len(idx_left[self.split_comp]) > 0:
                self._evaluate_child(self.left, idx_left, X, H, Y, V=V)
            if len(idx_right[self.split_comp]) > 0:
                self._evaluate_child(self.right, idx_right, X, H, Y, V=V)

    def get_uncertainty(self, xi):
        """Get the uncertainty of this element."""
        if self.is_leaf:
            xi_s = self._scale_input_comp(0, xi)
            n = np.prod([x.shape[0] for x in self.X[1:]])
            q = self.Y.shape[1]
            Y = np.ndarray((n, q), order='F')
            C = np.ndarray((n, n), order='F')
            X = [xi_s] + self.scaled_X[1:]
            H = ([np.ones((1, 1), order='F')] +
                 [np.ones((x.shape[0], 1), order='F') for x in self.X[1:]])
            self.model(X, H, Y, C)
            return (np.trace(C) / n) * (np.trace(self.model.Sigma) / q)
        else:
            if self.split_comp == 0:
                if xi[:, self.split_dim] < self.split_pt:
                    return self.left.get_uncertainty(xi)
                else:
                    return self.right.get_uncertainty(xi)
            else:
                return (self.left.pdf * self.left.get_uncertainty(xi) +
                        self.right.pdf * self.right.get_uncertainty(xi))