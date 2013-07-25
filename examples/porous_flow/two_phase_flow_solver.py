"""A particular two phase flow solver.

Author:
    Ilias Bilionis

Date:
    3/24/2013
"""

from examples.porous_flow import *
from uq.maps import *
import math
import numpy as np
import scipy
#import matplotlib.pyplot as plt


class TwoPhaseFlowSolver(Function):
    
    # The underlying general solver
    _gen_solver = None
    
    # The underlying input model
    _input_model = None
    
    # The well locations
    _well_locations = None
    
    # Source magintude
    _source_magnitude = None
    
    # The total time
    _total_time = None
    
    # The time step to be used
    _time_step = None
    
    @property
    def gen_solver(self):
        return self._gen_solver
    
    @property
    def input_model(self):
        return self._input_model

    @property
    def well_locations(self):
        return self._well_locations
    
    @property
    def source_magnitude(self):
        return self._source_magnitude
    
    @property
    def total_time(self):
        return self._total_time
    
    @property
    def time_step(self):
        return self._time_step
    
    def __init__(self, input_model, well_locations=None,
                 source_magnitude=1.,
                 total_time=1000.,
                 time_step=10.):
        """Initialize the object."""
        self._input_model = input_model
        num_input = input_model['kle'].num_input
        nx = input_model['num_cells'][0]
        ny = input_model['num_cells'][1]
        nz = input_model['num_cells'][2]
        if well_locations is None:
            well_locations = [[0.5, 0.5, 0.5],
                              [0., 0., 0.5],
                              [0., 1., 0.5],
                              [1., 0., 0.5],
                              [1., 1., 0.5]]
        self._well_locations = np.array(well_locations)
        self._source_magnitude = source_magnitude
        self._gen_solver = TwoPhaseFlow(nx=nx, ny=ny, nz=nz, Lx=200., Ly=200., Lz=1.,
                                        s_wc=0.2, s_or=0.2, v_w=1e-4, v_o=1e-3, por=1e-3)
        self._total_time = total_time
        self._time_step = time_step
        num_output = int(math.ceil(total_time / time_step)) * (self.well_locations.shape[0] - 1)
        super(TwoPhaseFlowSolver, self).__init__(num_input, num_output)
    
    def __call__(self, x, Kr=None):
        """Evaluate the solver."""
        K = np.zeros((3, ) + self.gen_solver.n)
        if Kr is None:
            K[0, :, :, :] = np.exp(self.input_model['kle'](x)).reshape(self.gen_solver.n, order='F')
        else:
            K[0, :, :, :] = Kr.reshape(self.gen_solver.n, order='F')
        K[1, :, :, :] = K[0, :, :, :]
        K[2, :, :, :] = K[0, :, :, :]
        self._time_step = 10
        t, m_o = self.gen_solver.solve(K, self.total_time, self.well_locations,
                                 self.source_magnitude, Pt=self.time_step,
                                 St=.5 * self.time_step)
        y = m_o[:, 1:].flatten(order='F')
        #y = []
        #print t
        #for i in range(1, 5):
        #    f = scipy.interpolate.interp1d(t, m_o[:, i])
        #    y.append(f(np.linspace(0., 950, 100)))
        #y = np.hstack(y)
        return np.log(y)