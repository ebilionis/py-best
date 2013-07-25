"""A porous flow solver on a square domain with a pressure gradient.

Author:
    Ilias Bilionis

Date:
    1/25/2013

"""

import math
import os
import numpy as np
from uq.maps import Function


# A constant defining the location of the pflow_p_grad
# executable relative to the directory right before examples
EXAMPLES_PFLOW_PFLOW_P_GRAD_EXEC = './examples/pflow/pflow_p_grad'


class PflowPGrad(Function):
    
    """An interface to the porsous flow solver."""
    
    # The executable of the solver
    _executable = None
    
    # The number of cells per dimension
    _num_cells = None
    
    # Output prefix
    _prefix = None
    
    # Permeability model
    _permeability_model = None
    
    # The spatial point on which the model should be evaluated
    _X = None
    
    # The name of the file containing X
    _X_file = None
    
    # Counts the samples taken by this solver
    _count = None
    
    # Pressure difference
    _p_diff = None
    
    # Flow direction
    _flow_direction = None

    # Clean up after running
    _clean_up = None

    @property
    def clean_up(self):
        """Get the clean up flag."""
        return self._clean_up

    @clean_up.setter
    def clean_up(self, value):
        """Set the clean up flag."""
        assert isinstance(value, bool)
        self._clean_up = value
    
    @property
    def executable(self):
        """Get the executable."""
        return self._executable
    
    @executable.setter
    def executable(self, value):
        """Set the executable."""
        if not isinstance(value, str):
            raise TypeError('Provide the name of the executable.')
        if not os.path.exists(value):
            raise RuntimeError('Couldn\'t find the executable ' + value)
        self._executable = value
    
    @property
    def domain(self):
        """Get the computational domain."""
        return self._domain
    
    @domain.setter
    def domain(self, value):
        """Set the computational domain."""
        if not isinstance(value, np.ndarray):
            raise TypeError('The computational domain must be a numpy array.')
        if len(value.shape) == 2:
            raise ValueError('The domain must be two dimensional.')
        assert value.shape[0] == 2 and value.shape[1] == 2, 'Domain must be 2 x 2'
        for i in range(value.shape[0]):
            assert value[i, 0] <= value[i, 1], 'Domain error: a_i > b_i'
        self._domain = value
    
    @property
    def num_cells(self):
        """Get the number of cells per dimension."""
        return self._num_cells
    
    @property
    def prefix(self):
        """Get the prefix."""
        if self._count == 0:
            return self._prefix
        else:
            return self._prefix + '_' + str(self._count)
    
    @prefix.setter
    def prefix(self, value):
        """Set the prefix."""
        assert isinstance(value, str)
        self._prefix = value
    
    @property
    def K_file(self):
        """Get the permeability file."""
        return self.prefix + '_K.dat'
    
    @property
    def permeability_model(self):
        """Get the permeability model."""
        return self._permeability_model
    
    @property
    def X(self):
        """Get the spatial points of evaluation."""
        return self._X
    
    @property
    def X_file(self):
        """Get the file containing X."""
        return self._X_file
    
    @property
    def p_diff(self):
        """Get the pressure differnce."""
        return self._p_diff
    
    @p_diff.setter
    def p_diff(self, value):
        """Set the pressure difference."""
        assert isinstance(value, float)
        self._p_diff = value
    
    @property
    def flow_direction(self):
        """Get the flow direction."""
        return self._flow_direction
    
    @flow_direction.setter
    def flow_direction(self, value):
        """Set the flow direction."""
        assert isinstance(value, str)
        assert value == 'lr' or value == 'bt'
        self._flow_direction = value
    
    @property
    def response_file(self):
        """Get the response file."""
        return self.prefix + '_response.dat'
    
    def __init__(self, permeability_model, X,
                 prefix='pflow', clean_up=True,
                 executable=EXAMPLES_PFLOW_PFLOW_P_GRAD_EXEC):
        """Initialize the object.
        
        Arguments:
            permeability_model  --- Specify the permeability model
            X                   --- The spatial point on which you
                                    wish to evaluate the output.
        
        Keyword Arguments:
            prefix      ---     The output prefix
            executable  ---     The executable
            clean_up    ---     If true, deletes all output files
                                of the solver.
        """
        self._count = 0
        self.clean_up = clean_up
        assert hasattr(permeability_model, 'num_input')
        assert hasattr(permeability_model, 'num_output')
        assert hasattr(permeability_model, '__call__')
        self._permeability_model = permeability_model
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        assert X.shape[1] == 2
        self._X = X
        super(PflowPGrad, self).__init__(self.permeability_model.num_input,
                                         self.X.shape[0],
                                         name='PflowPGrad')
        self._num_cells = int(math.sqrt(self.permeability_model.num_output))
        self.prefix = prefix
        self.executable = executable
        self._X_file = self.prefix + '_X.dat'
        np.savetxt(self.X_file, self.X)
        self.p_diff = 1.
        self.flow_direction = 'lr'
    
    def __call__(self, x):
        """Evaluate the solver."""
        self._count = np.random.rand()
        # 1. Construct the permeability
        K = np.exp(self.permeability_model(x)).reshape((self.num_cells,
                                                        self.num_cells),
            order='F')
        # 2. Write the permeability to a file
        np.savetxt(self.K_file, K)
        # 3. Create the command for the solver
        cmd = self.executable
        cmd += ' -n ' + str(self.num_cells)
        cmd += ' -K ' + self.K_file
        cmd += ' -p ' + self.prefix
        cmd += ' -x ' + self.X_file
        cmd += ' --p-diff ' + str(self.p_diff)
        cmd += ' --flow-direction ' + self.flow_direction
        cmd += ' > /dev/null'
        # 4. Run the solver command
        os.system(cmd)
        # 5. Read the output
        y = np.loadtxt(self.response_file)
        if self.clean_up:
            os.remove(self.K_file)
            os.remove(self.response_file)
            os.remove(self.prefix + '_K.pvd')
            os.remove(self.prefix + '_K000000.vtu')
            os.remove(self.prefix + '_p.pvd')
            os.remove(self.prefix + '_p000000.vtu')
            os.remove(self.prefix + '_u.pvd')
            os.remove(self.prefix + '_u000000.vtu')
        return y[:, 2]
    
    def _to_string(self, pad):
        """Get a string representation of the object."""
        s = super(PflowPGrad, self)._to_string(pad) + '\n'
        s += pad + 'prefix: ' + self.prefix + '\n'
        s += pad + 'X_file: ' + self.X_file + '\n'
        s += pad + 'K_file: ' + self.K_file + '\n'
        s += pad + 'response_file: ' + self.response_file + '\n'
        s += pad + 'num. cells: ' + str(self.num_cells) + '\n'
        s += pad + 'p diff.: ' + str(self.p_diff) + '\n'
        s += pad + 'flow dir.: ' + self.flow_direction + '\n'
        s += pad + 'exec. file: ' + self.executable + '\n'
        s += pad + 'perm. model: ' + str(self.permeability_model)
        return s
