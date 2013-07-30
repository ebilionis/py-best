"""A create a 2D upscaling map.

Author:
    Ilias Bilionis
    
Date:
    12/15/2012
    
"""

import numpy as np
from uq.maps import create1DUpscalingMap


def create2DUpscalingMap(x, X):
    """Create a 2D upscaling map.
    
    Arguments:
        x   ---     A 2D tuple describing the fine grid.
        X   ---     A 2D tuple describing the coarse grid.
    
    Returns:
        M   ---     A matrix representing the map.
    
    """
    if not isinstance(x, tuple):
        x = (x, )
    if not isinstance(X, tuple):
        X = (X, )
    M = (create1DUpscalingMap(x[0], X[0]), )
    if len(x) == 1 and len(X) == 2:
        M += (create1DUpscalingMap(x[0], X[1]), )
    elif len(x) == 2 and len(X) == 1:
        M += (create1DUpscalingMap(x[1], X[0]), )
    elif len(x) == 2 and len(X) == 2:
        M += (create1DUpscalingMap(x[1], X[1]), )
    else:
        M += (M[0], )
    return np.kron(M[0], M[1])