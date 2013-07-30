"""
Load the parts of the library that are written in C++.

Author:
    Ilias Bilioni

Date:
    11/26/2012
    31/1/2013   (Added incomplete Cholesky wrapper pstrf and zero_tri_part)
"""

# This is needed for loading the mkl library
#import sys
#import ctypes
#_old_rtld = sys.getdlopenflags()
#sys.setdlopenflags(_old_rtld | ctypes.RTLD_GLOBAL)
##from _kron import *
##del sys
#del ctypes
#from _lhs import *
#from _dpstrf import pstrf
#from _dpstrf import zero_tri_part
