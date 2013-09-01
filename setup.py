#!/usr/bin/env python


from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import os
import glob


setup(name='Bayesian Exploration Statistical Toolbox',
      author='Ilias Bilionis',
      version='0.0',
      ext_modules=[Extension('best.core.orthpol',
                             glob.glob(os.path.join('src', 'orthpol',
                                                    '*.f'))),
                   Extension('best.core.design',
                            glob.glob(os.path.join('src', 'design', '*.f90'))),
                   Extension('best.core.lapack',
                            glob.glob(os.path.join('src', 'lapack', '*.f')),
                            libraries=['lapack']),
                   Extension('best.core._ggsvd', ['src/_ggsvd.cpp'],
                             include_dirs=['src/', '/usr/include',
                                           '/usr/include/python2.7/numpy'],
                             libraries=['boost_python', 'lapack'])],
      packages=['best', 'best.core', 'best.misc', 'best.domain', 'best.gpc',
                'best.maps', 'best.linalg', 'best.random', 'best.rvm',
                'best.gp', 'best.design', 'best.uq', 'best.inverse'])
