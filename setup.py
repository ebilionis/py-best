#!/usr/bin/env python


from distutils.core import setup


setup(name='Bayesian Exploration Statistical Toolbox',
      author='Ilias Bilionis',
      version='0.0',
      packages=['best', 'best.core', 'best.misc', 'best.domain',
                'best.maps', 'best.linalg', 'best.random', 'best.rvm',
                'best.gp', 'best.design', 'best.uq', 'best.inverse'])
