#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

setup(name='deep_ckern',
      version="0.1",
      author="Adrià Garriga-Alonso",
      author_email="ag919@cam.ac.uk",
      description="GP kernels equivalent to CNNs",
      license="Apache License 2.0",
      url="http://github.com/rhaps0dy/convnets-as-gps",
      ext_modules=[],
      packages=["deep_ckern"],
      test_suite='testing',
      tests_require=[],
      install_requires=[
          'gpflow>=1.2<1.3',
          'tqdm',
          'pickle_utils>=0.1',
          'scikit-learn>=0.19',
          ])
