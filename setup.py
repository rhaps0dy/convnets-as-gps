#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from pkg_resources import parse_version

install_requires = [
    'gpflow>=1.2<1.3',
    'tqdm>=4<5',
    'pickle_utils>=0.1<1',
    'scikit-learn>=0.19<1',
    # Packages from here on are packages that GPflow also requires, but that
    # `deep_ckern` imports directly and uses.
    'numpy>=1.10<2',
    'scipy>=0.18<1',
    'pandas>=0.18.1<1',
    'absl-py>=0.4<0.5',
    ]

# Copied from GPflow's setup.py
min_tf_version = '1.5.0'
tf_cpu = 'tensorflow>={}'.format(min_tf_version)
tf_gpu = 'tensorflow-gpu>={}'.format(min_tf_version)

# Only detect TF if not installed or outdated. If not, do not do not list as
# requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip).

try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf
    if parse_version(tf.VERSION) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning) as e:
    # Add TensorFlow to dependencies to trigger installation/update
    install_requires.append(tf_cpu)

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
      install_requires=install_requires)
