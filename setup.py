#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import unicode_literals
import os
import fnmatch
import io
import re
import sys
from os.path import join

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


def makeExtensions():
    """Returns an Extension object for the given submodule of lpdecoding."""

    sources = []
    for root, dirnames, filenames in os.walk('lpdecres'):
        for filename in fnmatch.filter(filenames, '*.pyx'):
            sources.append(str(join(root, filename)))
    directives = {'embedsignature': True}
    if '--profile' in sys.argv:
        directives['profile'] = True
        sys.argv.remove('--profile')
    extensions = cythonize(sources, include_path=[np.get_include()],
                           compiler_directives=directives)
    for e in extensions:
        e.include_dirs += [np.get_include()] # the above does not work on windows
    if '--no-glpk' in sys.argv:
        extensions = [e for e in extensions if 'glpk' not in e.libraries]
        sys.argv.remove('--no-gurobi')
    if '--no-gurobi' in sys.argv:
        extensions = [e for e in extensions if 'gurobi' not in e.libraries]
    else:
        for e in extensions:
            if 'gurobi60' in e.libraries:
                try:
                    e.library_dirs = [join(os.environ['GUROBI_HOME'], 'lib')]
                    e.include_dirs = [join(os.environ['GUROBI_HOME'], 'include')]
                except KeyError:
                    print('GUROBI_HOME not set. Compiling gurobi extensions might fail. Use '
                          ' --no-gurobi to disable them.')
    return extensions


setup(
    name='lpdecres',
    version='0.1',
    author='Michael Helmling',
    author_email='helmling@uni-koblenz.de',
    install_requires=['numpy', 'cython'],
    include_package_data=True,
    ext_modules=makeExtensions(),
    packages=find_packages(exclude=['lpd_research']),
)
