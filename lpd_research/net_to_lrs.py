# -*- coding: utf-8 -*-
# Copyright 2011 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy
from lpdecoding import matrix

def net_to_lrs(naim, rhs, i):
    """Output the network constraints in LRS H-representation format."""
    cons, vars = naim.shape
    print('network')
    print('H-representation')
    print('linearity {0} {1}'.format(i, " ".join(map(str, range(1, i+1)))))
    print('begin')
    print('{0} {1} rational'.format(cons + vars, vars + 1))
    # constraints (network and equality)
    mat = numpy.hstack((-rhs, naim))
    for row in range(i):
        print(" ".join(map(str, mat[row,:])))
    for row in range(i, mat.shape[0]):
        print(" ".join(map(str, mat[row,:])))
        print(" ".join(map(str, -mat[row,:])))
    # sign constraints
    mat = numpy.hstack( (numpy.zeros((vars, 1), dtype = numpy.int), numpy.eye(vars, dtype = numpy.int) ))
    print(matrix.strBinary(mat))
    print('end')
    