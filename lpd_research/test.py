# -*- coding: utf-8 -*-
# Copyright 2011 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
import simplex
import numpy

A = numpy.array([ [ 1,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [-1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [ 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [ 1,-1,-1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [-1, 1,-1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [-1,-1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                  [ 0, 1,-1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [ 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 1]])
b = numpy.array([0,0,2,0,0,0,0,0])
c = numpy.hstack( (numpy.ones((1,3)), numpy.zeros((1,8)) ))
B = numpy.array(range(3,11))
#B = numpy.array([0,4,5,6,7,8,9,10])
tab = simplex.SimplexTableau(A,b,c,B)
print(tab)
print(tab.x())
print(tab.B[tab.pivotOptions(0)])
tab.pivotOperation(0,0)
print(tab, tab.B)
zero_bases = [numpy.sort(B)]
