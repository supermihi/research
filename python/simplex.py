# -*- coding: utf-8 -*-
# Copyright 2011 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy
from numpy import dot, vstack, hstack
from numpy.linalg import inv
class SimplexTableau:
    
    def __init__(self, A, b, c, B):
        '''Initialize the simplex tableau of the LP min c^T x: Ax=b with respect to the given basis B.'''
        
        m,n = A.shape
        c = c.reshape((1,n))
        b = b.reshape((m,1))
        invA_B = inv(A[:,B])
        p = dot(c[0,B],invA_B)
        self.T = vstack( (hstack( (c - dot(dot(c[0,B],invA_B),A), [[-dot(dot(c[0,B],invA_B),b[:,0])]])),
                          hstack((dot(invA_B, A), dot(invA_B, b) )) ))
        
        self.B = B
        self.m, self.n = m, n
    
    def x_B(self):
        return self.T[1 + numpy.argsort(numpy.argsort(self.B)),self.n]
    
    def x(self):
        x = numpy.zeros(self.n)
        for i,b_i in enumerate(self.B):
            x[b_i] = self.T[1 + i, self.n]
        return x
    
    def pivotOptions(self, index):
        '''returns the list of possible basis positions that could leave if the nonbasic at *index* would enter.
        
        Note: The leaving basic variable indexes are obtained by B[result]'''
        return numpy.nonzero(self.T[1:,index] > 0)
    
    def pivotOperation(self, entering_index, position):
        '''Carries out a pivoting operation on the tableau; after that, x[entering_index] will be the *position*-th basis
        variable.
        
        Note the different indexing: *entering_index* is an x-index, i.e. the position in the variable vector, whereas
        *position* denotes the position in the (ordered) basis; the x-index of the leaving basis would then be
        B[position]. Thus leaving_position equals one plus the pivot row in the tableau.'''
        pivot = self.T[1+position, entering_index]
        assert pivot > 0
        self.T[1+position] /= pivot
        for i in range(self.m+1):
            if i != 1+position:
                self.T[i,:] = self.T[i,:] - self.T[i,entering_index]*self.T[1+position]
        self.B[position] = entering_index
        
    def __str__(self):
        return str(self.T)