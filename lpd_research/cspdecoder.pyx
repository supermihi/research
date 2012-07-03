#!/usr/bin/python2
# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# Copyright 2012 Michael Helmling
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from lpdecoding.core import Decoder
from lpdecoding.codes import turbolike, interleaver
from lpdecoding.codes.ctrellis cimport Trellis 
from lpdecoding.algorithms.path cimport shortestPathScalarization 
from lpdecoding.codes.trellis import INPUT, PARITY
from libc.math cimport sqrt, abs 
import itertools, logging
import numpy
cimport numpy
cimport cython
import os
cdef double EPS = 1e-10
logging.basicConfig(level=logging.ERROR)
        
        
def zeroInConvexHull(points):
    import cplex as cpx
    cplex = cpx.Cplex()
    cplex.set_results_stream(None)

    lamb = ["lamb{0}".format(i) for i in range(len(points)) ]
    dim = len(points[0])
    x = ["x{0}".format(j) for j in range(dim)]
    cplex.variables.add(types=[cplex.variables.type.continuous]*len(points),
                        lb = [0]*len(points),
                        names = lamb)
    cplex.variables.add(types=[cplex.variables.type.continuous]*len(x),
                        lb = [-cpx.infinity]*len(x),
                        ub = [cpx.infinity]*len(x),
                        names = x)
    cplex.set_problem_type(cplex.problem_type.QP)

    cplex.linear_constraints.add(names = ["conv"],
                                 rhs = [1],
                                 senses = "E",
                                 lin_expr = [ cpx.SparsePair(ind = lamb, val = [1]*len(lamb))])
    cplex.linear_constraints.add(names = ["lx{0}".format(j) for j in range(len(x))],
                                 rhs = [0]*len(x),
                                 senses = "E"*len(x),
                                 lin_expr = [ cpx.SparsePair(ind = lamb + [x[j]], val = [points[i][j] for i in range(len(points))] + [-1]) for j in range(len(x))])
    cplex.objective.set_quadratic_coefficients( [ (xx,xx,2) for xx in x])
    cplex.objective.set_sense(cplex.objective.sense.minimize)
    cplex.solve()
    distance = sqrt(cplex.solution.get_objective_value())
    return distance < 1e-6, cplex.solution.get_values(x)

cdef inline double norm(numpy.double_t[:] a, int size):
    return sqrt(dot(a,a,size))

cdef inline double dot(numpy.double_t[:] a, numpy.double_t[:] b, int size):
    cdef double tmp = 0
    cdef int i
    for i in range(size):
        tmp += a[i]*b[i]
    return tmp
 
cdef void solveLT(numpy.double_t[:,:] a,
                  numpy.double_t[:] b,
                  numpy.double_t[:] x,
                  numpy.int_t[:] S,
                  int lenS):
    """Return x where x solves a[S,:S]x = b if a is a lower triangular matrix."""
    cdef:
        int i,j
        double tmp         
    for i in xrange(lenS):
        tmp = b[i]
        for j in xrange(i):
            tmp -= x[j]*a[S[i],S[j]]
        x[i] =  tmp / a[S[i],S[i]]
        #x[i] = (b[i] - numpy.dot(x[:i], a[i,:i])) / a[i,i]

cdef void solveUT(numpy.double_t[:,:] a,
                  numpy.double_t[:] b,
                  numpy.double_t[:] x,
                  numpy.int_t[:] S,
                  int lenS):
    """Return x where x solves ax = b if a is an upper triangular matrix."""
    cdef:
        int i,j
        double tmp
    for i in xrange(lenS-1, -1, -1):
        tmp = b[i]
        for j in xrange(i+1,lenS):
            tmp -= x[j]*a[S[i],S[j]]
        x[i] = tmp / a[S[i],S[i]]
        #x[i] = (b[i] - numpy.dot(x[i+1:], a[i,i+1:])) / a[i,i]

labelStr = { INPUT: "input", PARITY: "parity"}

cdef class NDFDecoder(Decoder):
    """Nondominated Facet Decoder."""
    def __init__(self, code):
        Decoder.__init__(self)
        self.code = code
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        
        for i, ((t1, s1, l1), (t2, s2, l2)) in enumerate(self.constraints):
            setattr(t1[s1], "g_{0}_index".format(labelStr[l1]), i)
            setattr(t1[s1], "g_{0}".format(labelStr[l1]), 1)
            #t1[s1].g_coeffs[l1].append( (i, 1) )
            
            setattr(t2[s2], "g_{0}_index".format(labelStr[l2]), i)
            setattr(t2[s2], "g_{0}".format(labelStr[l2]), -1)   
            #t2[s2].g_coeffs[l2].append( (i, -1) )

        
    cpdef solve(self):
        cdef:
            numpy.double_t[:] direction = numpy.zeros(self.k+1)
            numpy.ndarray[ndim=1, dtype=numpy.double_t] \
                w = numpy.empty(self.k+2),\
                v = numpy.empty(self.k+2),\
                Y = numpy.zeros(self.k+1),\
                X = numpy.empty(self.k+1)
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P = numpy.empty((self.k+1, self.k+2)), R = numpy.empty((self.k+2,self.k+2))
            numpy.ndarray[ndim=1, dtype=numpy.int_t] S = numpy.zeros(self.k+2, dtype=numpy.int)
            numpy.ndarray[ndim=1,dtype=numpy.uint8_t,cast=True] Sfree = numpy.ones(self.k+2,dtype=numpy.bool)
            double oldZ, z_d, a
            int i, k, lenS = 1
        
        self.code.setCost(self.llrVector)
        self.lstsq_time = self.sp_time = self.omg_time = 0
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        # find point with minimum cost in z-direction
        direction[-1] = 1
        self.solveScalarization(direction, Y)
        if norm(Y, self.k) < 1e-8:
            logging.info("initial point is feasible -> yeah!")
            self.solution = Y
            self.objectiveValue = Y[-1]
            return
        # set all but last component to 0 -> initial Y for nearest point algorithm
        for k in range(self.k):
            Y[k] = 0
        self.majorCycles = self.minorCycles = 0

        #oldZ = -1000
        oldZ = Y[-1]
        i = 0
        v= numpy.empty(self.k+1)
        while True:
            # initialize data arrays for NFA
            #*P[:,0] = -Y
            for k in range(self.k+1):
                P[k,0] = -Y[k]
            Sfree[0] = False
            for k in range(1, self.k+2):
                Sfree[k] = True
            w[0] = 1
            R[0,0] = sqrt(1+dot(Y, Y, self.k+1))
            lenS = 1
            S[0] = 0
            lenS = self.NPA(Y, P, S, Sfree, w, R, lenS, X)
            i += 1
            for k in range(self.k+1):
                v[k] = X[k] - Y[k]
            if norm(v, self.k+1) < 1e-8:
                #print('done(v)')
                break
            z_d = dot(X, v, self.k+1) / v[-1]
            if numpy.abs(z_d-oldZ) < 1e-8:
                #print('done')
                break
            Y[-1] = z_d 
        self.objectiveValue = Y[-1]
        #print('main iterations: {0}'.format(i))
        tmp = os.times()
        #print('total time: {}; least square solution time: {}; shortest path time: {}; omg time: {}'.format(tmp[0]+tmp[2]-time_a, self.lstsq_time, self.sp_time, self.omg_time))
        #print('major cycles: {0}    minor cycles: {1}'.format(self.majorCycles, self.minorCycles)) 
    
    cdef int NPA(self,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] Y,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P,
            numpy.ndarray[ndim=1, dtype=numpy.int_t] S,
            numpy.ndarray[ndim=1, dtype=numpy.uint8_t, cast = True] Sfree,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] w,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] R,
            int lenS,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] X) except -1:
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
        Y: reference point
        P: matrix of corral points (column-wise), if given
        w: weight vector, if P is given
        R: R from Method D, if P and w are given"""
        # type definitions
        cdef:
            numpy.ndarray[ndim=1, dtype=numpy.double_t] P_J, space1, space2, space3
            double normx, oldnormx, a, b, c, theta
            int majorCycle = 0, minorCycle = 0, \
                i, j, k, newIndex = 1, \
                IinS, I, Ip1, firstZeroIndex, firstZeroIndexInS
            bint cond = False
  
        P_J = numpy.empty(self.k+1)
        oldnormx = 1e20
        space1 = numpy.empty(self.k+2, dtype=numpy.double)
        space2 = numpy.empty(self.k+2, dtype=numpy.double)
        space3 = numpy.empty(self.k+2, dtype=numpy.double)
        while True:
            majorCycle += 1
            self.majorCycles += 1
            
            #*numpy.dot(P[:,S], w[S], X)
            for i in range(self.k+1):
                X[i] = 0
                for j in range(lenS):
                    X[i] += P[i,S[j]]*w[S[j]]
            #*
            normx = norm(X, self.k+1)
            if normx < 1e-8:
                break
            if normx > oldnormx+EPS:
                print("∥X∥ increased in cycle {0}: {1} > {2}".format(majorCycle, normx, oldnormx))
                break
            oldnormx = normx
            for k in range(self.k+1):
                P_J[k] = 0
            self.solveScalarization(X, P_J)
            P_J[self.k] -= Y[self.k]
            b = dot(P_J, P_J, self.k+1)
            if dot(X, P_J, self.k+1) > normx*normx - 1e-12*sqrt(self.k):
                #logging.debug('stop in 1 (c)')
                break
            #*newIndex = numpy.flatnonzero(Sfree)[0]
            newIndex = -1
            for k in range(self.k+2):
                if Sfree[k] == 1:
                    newIndex = k
                    break
            #*rhs= P[:,S].T*P_J
            for i in range(lenS):
                space2[i] = 0
                for j in range(self.k+1):
                    space2[i] += P[j,S[i]]*P_J[j]
            for k in range(lenS):
                space2[k] += 1
            solveLT(R.T, space2, space1, S, lenS)
            
            # augment R
            #*R[S,newIndex] = space1
            for k in range(lenS):
                R[S[k],newIndex] = space1[k]
                R[newIndex,S[k]] = 0
            c = dot(space1, space1, lenS)
            R[newIndex,newIndex] = sqrt(1 + b - c)
                
            #*P[:,newIndex] = P_J
            for k in range(self.k+1):
                P[k,newIndex] = P_J[k]
            # augment S
            #S = numpy.append(S, newIndex)
            S[lenS] = newIndex
            Sfree[newIndex] = False
            w[newIndex] = 0
            lenS += 1
            # check if R augmentation is correct
            # assert numpy.linalg.norm(numpy.dot(R[S][:,S].T,R[S][:,S]) - numpy.dot(Q.T,Q) - numpy.ones((lenS, lenS))) < EPS
            minorCycle = 0
            while True:
                minorCycle += 1
#                logging.debug('***STEP 2 [minor {0}]'.format(minorCycle))
                #*space3 = numpy.ones(lenS)
                for k in range(self.k+2):
                    space3[k] = 1
                solveLT(R.T, space3, space1, S, lenS) #space1= \bar u
#                ONE = numpy.dot(R[S[:lenS]][:,S[:lenS]].T, space1[:lenS])
#                TWO = space3[:lenS]
#                if not numpy.allclose(ONE, TWO):
#                    print('spast1')
                solveUT(R, space1, space2, S, lenS) #space2 = u
#                ONE = numpy.dot(R[S[:lenS]][:,S[:lenS]], space2[:lenS])
#                TWO = space1[:lenS]
#                if not numpy.allclose(ONE, TWO):
#                    print('spast2')
                # check
                #result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(lenS), Q)), e1)
                #u_correct = result[0]
                
                #*space1 = space2 / numpy.sum(space2) #space1=v
                # a = numpy.sum(space2) # remember: space3 = ones!
                a= dot(space2, space3, lenS)
#                if not (a < 10) and not (a > 10): # NaN test
#                    print('no no no')
#                    print(space1)
#                    print(space2)
#                    print(space3)
#                    print(S)
#                    print(lenS)
#                    print(a)
#                    print(R[S][:,S])
#                    return -1
                for k in range(lenS):
                    space1[k] = space2[k]/a
                    
                #*
                
                #*if numpy.all(space1 > EPS):
                cond = True
                for k in range(lenS):
                    if space1[k] <= EPS:
                        cond = False
                        break
                if cond:
                    #*w[S] = space1
                    for k in range(lenS):
                        w[S[k]] = space1[k]
                    break
                else:
                    self.minorCycles += 1
                    #*POS = numpy.flatnonzero(space1 <= EPS) # 3 (a) corrected                    
                    #*theta = min(1, numpy.max(space1[POS]/(space1[POS] - w[S][POS]))) # 3 (b) corrected
                    a = -1
                    for k in range(lenS):
                        if space1[k] <= EPS:
                            b = space1[k]/(space1[k]-w[S[k]])
                            if b > a:
                                a = b
                    if a < 1 and a != -1:
                        theta = a
                    else:
                        theta = 1
                    #*
                    #*w[S] = theta*w[S] + (1-theta)*space1 # 3 (c)
                    #*w[numpy.flatnonzero(w<=EPS)] = 0 # 3 (d)
                    #*firstZeroIndexInS = numpy.flatnonzero(w[S]==0)[0]
                    # index of S that will leave
                    firstZeroIndexInS = -1
                    for k in range(lenS):
                        i = S[k]
                        w[i] = theta*w[i] + (1-theta)*space1[k]
                        if w[i] <= EPS:
                            w[i] = 0
                            if firstZeroIndexInS == -1:
                                firstZeroIndexInS = k
                    #*
                    firstZeroIndex = S[firstZeroIndexInS]
                    IinS = firstZeroIndexInS
                    
                    if IinS < lenS-1:
                        i = S[IinS]
                        j = S[IinS+1]
                        #*tmp = R[S[IinS+1]].copy()
                        #*R[S[IinS+1]] = R[S[IinS]]
                        for k in range(lenS):
                            space1[k] = R[j,S[k]]
                            R[j,S[k]] = R[i,S[k]] 
                        
                    while IinS < lenS-1:
                        I = S[IinS]
                        Ip1 = S[IinS+1]
                        a = R[Ip1,Ip1]
                        b = space1[IinS+1]
                        c = sqrt(a*a+b*b)
                        #*first = (a*R[Ip1,S] + b*space1)/c
                        #*second = (-b*R[Ip1,S] + a*space1)/c
                        for k in range(lenS):
                            space2[k] = (a*R[Ip1,S[k]] + b*space1[k])/c # first
                            space3[k] = (-b*R[Ip1,S[k]] + a*space1[k])/c # second
                        #*R[Ip1,S] = first
                        for k in range(lenS):
                            R[Ip1,S[k]] = space2[k]
                        if IinS < lenS-2:
                            #*space1 = R[S[IinS+2],S].copy()
                            #*R[S[IinS+2],S] = second
                            for k in range(lenS):
                                space1[k] = R[S[IinS+2],S[k]]
                                R[S[IinS+2],S[k]] = space3[k]
                        IinS+=1
                    # shrink S
                    for i in range(firstZeroIndexInS, lenS-1):
                        S[i] = S[i+1]
                    Sfree[firstZeroIndex] = True
                    lenS -= 1

        for k in range(self.k+1):
            X[k] += Y[k]
        return lenS

    cdef void solveScalarization(self, numpy.double_t[:] direction, numpy.double_t[:] result):
        cdef:
            double lamb, c_result, time_a
        lamb = direction[-1]
        c_result = 0
        for enc in self.code.encoders:
            c_result += shortestPathScalarization(enc.trellis, lamb, direction, result)
        result[self.k] = c_result

class NDFInteriorDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self)
        self.code = code
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        for i, ((t1, s1, l1), (t2, s2, l2)) in enumerate(self.constraints):
            if not hasattr(t1[s1], "g_coeffs"):
                t1[s1].g_coeffs = { INPUT: [], PARITY: [] }
            t1[s1].g_coeffs[l1].append( (i, 1) )
                
            if not hasattr(t2[s2], "g_coeffs"):
                t2[s2].g_coeffs = { INPUT: [], PARITY: [] }
            t2[s2].g_coeffs[l2].append( (i, -1) )
