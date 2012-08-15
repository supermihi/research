# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# Copyright 2012 Michael Helmling
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains the constrained shortest path LP decoder for turbo-like codes."""

from __future__ import division, print_function

from collections import OrderedDict
import logging

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, abs

from lpdecoding.core import Decoder
from lpdecoding.codes import turbolike, interleaver
from lpdecoding.codes.ctrellis cimport Trellis 
from lpdecoding.algorithms.path cimport shortestPathScalarization 
from lpdecoding.codes.trellis import INPUT, PARITY
from lpdecoding.utils cimport StopWatch
 

logging.basicConfig(level=logging.ERROR)

DEF TIMING = True # compile-time variable; if True, time of various steps is measured and output
DEF CREATE_SOLUTION = True

cdef double EPS = 1e-10

cdef inline double norm(np.double_t[:] a, int size):
    """computes the L2-norm of a[:size]"""
    return sqrt(dot(a,a,size))

cdef inline double dot(np.double_t[:] a, np.double_t[:] b, int size):
    """computes np.dot(a[:size], b[:size]) efficiently"""
    cdef:
        double tmp = 0
        int i
    for i in range(size):
        tmp += a[i]*b[i]
    return tmp
 
cdef void solveLT(np.double_t[:,:] a,
                  np.double_t[:] b,
                  np.double_t[:] x,
                  np.int_t[:] S,
                  int lenS):
    """Ensure that x[:lenS] solves numpy.dot(a[S[:lenS][:,S[:lenS], x[:lenS]) = b[:lenS] if a is a lower triangular matrix."""
    cdef:
        int i,j
        double tmp         
    for i in xrange(lenS):
        tmp = b[i]
        for j in xrange(i):
            tmp -= x[j]*a[S[i],S[j]]
        x[i] =  tmp / a[S[i],S[i]]
        #x[i] = (b[i] - np.dot(x[:i], a[i,:i])) / a[i,i]

cdef void solveUT(np.double_t[:,:] a,
                  np.double_t[:] b,
                  np.double_t[:] x,
                  np.int_t[:] S,
                  int lenS):
    """Ensure that x[:lenS] solves numpy.dot(a[S[:lenS][:,S[:lenS], x[:lenS]) = b[:lenS] if a is an upper triangular matrix."""
    cdef:
        int i,j
        double tmp
    for i in xrange(lenS-1, -1, -1):
        tmp = b[i]
        for j in xrange(i+1,lenS):
            tmp -= x[j]*a[S[i],S[j]]
        x[i] = tmp / a[S[i],S[i]]

labelStr = { INPUT: "input", PARITY: "parity"}

cdef class CSPDecoder(Decoder):
    """Constrained Shortest Path Decoder."""
    
    def __init__(self, code, name=None, maxMajorCycles=0):
        """Initialize the decoder for a *code* and name it *name*.
        
        Optionally you may supply *maxMajorCycles* to limit the maximum number of major cycles
        (nearest point calculations) performed. A value of 0 means no limit.
        """
           
        Decoder.__init__(self, code)
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        self.maxMajorCycles = maxMajorCycles
        if name is None:
            name = "CSPDecoder" 
        self.name = name
        #=======================================================================
        # Set coefficients and indices for constraints g_i on the involved
        # trellis graphs.
        #=======================================================================
        for i, ((trellis1, segment1, label1), (trellis2, segment2, label2)) \
                in enumerate(self.constraints):
            setattr(trellis1[segment1], "g_{0}_index".format(labelStr[label1]), i)
            setattr(trellis1[segment1], "g_{0}".format(labelStr[label1]), 1)
            
            setattr(trellis2[segment2], "g_{0}_index".format(labelStr[label2]), i)
            setattr(trellis2[segment2], "g_{0}".format(labelStr[label2]), -1)   

        #=======================================================================
        # Initialize arrays and matrices
        #=======================================================================
        self.space1 = np.empty(self.k+2, dtype=np.double)
        self.space2 = np.empty(self.k+2, dtype=np.double)
        self.space3 = np.empty(self.k+2, dtype=np.double)
        self.X = np.empty(self.k+1)
        self.Y = np.empty(self.k+1)
        self.P = np.empty((self.k+1, self.k+2))
        self.R = np.empty((self.k+2,self.k+2))
        self.S = np.empty(self.k+2, dtype=np.int)
        self.Sfree = np.empty(self.k+2,dtype=np.bool)
        self.w = np.empty(self.k+2)
        self.RHS = np.empty((self.k+2, self.k+2))
        self.P_J = np.empty(self.k+1)
        self.direction = np.empty(self.k+1)
        self.timer = StopWatch()
        
    cpdef solve(self, np.int_t[:] hint=None):
        """Solve the LP problem by a number of nearest point problems in constraints space.
        """
        
        cdef:
            double old_ref = 0, ref = 0, b_r, norm_a_r, delta_r
            int k, mainIterations = 0
            np.double_t[:] direction = self.direction, Y = self.Y, X = self.X           
        
        self.code.setCost(self.llrVector)
        self.majorCycles = self.minorCycles = 0
        IF TIMING:
            self.lstsq_time = self.sp_time = self.cho_time = self.r_time = 0
        
        #  find path with minimum cost
        for k in range(self.k):
            direction[k] = 0
        direction[self.k] = 1
        self.solveScalarization(direction, Y)
        self.lenS = 1
        #  test if initial path is feasible
        if norm(Y, self.k) < 1e-8:
            self.solution = np.asarray(Y).copy()
            self.objectiveValue = Y[self.k]
            self.mlCertificate = self.foundCodeword = True
        else:
            
            self.resetData()
            
            while self.maxMajorCycles == 0 or self.majorCycles < self.maxMajorCycles:
                #===================================================================
                # Main iterations: push up reference point until we hit the LP solution            
                #===================================================================
                mainIterations += 1
                #  adjust last coordinate of points in P
                delta_r = old_ref - ref
                IF TIMING:
                    self.timer.start()
                self.updateData(delta_r)
                IF TIMING:
                    self.cho_time += self.timer.stop()
                # save current objective value
                old_ref = Y[self.k]
                #  call nearest point algorithm
                self.NearestPointAlgorithm()
                b_r = Y[self.k]*X[self.k] - dot(X, X, self.k+1)
                norm_a_r = -b_r + Y[self.k]*(Y[self.k] -X[self.k])
                if norm_a_r < 1e-7:
                    break
                #  compute b_r = intersection of hyperplane with axis
                ref = b_r / (Y[self.k]
                              - X[self.k])
                if ref-old_ref < 1e-7:
                    break
                Y[self.k] = ref 
                self.objectiveValue = ref
            self.mlCertificate = self.foundCodeword = ( np.max(self.w[self.S[:self.lenS]]) > 1-1e-7 )
        
        IF TIMING:
            if "sp_time" not in self.stats:
                self.stats["lstsq_time"] = self.stats["sp_time"] = self.stats["cho_time"] = self.stats["r_time"] = 0
            self.stats["lstsq_time"] += self.lstsq_time
            self.stats["sp_time"] += self.sp_time
            self.stats["cho_time"] += self.cho_time
            self.stats["r_time"] += self.r_time
        if self.lenS == 1:
            try:
                self.stats["vertexSolutions"] += 1
            except KeyError:
                self.stats["vertexSolutions"] = 1
        else:
            try:
                self.stats["faceDimension"] += self.lenS -1
            except KeyError:
                self.stats["faceDimension"] = self.lenS
        try:
            self.stats["mainIterations"] += mainIterations
        except KeyError:
            self.stats["mainIterations"] = mainIterations
        try:
            self.stats["majorCycles"] += self.majorCycles
        except KeyError:
            self.stats["majorCycles"] = self.majorCycles
    
    cdef void NearestPointAlgorithm(self):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
        Y: reference point
        P: matrix of corral points (column-wise)
        S: vector of used indexes for P, w and R
        Sfree: bool array storing which indexes are free in S
        w: weight vector
        R: R for method D
        lenS: number of indexes
        X: result array
        """
        # ************ explanation of the optimizations *********
        # This algorithm is highly optimized using Cython, it basically
        # runs in C completely. To cope with the varying number of points
        # in the corral, we use fixed-size arrays everywhere. The integer
        # lenS holds the actual size of the corral, and the first lenS entries
        # of S are the indexes used; i.e., S[:lenS] corresponds to the S in
        # the paper, and Q would be P[:, S[:lenS] ].
        #
        # Similarly, the 2-d array R is used in such a way that what is
        # R[i,j] in the paper is R[S[i], S[j]] in our code.
        #
        # This tweaks admittedly do not increase readability of the code, but
        # completely eliminate any memory reallocation, such that the speed
        # of this implementation should be hard to beat. If there are any
        # unobvious optimizations, lines starting with " #* " contain the original
        # code, and the optimized version is written below that comment line.
        # type definitions
        cdef:
            double normx, oldnormx, a, b, c
            int i, j, k
            int newIndex = 1, I
            int lenS = self.lenS
            
            np.double_t[:,:] R = self.R, \
                             P = self.P
            np.double_t[:] space1 = self.space1, \
                           space2 = self.space2, \
                           space3 = self.space3, \
                           w = self.w, \
                           P_J = self.P_J, \
                           X = self.X, \
                           Y = self.Y
            np.int_t[:] S = self.S
            np.ndarray[ndim=1,dtype=np.uint8_t,cast=True] Sfree = self.Sfree
  
        oldnormx = 1e20
        while True:
            self.majorCycles += 1
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (a): X = P[S] * w
            #*np.dot(P[:,S], w[S], X)
            for i in range(self.k+1):
                X[i] = 0
                for j in range(lenS):
                    X[i] += P[i,S[j]]*w[S[j]]
            #*
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # stopping criteria: ||X|| small or increased
            #
            normx = norm(X, self.k+1)
            if normx < 1e-8:
                break
            if abs(normx-oldnormx) < 1e-10 and normx < 1e-7:
                #print('small change {0}'.format(normx))
                break
            if normx >= oldnormx:#+EPS:
                #print("∥X∥ increased in cycle {0}: {1} > {2}".format(majorCycle, normx, oldnormx))
                break
            oldnormx = normx
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (b): P_J = ArgMin(X^T P: P in Polytope)
            for k in range(self.k+1):
                P_J[k] = 0
            IF TIMING:
                self.timer.start()
            self.solveScalarization(X, P_J)
            IF TIMING:
                self.sp_time += self.timer.stop()
            P_J[self.k] -= Y[self.k] # translate polytope by -Y
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (c): if X^T P_J > X^T X - bla: STOP
            b = dot(P_J, P_J, self.k+1)
            if dot(X, P_J, self.k+1) > normx*normx - 1e-12*sqrt(self.k):
                #logging.debug('stop in 1 (c)')
                break
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            #*newIndex = np.flatnonzero(Sfree)[0]
            
            # reserve a new index for augmenting S
            for k in range(self.k+2):
                if Sfree[k] == 1:
                    newIndex = k
                    break
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (f)
            # rhs = P[S]^T P_J    
            IF TIMING:
                self.timer.start()
            #*rhs= P[:,S].T*P_J
            for i in range(lenS):
                space2[i] = 0
                for j in range(self.k+1):
                    space2[i] += P[j,S[i]]*P_J[j]
            # rhs += 1
            for k in range(lenS):
                space2[k] += 1
            solveLT(R.T, space2, space1, S, lenS)
            # solve for r (as space1)
            IF TIMING:
                self.lstsq_time += self.timer.stop() 
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (g): augment R
            #*R[S,newIndex] = space1
            IF TIMING:
                self.timer.start()
            for k in range(lenS):
                R[S[k],newIndex] = space1[k]
                R[newIndex,S[k]] = 0
            c = dot(space1, space1, lenS)
            R[newIndex,newIndex] = sqrt(1 + b - c)
            IF TIMING:
                self.r_time += self.timer.stop()
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (e): S = S \cup J
            #*P[:,newIndex] = P_J
            for k in range(self.k+1):
                P[k,newIndex] = P_J[k]
            # augment S
            #S = np.append(S, newIndex)
            S[lenS] = newIndex
            Sfree[newIndex] = False
            w[newIndex] = 0
            lenS += 1
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            self.lenS = lenS
            if self.innerLoop() == -1:
                return
            lenS = self.lenS
            
        X[self.k] += Y[self.k]
        self.lenS = lenS

    cdef int innerLoop(self):
        cdef:
            int i, j, k, lenS = self.lenS
            int firstZeroIndex, firstZeroIndexInS, IinS, Ip1, I
            double a, b, c, theta
            bint cond
            np.double_t[:] space1 = self.space1, \
                           space2 = self.space2, \
                           space3 = self.space3, \
                           w = self.w
            np.double_t[:,:] R = self.R
            np.int_t[:] S = self.S
            np.ndarray[ndim=1,dtype=np.uint8_t,cast=True] Sfree = self.Sfree
        
        while True:
            for k in range(self.k+2):
                space3[k] = 1
            IF TIMING:
                self.timer.start()
            solveLT(R.T, space3, space1, S, lenS) #space1= \bar u
            solveUT(R, space1, space2, S, lenS) #space2 = u
            IF TIMING:
                self.lstsq_time += self.timer.stop()
            # check
            #result = np.linalg.lstsq(np.vstack((np.ones(lenS), Q)), e1)
            #u_correct = result[0]
            
            #*space1 = space2 / np.sum(space2) #space1=v
            # a = np.sum(space2) # remember: space3 = ones!
            a= dot(space2, space3, lenS)
            if np.isnan(a):
                # a numerically bad case that happens extremely seldom
                if 'NaNs' not in self.stats:
                    self.stats['NaNs'] = 0
                self.stats['NaNs'] += 1
                self.lenS = lenS-1
                return -1
            for k in range(lenS):
                space1[k] = space2[k]/a
    
            #*if np.all(space1 > EPS):
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
            self.minorCycles += 1
            #*POS = np.flatnonzero(space1 <= EPS) # 3 (a) corrected                    
            #*theta = min(1, np.max(space1[POS]/(space1[POS] - w[S][POS]))) # 3 (b) corrected
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
            #*w[np.flatnonzero(w<=EPS)] = 0 # 3 (d)
            #*firstZeroIndexInS = np.flatnonzero(w[S]==0)[0]
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
            
            IF TIMING:
                self.timer.start()
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
            IF TIMING:
                self.r_time += self.timer.stop()
            # shrink S
            for i in range(firstZeroIndexInS, lenS-1):
                S[i] = S[i+1]
            Sfree[firstZeroIndex] = True
            lenS -= 1
        self.lenS = lenS
        return 1
    
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result):
        """Solve the weighted sum scalarization problem, i.e. shortest path with modified cost.
        
        *direction* defines the weights for the different constraints, where direction[-1] is
        the weight of the original objective function. The resulting values of g_i(f) and
        c(f) are placed in *result*.
        """
        
        cdef:
            double lamb = direction[self.k], c_result = 0
        for enc in self.code.encoders:
            c_result += shortestPathScalarization(enc.trellis, lamb, direction, result)
        result[self.k] = c_result
    
    cdef void resetData(self):
        cdef:
            int k
            double norm_p1_sq = 0
            np.double_t[:,:] P = self.P, \
                             R = self.R
            np.double_t[:] Y = self.Y, w = self.w
            np.int_t[:] S = self.S
            np.ndarray[ndim=1,dtype=np.uint8_t,cast=True] Sfree = self.Sfree

        
        R[0,0] = sqrt(1+dot(Y, Y, self.k))
        for k in range(self.k):
            P[k,0] = Y[k]
            Y[k] = 0
        P[self.k, 0] = 0
        # initialize S, w, and R
        self.lenS = 1
        S[0] = 0
        Sfree[0] = False
        for k in range(1, self.k+2):
            Sfree[k] = True
        w[0] = 1
        
    cdef void updateData(self, double delta_r):
        """ Update R via cholesky decomposition of Q^T Q.
        """
        # compute RHS = ee^T + Q^TQ
        cdef:
            int i, j, k
            double a
            int lenS = self.lenS
            np.double_t[:,:] P = self.P, \
                             R = self.R, \
                             RHS = self.RHS
            np.int_t[:] S = self.S
            
        for k in range(self.lenS):
            P[self.k, S[k]] += delta_r
        for i in range(lenS):
            for j in range(i,lenS):
                a = 0
                for k in range(self.k+1):
                    a += P[k,S[i]]*P[k,S[j]]
                RHS[i,j] = 1 + a
        # compute cholesky decomposition of Q^tQ
        for i in range(lenS):
            for j in range(i):
                a = RHS[j,i]
                for k in range(j):
                    a -= R[S[k],S[i]]*R[S[k],S[j]]
                R[S[j],S[i]] = a / R[S[j],S[j]]
            a = RHS[i,i]
            for k in range(i):
                a -= R[S[k],S[i]]*R[S[k],S[i]]
            R[S[i],S[i]] = sqrt(a)
        # END cholesky decomposition
        self.innerLoop()
        
    cdef void updateData2(self, double delta_r):
        cdef:
            int i, j, k, min_index
            double a, min_value=9e20
            int lenS = self.lenS
            np.double_t[:,:] P = self.P, \
                             R = self.R
            np.int_t[:] S = self.S
            np.double_t[:] w = self.w
            
            np.ndarray[ndim=1,dtype=np.uint8_t,cast=True] Sfree = self.Sfree
            
        for k in range(self.lenS):
            P[self.k, S[k]] += delta_r
            a = 0
            for i in range(self.k+1):
                a += np.square(P[i, S[k]])
            if a < min_value:
                min_value = a
                min_index = k
        
        R[min_index,min_index] = sqrt(1+a)
        # initialize S, w, and R
        self.lenS = 1
        S[0] = min_index
        Sfree[min_index] = False
        for k in range(self.k+2):
            Sfree[k] = k != min_index
        w[min_index] = 1
    
    cpdef params(self):
        return OrderedDict([ ("name", self.name), ("maxMajorCycles", self.maxMajorCycles) ])


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
