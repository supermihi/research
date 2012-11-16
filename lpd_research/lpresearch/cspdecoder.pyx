# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# Copyright 2012 Michael Helmling
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains the constrained shortest path LP decoder for turbo-like codes."""

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, abs

from lpdecoding.core import Decoder
from lpdecoding.algorithms.path cimport shortestPathScalarization 
from lpdecoding.codes.trellis cimport _INFO, _PARITY
from lpdecoding.utils cimport StopWatch

DEF EPS = 1e-10

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

cdef void solveLTTrans(np.double_t[:,:] a,
                  np.double_t[:] b,
                  np.double_t[:] x,
                  np.int_t[:] S,
                  int lenS):
    """Ensure that x[:lenS] solves numpy.dot(a[S[:lenS][:,S[:lenS].T, x[:lenS]) = b[:lenS]
    if a is a lower triangular matrix."""
    cdef:
        int i,j
        double tmp         
    for i in xrange(lenS):
        tmp = b[i]
        for j in xrange(i):
            tmp -= x[j]*a[S[j],S[i]]
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

labelStr = { _INFO: "info", _PARITY: "parity"}

cdef class CSPDecoder(Decoder):
    """Constrained Shortest Path Decoder."""
    
    def __init__(self, code, name=None, maxMajorCycles=0, measureTimes=False):
        """Initialize the decoder for a *code* and name it *name*.
        
        Optionally you may supply *maxMajorCycles* to limit the maximum number of major cycles
        (nearest point calculations) performed. A value of 0 means no limit.
        """
           
        Decoder.__init__(self, code)
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        self.blocklength = code.blocklength
        self.maxMajorCycles = maxMajorCycles
        self.measureTimes = measureTimes
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
        self.P = np.empty((self.k+1, self.k+2))
        self.R = np.empty((self.k+2,self.k+2))
        self.S = np.empty(self.k+2, dtype=np.int)
        self.Sfree = np.empty(self.k+2,dtype=np.int)
        self.w = np.empty(self.k+2)
        self.RHS = np.empty((self.k+2, self.k+2))
        self.codewords = np.empty((self.k+2, self.blocklength), dtype=np.double)
        self.direction = np.empty(self.k+1)
        self.timer = StopWatch()
        
    cpdef solve(self, np.int_t[:] hint=None):
        """Solve the LP problem by a number of nearest point problems in constraints space.
        """
        
        cdef:
            double old_ref = 0, ref = 0, b_r, norm_a_r, delta_r
            int k, mainIterations = 0
            np.double_t[:] direction = self.direction, X = self.X
            np.double_t[:,:] codewords = self.codewords           
        self.majorCycles = self.minorCycles = 0
        if self.measureTimes:
            self.lstsq_time = self.sp_time = self.cho_time = self.r_time = self.gensol_time = self.setcost_time = 0
            self.timer.start()
        self.code.setCost(.1*self.llrVector)
        if self.measureTimes:
            self.setcost_time += self.timer.stop()
        
        #  find path with minimum cost
        for k in range(self.k):
            direction[k] = 0
        direction[self.k] = 1
        self.solveScalarization(direction, X, codewords[0,:])
        self.lenS = 1
        #  test if initial path is feasible
        if norm(X, self.k) < 1e-8:
            self.lenS = 1
            self.S[0] = 0
            self.w[0] = 1
            self.calculateSolution()
            self.objectiveValue = X[self.k]
            self.mlCertificate = self.foundCodeword = True
            try:
                self.stats["immediateSolutions"] += 1
            except KeyError:
                self.stats["immediateSolutions"] = 1
        else:
            self.current_ref = X[self.k]
            self.resetData(X)
            while self.maxMajorCycles == 0 or self.majorCycles < self.maxMajorCycles:
                #===================================================================
                # Main iterations: push up reference point until we hit the LP solution            
                #===================================================================
                mainIterations += 1
                if self.measureTimes:
                    self.timer.start()
                delta_r = old_ref - ref
                self.updateData(delta_r)
                if self.measureTimes:
                    self.cho_time += self.timer.stop()
                # save current objective value
                old_ref = self.current_ref
                if not self.NearestPointAlgorithm():
                    try:
                        self.stats["NaNs"] += 1
                    except KeyError:
                        self.stats["NaNs"] = 1
                    print(self.objectiveValue)
                    print('NAN')
                    break
                #  compute intersection of hyperplane with c-axis
                b_r = self.current_ref*X[self.k] - dot(X, X, self.k+1)
                norm_a_r = -b_r + self.current_ref*(self.current_ref -X[self.k])
                ref = b_r / (self.current_ref - X[self.k])
                if norm_a_r < EPS:
                    #print('norm_a_r={}'.format(norm_a_r))
                    #self.objectiveValue = ref
                    break
                if X[self.k] <= self.current_ref + EPS:
                    #print('INFEASIBLE')
                    self.objectiveValue = np.inf
                    self.solution = None
                    return
                elif np.abs(ref-old_ref < EPS):
                    #self.objectiveValue = ref
                    break
                self.objectiveValue = self.current_ref = ref #  update reference point 
            # if the LP solution is convex combination of only 1 vertex, it must
            # be a codeword (and thus, ML certificate is present)
            self.mlCertificate = self.foundCodeword = (self.lenS==1)
            if self.measureTimes:
                self.timer.start()
            self.calculateSolution()
            if self.measureTimes:
                self.gensol_time += self.timer.stop()
        self.objectiveValue *= 10
        if self.measureTimes:
            if "sp_time" not in self.stats:
                self.stats["lstsq_time"] = self.stats["sp_time"] = 0
                self.stats["cho_time"] = self.stats["r_time"] = 0
                self.stats["gensol_time"] = self.stats["setcost_time"] = 0
            self.stats["lstsq_time"] += self.lstsq_time
            self.stats["sp_time"] += self.sp_time
            self.stats["cho_time"] += self.cho_time
            self.stats["r_time"] += self.r_time
            self.stats["gensol_time"] += self.gensol_time
            self.stats["setcost_time"] += self.setcost_time
        
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
    
    cdef bint NearestPointAlgorithm(self):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
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
            bint badInstance = False
            np.double_t[:,:] R = self.R, \
                             P = self.P, \
                             codewords = self.codewords
            np.double_t[:] space1 = self.space1, \
                           space2 = self.space2, \
                           space3 = self.space3, \
                           w = self.w, \
                           X = self.X
            np.int_t[:] S = self.S, Sfree = self.Sfree
  
        oldnormx = 1e20
        while True:
            self.majorCycles += 1
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (a): X = P[S] * w
            for i in range(self.k+1):
                X[i] = 0
                for j in range(lenS):
                    X[i] += P[i,S[j]]*w[S[j]]
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # stopping criteria: ||X|| small or increased
            normx = norm(X, self.k+1)
            if normx < 1e-8:
                break
            if abs(normx-oldnormx) < EPS and normx < 1e-6:
                #print('small change {0}'.format(normx))
                break
            if normx >= oldnormx:
                #print("∥X∥ increased in cycle {0}: {1} > {2}".format(self.majorCycles, normx, oldnormx))
                break
            oldnormx = normx
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # reserve a new index (J) for augmenting S
            for k in range(self.k+2):
                if Sfree[k] == 1:
                    newIndex = k
                    break
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (b): P_J = ArgMin(X^T P: P in Polytope)
            if self.measureTimes:
                self.timer.start()
            self.solveScalarization(X, P[:, newIndex], codewords[newIndex,:])
            if self.measureTimes:
                self.sp_time += self.timer.stop()
            P[self.k, newIndex] -= self.current_ref # translate polytope by -Y
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (c): if X^T P_J > X^T X - bla: STOP
            if dot(X, P[:, newIndex], self.k+1) > normx*normx - 1e-12*sqrt(self.k):
                break
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (f)
            # rhs = P[S]^T P_J + 1
            if self.measureTimes:
                self.timer.start()
            for i in range(lenS):
                space2[i] = 0
                for j in range(self.k+1):
                    space2[i] += P[j,S[i]]*P[j, newIndex]
            for k in range(lenS):
                space2[k] += 1
            # solve for r (r=space1 here)
            solveLTTrans(R, space2, space1, S, lenS)
            if self.measureTimes:
                self.lstsq_time += self.timer.stop() 
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (g): augment R
            #*R[S,newIndex] = space1
            if self.measureTimes:
                self.timer.start()
            for k in range(lenS):
                R[S[k],newIndex] = space1[k]
                R[newIndex,S[k]] = 0
            c = dot(space1, space1, lenS)
            b = dot(P[:, newIndex], P[:, newIndex], self.k+1)
            R[newIndex,newIndex] = sqrt(1 + b - c)
            if self.measureTimes:
                self.r_time += self.timer.stop()
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 1. (e): S = S \cup J
            S[lenS] = newIndex
            Sfree[newIndex] = 0
            w[newIndex] = 0
            lenS += 1
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # run inner loops (steps 2 & 3)
            self.lenS = lenS
            if self.innerLoop() == -1:
                X[self.k] += self.current_ref
                badInstance = True
                break
            lenS = self.lenS
        
        # round out
        if lenS > 1:
            j=-1
            for k in range(lenS):
                if w[S[k]] > 1-1e-8:
                    j=S[k]
                    w[S[k]] = 1
                    break
            if j != -1:
                for k in range(self.k+1):
                    X[k] = P[k,j]
                lenS = 1
                S[0] = j
                for k in range(self.k+2):
                    Sfree[k] = (k != j)
                
        # translate  solution back
        X[self.k] += self.current_ref
        self.lenS = lenS
        return (not badInstance)

    cdef int innerLoop(self):
        """Perform the inner loop (step 2+3) of the nearest point algorithm."""
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
            np.int_t[:] S = self.S, Sfree = self.Sfree
        while True:
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # Step 2 (a): solve the equations (4.1) using method D
            #
            # space3 = e
            for k in range(self.k+2):
                space3[k] = 1
            if self.measureTimes:
                self.timer.start()
            # space1 = \bar u
            solveLTTrans(R, space3, space1, S, lenS)
            # space2 = u
            solveUT(R, space1, space2, S, lenS)
            if self.measureTimes:
                self.lstsq_time += self.timer.stop()
            # space1 = v = space2 / np.sum(space2)
            # a = np.sum(space2) # remember: space3 = ones!
            a= dot(space2, space3, lenS)
            if np.isnan(a):
                print('baah')
                return -1
            for k in range(lenS):
                space1[k] = space2[k]/a
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Step 2 (b): if (v=space1) > EPS:
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
            if firstZeroIndexInS == -1:
                # numerically desastrous case
                print('omg')
                return -1
            firstZeroIndex = S[firstZeroIndexInS]
            IinS = firstZeroIndexInS
            
            if self.measureTimes:
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
            if self.measureTimes:
                self.r_time += self.timer.stop()
            # shrink S
            for i in range(firstZeroIndexInS, lenS-1):
                S[i] = S[i+1]
            Sfree[firstZeroIndex] = 1
            lenS -= 1
        self.lenS = lenS
        return 1
    
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result, np.double_t[:] codeword):
        """Solve the weighted sum scalarization problem, i.e. shortest path with modified cost.
        
        *direction* defines the weights for the different constraints, where direction[-1] is
        the weight of the original objective function. The resulting values of g_i(f) and
        c(f) are placed in *result*.
        """
        
        cdef:
            double lamb = direction[self.k], c_result = 0
            int k
        for k in range(self.k):
            result[k] = 0
        for k in range(self.blocklength):
            codeword[k] = 0
        for enc in self.code.encoders:
            c_result += shortestPathScalarization(enc.trellis, lamb, direction, result, codeword)
        result[self.k] = c_result
    
    cdef void resetData(self, np.double_t[:] initPoint):
        """Initialize S, w, R, and P given the initial point."""
        cdef:
            int k
            double norm_p1_sq = 0
            np.double_t[:,:] P = self.P, \
                             R = self.R
            np.double_t[:] w = self.w
            np.int_t[:] S = self.S, Sfree = self.Sfree

        R[0,0] = sqrt(1+dot(initPoint, initPoint, self.k))
        for k in range(self.k):
            P[k,0] = initPoint[k]
        P[self.k, 0] = 0
        self.lenS = 1
        S[0] = 0
        Sfree[0] = 0
        for k in range(1, self.k+2):
            Sfree[k] = 1
        w[0] = 1
        self.current_ref = initPoint[self.k]
        
    cdef void updateData(self, double delta_r):
        """ Update P and R via cholesky decomposition of Q^T Q."""
        
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
        # compute RHS = ee^T + Q^TQ
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
    
    cdef void calculateSolution(self):
        cdef:
            np.ndarray[ndim=1, dtype=np.double_t] solution = np.zeros(self.blocklength,
                                                                      dtype=np.double)
            np.double_t[:,:] codewords = self.codewords
            np.int_t[:] S = self.S
            np.double_t[:] w = self.w
            int i, k
        for i in range(self.blocklength):
            for k in range(self.lenS):
                solution[i] += w[S[k]]*codewords[S[k],i]
        self.solution = solution

    def printSolutionParts(self):
        for k in range(self.lenS):
            print(np.asarray(self.paths[self.S[k],:], dtype=np.int), self.P[self.k, self.S[k]])
        print('')
        for k in range(self.lenS):
            print(self.code.encode(np.asarray(self.paths[self.S[k],:], dtype=np.int)))
        
    
    cpdef params(self):
        return OrderedDict([("name", self.name),
                            ("maxMajorCycles", self.maxMajorCycles),
                            ("measureTimes", self.measureTimes)])


class NDFInteriorDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self)
        self.code = code
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        for i, ((t1, s1, l1), (t2, s2, l2)) in enumerate(self.constraints):
            if not hasattr(t1[s1], "g_coeffs"):
                t1[s1].g_coeffs = { _INFO: [], _PARITY: [] }
            t1[s1].g_coeffs[l1].append( (i, 1) )
                
            if not hasattr(t2[s2], "g_coeffs"):
                t2[s2].g_coeffs = { _INFO: [], _PARITY: [] }
            t2[s2].g_coeffs[l2].append( (i, -1) )
