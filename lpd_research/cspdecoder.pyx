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
import logging, os
import numpy as np
cimport numpy as np 

logging.basicConfig(level=logging.ERROR)

DEF TimeMeasure = True # compile-time variable; if True, time of various steps is measured and output

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

cdef class NDFDecoder(Decoder):
    """Nondominated Facet Decoder."""
    def __init__(self, code, name = None):
        Decoder.__init__(self, code)
        self.constraints = code.equalityPairs()
        self.k = len(self.constraints)
        if name is None:
            self.name = str(self)
        else:
            self.name = name
        
        for i, ((t1, s1, l1), (t2, s2, l2)) in enumerate(self.constraints):
            setattr(t1[s1], "g_{0}_index".format(labelStr[l1]), i)
            setattr(t1[s1], "g_{0}".format(labelStr[l1]), 1)
            #t1[s1].g_coeffs[l1].append( (i, 1) )
            
            setattr(t2[s2], "g_{0}_index".format(labelStr[l2]), i)
            setattr(t2[s2], "g_{0}".format(labelStr[l2]), -1)   
            #t2[s2].g_coeffs[l2].append( (i, -1) )
        self.X = np.empty(self.k+1)
        
    cpdef solve(self):
        cdef:
            np.double_t[:] direction = np.zeros(self.k+1)
            np.double_t[:,:] RHS = np.empty((self.k+2, self.k+2))
            np.ndarray[ndim=1, dtype=np.double_t] \
                w = np.empty(self.k+2),\
                v = np.empty(self.k+1),\
                Y = np.zeros(self.k+1),\
                X = np.empty(self.k+1)
            np.ndarray[ndim=2, dtype=np.double_t] \
                P = np.empty((self.k+1, self.k+2)),\
                R = np.empty((self.k+2,self.k+2))
            np.ndarray[ndim=1, dtype=np.int_t] S = np.zeros(self.k+2, dtype=np.int)
            np.ndarray[ndim=1,dtype=np.uint8_t,cast=True] Sfree = np.ones(self.k+2,dtype=np.bool)
            double oldZ = 0, z_d = 0, a, time_a, time_b
            int i, j, k, mainIterations = 0, lenS = 1
        
        self.code.setCost(self.llrVector)
        IF TimeMeasure:
            self.lstsq_time = self.sp_time = self.cho_time = self.npa_time = self.r_time = 0
            tmp = os.times()
            time_a = tmp[0] + tmp[2]
        # find point with minimum cost in z-direction
        direction[-1] = 1
        self.solveScalarization(direction, Y)
        if norm(Y, self.k) < 1e-8:
            print("i",end='')
            self.solution = Y
            self.objectiveValue = Y[-1]
            self.X = np.zeros(self.k+1)
            return
        # set all but last component to 0 -> initial Y for nearest point algorithm
        for k in range(self.k):
            Y[k] = 0
        self.majorCycles = self.minorCycles = 0
        
        for k in range(self.k+1):
                P[k,0] = -Y[k]
        Sfree[0] = False
        for k in range(1, self.k+2):
            Sfree[k] = True
        w[0] = 1
        R[0,0] = sqrt(1+dot(Y, Y, self.k+1))
        S[0] = 0
        while True:
            # initialize data arrays for NFA
            
            # adjust points in Q
            for k in range(lenS):
                P[self.k, S[k]] += oldZ - z_d
            
            for k in range(self.k+1):    
                self.X[k] = v[k]
            
            # adjust R
            IF TimeMeasure:
                tmp = os.times()
                time_b = tmp[0] + tmp[2]
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
            IF TimeMeasure:
                tmp = os.times()
                self.cho_time += tmp[0] + tmp[2] - time_b
            oldZ = Y[-1]
            IF TimeMeasure:
                tmp = os.times()
                time_b = tmp[0] + tmp[2]
            lenS = self.NPA(Y, P, S, Sfree, w, R, lenS, X)
            IF TimeMeasure:
                tmp = os.times()
                self.npa_time += tmp[0] + tmp[2] - time_b 
            mainIterations += 1
            for k in range(self.k+1):
                v[k] = X[k] - Y[k]
            if norm(v, self.k+1) < 1e-8:
                #print('done(v)')
                break
            z_d = dot(X, v, self.k+1) / v[-1]
            if abs(z_d-oldZ) < 1e-8:
                #print('done')
                break
            Y[-1] = z_d 
        self.objectiveValue = Y[-1]
        
        IF TimeMeasure:
            tmp = os.times()
            if len(self.stats) == 0:
                self.stats["lstsq_time"] = self.stats["sp_time"] = self.stats["cho_time"] = self.stats["r_time"] = 0
            self.stats["lstsq_time"] += self.lstsq_time
            self.stats["sp_time"] += self.sp_time
            self.stats["cho_time"] += self.cho_time
            self.stats["r_time"] += self.r_time
            #print('total time: {}; least square solution time: {}; shortest path time: {}; cho time: {}; r time: {}; npa time: {}'.format(
            #              tmp[0]+tmp[2]-time_a,
            #              self.lstsq_time, self.sp_time, self.cho_time, self.r_time, self.npa_time))
            #print('main iterations: {0}   major cycles: {1}   minor cycles: {2}'.format(mainIterations, self.majorCycles, self.minorCycles)) 
    
    cdef int NPA(self,
            np.ndarray[ndim=1, dtype=np.double_t] Y,
            np.ndarray[ndim=2, dtype=np.double_t] P,
            np.ndarray[ndim=1, dtype=np.int_t] S,
            np.ndarray[ndim=1, dtype=np.uint8_t, cast = True] Sfree,
            np.ndarray[ndim=1, dtype=np.double_t] w,
            np.ndarray[ndim=2, dtype=np.double_t] R,
            int lenS,
            np.ndarray[ndim=1, dtype=np.double_t] X):
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
            np.ndarray[ndim=1, dtype=np.double_t] P_J, space1, space2, space3
            double normx, oldnormx, a, b, c, theta, time_a
            int majorCycle = 0, minorCycle = 0, \
                i, j, k, newIndex = 1, \
                IinS, I, Ip1, firstZeroIndex, firstZeroIndexInS
            bint cond = False
  
        P_J = np.empty(self.k+1)
        oldnormx = 1e20
        space1 = np.empty(self.k+2, dtype=np.double)
        space2 = np.empty(self.k+2, dtype=np.double)
        space3 = np.empty(self.k+2, dtype=np.double)
        while True:
            majorCycle += 1
            self.majorCycles += 1
            
            #*np.dot(P[:,S], w[S], X)
            for i in range(self.k+1):
                X[i] = 0
                for j in range(lenS):
                    X[i] += P[i,S[j]]*w[S[j]]
            #*
            normx = norm(X, self.k+1)
            if normx < 1e-8:
                break
            if abs(normx-oldnormx) < 1e-10 and normx < 1e-7:
                #print('small change {0}'.format(normx))
                #raw_input()
                break
            if normx > oldnormx+EPS:
                print("∥X∥ increased in cycle {0}: {1} > {2}".format(majorCycle, normx, oldnormx))
                break
            oldnormx = normx
            for k in range(self.k+1):
                P_J[k] = 0
            IF TimeMeasure:
                tmp = os.times()
                time_a = tmp[0] + tmp[2]
            self.solveScalarization(X, P_J)
            IF TimeMeasure:
                tmp= os.times()
                self.sp_time += tmp[0] + tmp[2] - time_a
            P_J[self.k] -= Y[self.k]
            b = dot(P_J, P_J, self.k+1)
            if dot(X, P_J, self.k+1) > normx*normx - 1e-12*sqrt(self.k):
                #logging.debug('stop in 1 (c)')
                break
            #*newIndex = np.flatnonzero(Sfree)[0]
            newIndex = -1
            for k in range(self.k+2):
                if Sfree[k] == 1:
                    newIndex = k
                    break
                
            IF TimeMeasure:
                    tmp = os.times()
                    time_a = tmp[0] + tmp[2]
            #*rhs= P[:,S].T*P_J
            for i in range(lenS):
                space2[i] = 0
                for j in range(self.k+1):
                    space2[i] += P[j,S[i]]*P_J[j]
            for k in range(lenS):
                space2[k] += 1
            solveLT(R.T, space2, space1, S, lenS)
            IF TimeMeasure:
                tmp = os.times()
                self.lstsq_time += tmp[0] + tmp[2] - time_a 
            
            # augment R
            #*R[S,newIndex] = space1
            IF TimeMeasure:
                tmp = os.times()
                time_a = tmp[0] + tmp[2]
            for k in range(lenS):
                R[S[k],newIndex] = space1[k]
                R[newIndex,S[k]] = 0
            c = dot(space1, space1, lenS)
            R[newIndex,newIndex] = sqrt(1 + b - c)
            IF TimeMeasure:
                tmp = os.times()
                self.r_time += tmp[0] + tmp[2] - time_a
                
            #*P[:,newIndex] = P_J
            for k in range(self.k+1):
                P[k,newIndex] = P_J[k]
            # augment S
            #S = np.append(S, newIndex)
            S[lenS] = newIndex
            Sfree[newIndex] = False
            w[newIndex] = 0
            lenS += 1
            # check if R augmentation is correct
            # assert np.linalg.norm(np.dot(R[S][:,S].T,R[S][:,S]) - np.dot(Q.T,Q) - np.ones((lenS, lenS))) < EPS
            minorCycle = 0
            while True:
                minorCycle += 1
#                logging.debug('***STEP 2 [minor {0}]'.format(minorCycle))
                #*space3 = np.ones(lenS)
                for k in range(self.k+2):
                    space3[k] = 1
                IF TimeMeasure:
                    tmp = os.times()
                    time_a = tmp[0] + tmp[2]
                solveLT(R.T, space3, space1, S, lenS) #space1= \bar u
                solveUT(R, space1, space2, S, lenS) #space2 = u
                IF TimeMeasure:
                    tmp = os.times()
                    self.lstsq_time += tmp[0] + tmp[2] - time_a 
                # check
                #result = np.linalg.lstsq(np.vstack((np.ones(lenS), Q)), e1)
                #u_correct = result[0]
                
                #*space1 = space2 / np.sum(space2) #space1=v
                # a = np.sum(space2) # remember: space3 = ones!
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
                else:
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
                    
                    IF TimeMeasure:
                        tmp = os.times()
                        time_a = tmp[0] + tmp[2]
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
                    IF TimeMeasure:
                        tmp = os.times()
                        self.r_time += tmp[0] + tmp[2] - time_a
                    # shrink S
                    for i in range(firstZeroIndexInS, lenS-1):
                        S[i] = S[i+1]
                    Sfree[firstZeroIndex] = True
                    lenS -= 1

        for k in range(self.k+1):
            X[k] += Y[k]
        return lenS

    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result):
        cdef:
            double lamb = direction[-1], c_result = 0
        for enc in self.code.encoders:
            c_result += shortestPathScalarization(enc.trellis, lamb, direction, result)
        result[self.k] = c_result
                
    def params(self):
        return {"name" : self.name}
    
    def __str__(self):
        return "NDFDecoder"

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
