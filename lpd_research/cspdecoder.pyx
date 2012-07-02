#!/usr/bin/python2
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: nonecheck=False
#cython: cdivision=True
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
from libc.math cimport sqrt 
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

cdef void solveLT(numpy.double_t[:,:] a,
                  numpy.double_t[:] b,
                  numpy.double_t[:] x):
    """Return x where x solves ax = b if a is a lower triangular matrix."""
    cdef:
        int i,j
        double tmp         
    for i in xrange(a.shape[0]):
        tmp = b[i]
        for j in xrange(i):
            tmp -= x[j]*a[i,j]
        x[i] =  tmp / a[i,i]
        #x[i] = (b[i] - numpy.dot(x[:i], a[i,:i])) / a[i,i]

cdef void solveUT(numpy.double_t[:,:] a,
                  numpy.double_t[:] b,
                  numpy.double_t[:] x):
    """Return x where x solves ax = b if a is an upper triangular matrix."""
    cdef:
        int size = a.shape[0]
        int i,j
        double tmp
    for i in xrange(size-1, -1, -1):
        tmp = b[i]
        for j in xrange(i+1,size):
            tmp -= x[j]*a[i,j]
        x[i] = tmp / a[i,i]
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
            numpy.double_t[:] direction
            numpy.ndarray[ndim=1, dtype=numpy.double_t] w, v, Y, z
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P, R
            numpy.ndarray[ndim=1, dtype=numpy.int_t] S
            double oldZ, z_d
            int i
        self.code.setCost(self.llrVector)
        self.lstsq_time = self.sp_time = self.omg_time = 0
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        # find point with minimum cost in z-direction
        direction = numpy.zeros(self.k+1)
        direction[-1] = 1
        Y = numpy.asarray(self.solveScalarization(direction))
        if numpy.linalg.norm(Y[:-1]) < 1e-8:
            logging.info("initial point is feasible -> yeah!")
            self.solution = Y
            self.objectiveValue = Y[-1]
            return
        # set all but last component to 0 -> initial Y for nearest point algorithm
        
        Y[:-1] = 0
        logging.info("Initial reference point: Y={0}".format(repr(Y)))
        self.majorCycles = self.minorCycles = 0
        P = w = R = S = None
        #oldZ = -1000
        oldZ = Y[-1]
        i = 0
        while True:
            #X, P, w = self.NPA(Y, P, w)
            #ret = self.NPA(Y, P, S, w, R, True)
            print('npa z={0}'.format(oldZ))
            X, P, S, w, R = self.NPA(Y, None, None, None, None)
            i += 1
            v = X - Y
            if numpy.linalg.norm(v) < EPS:
                print('done(v)')
                break
            s = X.copy()
            logging.info("X={0}".format(repr(X)))
            logging.info("Y={0}".format(repr(Y)))
            logging.info("w={0}".format(w))
            logging.info("v[-1]={0}".format(v[-1]))
            z_d = numpy.dot(X.T, v) / v[-1]
            if numpy.abs(z_d - oldZ) < 1e-8:
                print('done')
                break
            P[-1,:] += oldZ - z_d
            oldZ = z_d
            s[-1] -= z_d
            z = X - s
            Y = z
            
            #raw_input()
            
            
            #break
        self.objectiveValue = Y[-1]
        print('main iterations: {0}'.format(i))
        tmp = os.times()
        print('total time: {}; least square solution time: {}; shortest path time: {}; omg time: {}'.format(tmp[0]+tmp[2]-time_a, self.lstsq_time, self.sp_time, self.omg_time))
        print('major cycles: {0}    minor cycles: {1}'.format(self.majorCycles, self.minorCycles)) 
    
    cdef object NPA(self,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] Y,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P,
            numpy.ndarray[ndim=1, dtype=numpy.int_t] S,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] w,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] R):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
        Y: reference point
        P: matrix of corral points (column-wise), if given
        w: weight vector, if P is given
        R: R from Method D, if P and w are given"""
        cdef:
            numpy.ndarray[ndim=1, dtype=numpy.double_t] v, X, e1, P_J, rhs, space1, space2, space3
            numpy.ndarray[ndim=2, dtype=numpy.double_t] Q, RR
            double normx, oldnormx, time_a, a, b, c, theta
            int majorCycle, minorCycle, i, j, k,l,newIndex = 1
            int lenS = 1, IinS, I, Ip1, firstZeroIndex, firstZeroIndexInS
            numpy.ndarray[ndim=1,dtype=numpy.uint8_t,cast=True] Sfree
            object tmptime
            
        Sfree = numpy.ones(self.k+2,dtype=numpy.bool) # stores which indices are free for S
#        logging.debug("starting NPA with Y = {0}".format(Y))
        
        if P is None:
            P = numpy.empty((self.k+1, self.k+2))
            P[:,0] = -Y
            S = numpy.zeros(1, dtype=numpy.int)
            Sfree = numpy.ones(self.k+2,dtype=numpy.bool) # stores which indices are free for S
            Sfree[0] = False
            w = numpy.empty(self.k+2)
            v = numpy.empty(self.k+2)
            w[0] = 1
            R = numpy.zeros((self.k+2,self.k+2))
            R[0,0] = sqrt(1+numpy.sum(numpy.square(P[:,0])))
        else:
            for k in S:
                Sfree[k] = False
            lenS = S.size            
        X = numpy.empty(self.k+1)
        oldnormx = numpy.inf
        e1 = numpy.zeros(self.k+2)
        e1[0] = 1
        Q = P[:,S]
        space1 = numpy.empty(lenS, dtype=numpy.double)
        space2 = numpy.empty(lenS, dtype=numpy.double)
        space3 = numpy.empty(lenS, dtype=numpy.double)
        for majorCycle in itertools.count():
#            logging.debug('\n\n**STEP 1 [major {0}]'.format(majorCycle))
            
            self.majorCycles += 1
            # step 1
            numpy.dot(Q, w[S], X)
#            logging.debug('X={0}'.format(repr(X)))
            normx = numpy.linalg.norm(X)
            if normx > oldnormx + EPS:
                print("∥X∥ increased: {0} > {1}".format(normx, oldnormx))
                raw_input()
#            logging.debug('step 1: ∥X∥ = {0}'.format(normx))
            oldnormx = normx
            #print('step 1: P = {0}'.format(repr(P)))
            #print('step 1: w = {0}'.format(repr(w)))
            P_J = self.solveScalarization(X) - Y
            if numpy.dot(X.T, P_J) > normx*normx - EPS:
                logging.debug('stop in 1 (c)')
                break
            #*newIndex = numpy.flatnonzero(Sfree)[0]
            for k in range(Sfree.size):
                if Sfree[k] == 1:
                    newIndex = k
                    break

            rhs = numpy.dot(Q.T, P_J) + 1
            solveLT(R[S][:,S].T, rhs, space1)
            
            # augment R
            R[S,newIndex] = space1
            R[newIndex,newIndex] = sqrt(1 + numpy.dot(P_J, P_J) - numpy.dot(space1, space1))
                
            # augment S
            P[:,newIndex] = P_J
            S = numpy.append(S, newIndex)
            Sfree[newIndex] = False
            w[newIndex] = 0
            lenS += 1
            space1 = numpy.empty(lenS, dtype=numpy.double)
            space2 = numpy.empty(lenS, dtype=numpy.double)
            space3 = numpy.empty(lenS, dtype=numpy.double)
            Q = P[:,S]
            
            # check if R augmentation is correct
#            assert numpy.linalg.norm(numpy.dot(R[S][:,S].T,R[S][:,S]) - numpy.dot(Q.T,Q) - numpy.ones((lenS, lenS))) < EPS
            for minorCycle in itertools.count():
#                logging.debug('***STEP 2 [minor {0}]'.format(minorCycle))
                RR = R[S][:,S]
                tmptime = os.times()
                time_a = tmptime[0] + tmptime[2]
                solveLT(RR.T, numpy.ones(lenS), space1) #space1= \bar u
                solveUT(RR, space1, space2) #space2 = u
                
                # check
                #result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(lenS), Q)), e1)
                #u_correct = result[0]
                
                #*space1 = space2 / numpy.sum(space2) #space1=v
                a=0
                for k in range(lenS):
                    a += space2[k]
                for k in range(lenS):
                    space1[k] = space2[k]/a
                #*
                
                tmptime = os.times()
                self.lstsq_time += tmptime[0] + tmptime[2] - time_a
                if numpy.all(space1 > EPS):
                    w[S] = space1
                    break
                else:
                    tmptime = os.times()
                    time_a = tmptime[0] + tmptime[2]
                    self.minorCycles += 1
                    POS = numpy.flatnonzero(space1 <= EPS) # 3 (a) corrected                    
                    #theta = min(1, numpy.min(w[POS]/(w[POS] - v[POS]))) # 3 (b)
                    theta = min(1, numpy.max(space1[POS]/(space1[POS] - w[S][POS]))) # 3 (b)
                    
#                    logging.debug("step 3 (c): θ = {0}".format(theta))
                    w[S] = theta*w[S] + (1-theta)*space1 # 3 (c)
                    w[numpy.flatnonzero(w<=EPS)] = 0 # 3 (d)
#                    logging.debug("new w={0}".format(w))
                    # index of S that will leave
                    firstZeroIndexInS = numpy.flatnonzero(w[S]==0)[0]
                    firstZeroIndex = S[firstZeroIndexInS]
#                    logging.debug("firstZeroInS={0}".format(firstZeroIndexInS))
#                    logging.debug("firstZeroIndex={0}".format(firstZeroIndex))
                    
                    
                    IinS = firstZeroIndexInS
                    #*R[:,firstZeroIndex] = 0
                    for k in range(lenS):
                        R[S[k],firstZeroIndex] = 0
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
                    tmptime = os.times()
                    self.omg_time += tmptime[0] + tmptime[2] - time_a
                    # remove last row
                    #*R[firstZeroIndex,:] = 0
                    for k in range(lenS):
                        R[firstZeroIndex,S[k]] = 0
                    # shrink S
                    S = numpy.delete(S, firstZeroIndexInS)
                    Sfree[firstZeroIndex] = True
                    lenS -= 1
                    Q = P[:,S]
                    space1 = numpy.empty(lenS, dtype=numpy.double)
                    space2 = numpy.empty(lenS, dtype=numpy.double)
                    space3 = numpy.empty(lenS, dtype=numpy.double)

        if normx < EPS:
            return Y, P, S, w, R
        else:
            return X + Y, P, S, w, R

    cdef numpy.double_t[:] solveScalarization(self, numpy.double_t[:] direction):
        cdef:
            numpy.double_t[:] mu = direction[:-1], g_result
            double lamb, c_result, time_a
        mu = direction[:-1]
        lamb = direction[-1]
        g_result = numpy.zeros(self.k, dtype = numpy.double)
        c_result = 0
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        for enc in self.code.encoders:
            c_result += shortestPathScalarization(enc.trellis, lamb, mu, g_result)
        tmp = os.times()
        self.sp_time += tmp[0] + tmp[2] - time_a
        return numpy.hstack((g_result, c_result))

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
