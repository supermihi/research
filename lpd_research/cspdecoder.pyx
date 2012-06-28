#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from lpdecoding.core import Decoder
from lpdecoding.codes import turbolike, interleaver
from lpdecoding.codes.ctrellis cimport Trellis 
from lpdecoding.algorithms.path cimport shortestPathScalarization 
from lpdecoding.codes.trellis import INPUT, PARITY
import itertools, logging
import numpy
cimport numpy
import os
EPS = 1e-10
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
    distance = numpy.sqrt(cplex.solution.get_objective_value())
    return distance < 1e-6, cplex.solution.get_values(x)

cdef numpy.ndarray[dtype=numpy.double_t, ndim=2] solveLT(numpy.ndarray[ndim=2, dtype=numpy.double_t] a,
                                                         numpy.ndarray[ndim=2, dtype=numpy.double_t] b):
    """Return x where x solves ax = b if a is a lower triangular matrix."""
    cdef numpy.ndarray x = numpy.empty((a.shape[0], 1))
    cdef int i             
    for i in xrange(a.shape[0]):
        x[i,0] = (b[i,0] - numpy.dot(x[:i,0], a[i,:i])) / a[i,i]
    return x

cdef numpy.ndarray[dtype=numpy.double_t, ndim=2] solveUT(numpy.ndarray[ndim=2, dtype=numpy.double_t] a,
                                                         numpy.ndarray[ndim=2, dtype=numpy.double_t] b):
    """Return x where x solves ax = b if a is an upper triangular matrix."""
    cdef numpy.ndarray x = numpy.empty((a.shape[0], 1))
    cdef int i             
    for i in xrange(a.shape[0]-1, -1, -1):
        x[i,0] = (b[i,0] - numpy.dot(x[i+1:,0], a[i,i+1:])) / a[i,i]
#    x_ref = numpy.linalg.solve(a, b)
#    if not numpy.allclose(x_ref, x):
#        print(a)
#        print(b)
#        raise RuntimeError("SolveLT: {0} != {1}".format(x, x_ref))
    return x
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
        self.code.setCost(self.llrVector)
        self.lstsq_time = self.sp_time = self.omg_time = 0
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        # find point with minimum cost in z-direction
        direction = numpy.zeros(self.k+1)
        direction[-1] = 1
        Y = self.solveScalarization(direction).reshape((self.k+1,1))
        P = Y.copy()
        P[-1][0] = 0
        w = numpy.ones((1,1), dtype = numpy.double)
        if numpy.linalg.norm(Y[:-1]) < 1e-8:
            logging.info("initial point is feasible -> yeah!")
            self.solution = Y
            self.objectiveValue = Y[-1]
            return
        # set all but last component to 0 -> initial Y for nearest point algorithm
        
        Y[:-1] = 0
        logging.info("Initial reference point: Y={0}".format(repr(Y)))
        self.majorCycles = self.minorCycles = 0
        #P = w = None
        #oldZ = -1000
        oldZ = Y[-1]
        e1_test = numpy.zeros((self.k+1,1),dtype=numpy.double)
        e1_test[0,0] = 1
        i = 0
        while True:
            #X, P, w = self.NPA(Y, P, w)
            X, P, w = self.NPA(Y, methodD = True)#False)
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
            z_d = (numpy.dot(X.T, v) / v[-1]).ravel()
            if abs(z_d - oldZ) < 1e-8:
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
    
    def NPA(self,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] Y,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P = None,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] w = None,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] R = None, methodD = False):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
        Y: reference point
        P: matrix of corral points (column-wise), if given
        w: weight vector, if P is given
        R: R from Method D, if P and w are given"""
        
        cdef numpy.ndarray[ndim=2, dtype=numpy.double_t] ubar, u, X, e1
        cdef double oldnormx
        cdef int i
        logging.debug("starting NPA with Y = {0}".format(Y))
        if P is None:
            assert w is None and R is None
            P = -Y.reshape((self.k+1,1))
            w = numpy.ones((1,1),dtype=numpy.double)
            if methodD:
                R = numpy.array([[numpy.sqrt(1+numpy.sum(numpy.square(P)))]])
        else:
            assert w is not None
            if methodD:
                assert R is not None            
            
        X = numpy.empty((self.k+1,1), dtype=numpy.double)
        oldnormx = numpy.inf
        e1 = numpy.zeros((self.k+2,1),dtype=numpy.double)
        e1[0,0] = 1
        
        for i in itertools.count():
            logging.debug('\n\n**STEP 1 [iteration {0}'.format(i))
            self.majorCycles += 1
            # step 1
            numpy.dot(P, w, X)
            #print('step 1: X = {0}'.format(repr(X)))
            normx = numpy.linalg.norm(X)
            if normx > oldnormx + EPS:
                print("∥X∥ increased: {0} > {1}".format(normx, oldnormx))
                raw_input()
            logging.debug('step 1: ∥X∥ = {0}'.format(normx))
            oldnormx = normx
            #print('step 1: P = {0}'.format(repr(P)))
            #print('step 1: w = {0}'.format(repr(w)))
            P_J = self.solveScalarization(X.ravel()).reshape((self.k+1,1)) - Y
                            
            if numpy.dot(X.T, P_J) > numpy.dot(X.T, X) -EPS:
                logging.debug('stop in 1 (c)')
                break
            if methodD:
                rhs = numpy.ones((R.shape[0],1)) + numpy.dot(P.T, P_J)
                r = solveLT(R.T, rhs)
                r = numpy.append(r, numpy.sqrt(1 + numpy.dot(P_J.T, P_J) - numpy.dot(r.T, r)))
                R = numpy.vstack( (R, numpy.zeros((1, R.shape[0]))) )
                R = numpy.hstack( (R, r.reshape((R.shape[1] + 1, 1))) )
                logging.info("{0}x{1} R={2}".format(R.shape[0], R.shape[1], repr(R)))
                logging.info("r={0}".format(r))
            P = numpy.hstack((P, P_J.reshape((self.k+1,1))))
            #print('step 1 (b): P_J = {0}'.format(repr(P_J)))
            w = numpy.vstack((w, 0))
            #print('step 1 (e): w = {0}'.format(repr(w)))
            
            for j in itertools.count():
                logging.debug('\n**STEP 2 [iteration {0}]'.format(j))
                tmp = os.times()
                time_a = tmp[0] + tmp[2]
                if methodD:
                    ubar = solveLT(R.T, numpy.ones((R.shape[0],1)))
                    u= solveUT(R, ubar)
                else:
                    result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(P.shape[1]), P)), e1)
                    u = result[0]
                v = u / numpy.sum(u)
                tmp = os.times()
                self.lstsq_time += tmp[0] + tmp[2] - time_a
                
                #print("step 2 (a): v = {0}".format(repr(v)))                 
                if numpy.all(v > EPS):
                    w = v
                    break
                else:
                    self.minorCycles += 1
                    POS = numpy.nonzero(v <= EPS) # 3 (a) corrected                    
                    logging.debug("step 3 (a): POS = {0}".format(repr(POS)))
                    #theta = min(1, numpy.min(w[POS]/(w[POS] - v[POS]))) # 3 (b)
                    theta = min(1, numpy.max(v[POS]/(v[POS] - w[POS]))) # 3 (b)
                    
                    logging.debug("step 3 (c): θ = {0}".format(theta))
                    w = theta*w + (1-theta)*v # 3 (c)
                    w[numpy.nonzero(w<=EPS)] = 0 # 3 (d)
                    #print('step 3 (d): w = {0}'.format(repr(w)))
                    firstZeroIndex = numpy.nonzero(w == 0)[0][0]
                    w = numpy.delete(w, firstZeroIndex, 0)
                    P = numpy.delete(P, firstZeroIndex, 1)
                    
                    # method D
                    if methodD:
                        I = firstZeroIndex
                        R = numpy.delete(R, I, 1)
                        while I < R.shape[1]:
                            a = R[I,I]
                            b = R[I+1,I]
                            c = numpy.sqrt(a*a+b*b)
                            R[I], R[I+1] = (a*R[I] + b*R[I+1])/c, (-b*R[I] + a*R[I+1])/c
                            I += 1
                        # remove last row
                        R = numpy.delete(R, -1, 0)
                        logging.info("new R: {0}".format(repr(R)))
        
            
            #raw_input()    
        
        if normx < EPS:
            solution = Y
        else:
            solution = X + Y
        return solution, P, w

    def solveScalarization(self, direction):
        assert direction.size == self.k+1
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
