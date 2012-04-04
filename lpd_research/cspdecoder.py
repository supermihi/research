#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from lpdecoding.core import Decoder
from lpdecoding.codes import trellis, turbolike, interleaver
from lpdecoding.codes.trellis import INPUT, PARITY
from lpdecoding.algorithms import path
import itertools
import numpy
EPS = 1e-10
import logging
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
        
class NDFDecoder(Decoder):
    """Nondominated Facet Decoder."""
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

        
    def solve(self):
        self.code.setCost(self.llrVector)
        
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
            X, P, w = self.NPA(Y, P, w)
            P_reduced = P[:-1,:]
            test = numpy.linalg.lstsq(numpy.vstack((numpy.ones(P_reduced.shape[1]), P_reduced)), e1_test)
            u_test = test[0]
            v_test = u_test / numpy.sum(u_test)
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
        print('major cycles: {0}    minor cycles: {1}'.format(self.majorCycles, self.minorCycles)) 
    
    def NPA(self, Y, P = None, w = None):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149.
        Y: reference point
        P: matrix of corral points (column-wise), if given
        w: weight vector, if P is given"""
        
        logging.debug("starting NPA with Y = {0}".format(Y))
        if P is None:
            P = -Y.reshape((self.k+1,1))
        if w is None:
            w = numpy.ones((1,1),dtype=numpy.double)
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
            if normx > oldnormx:
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

            P = numpy.hstack((P, P_J.reshape((self.k+1,1))))
            #print('step 1 (b): P_J = {0}'.format(repr(P_J)))
            w = numpy.vstack((w, 0))
            #print('step 1 (e): w = {0}'.format(repr(w)))
            for j in itertools.count():
                logging.debug('\n**STEP 2 [iteration {0}]'.format(j))
                result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(P.shape[1]), P)), e1)
                u = result[0]
                v = u / numpy.sum(u)
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
        for enc in self.code.encoders:
            c_result += path.shortestPathScalarization(enc.trellis, lamb, mu, g_result)
        return numpy.hstack((g_result, c_result))

if __name__ == "__main__":
    from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
    from lpdecoding import simulate
    numpy.random.seed(1337)
    #inter = interleaver.Interleaver(repr = [1,0] )
    #encoder = trellis.TD_InnerEncoder() # 4 state encoder
    
    inter = interleaver.lte_interleaver(64)
    encoder = trellis.LTE_Encoder()
    code = turbolike.StandardTurboCode(encoder, inter)
    
    decoder = NDFDecoder(code)
    ref_decoder =CplexTurboLikeDecoder(code, ip = False)
    
    gen = simulate.AWGNSignalGenerator(code, snr = 1)
    for i in range(10):
        llr = next(gen)
        #llr = numpy.array([-0.2, -0.8,  1.2,  1.1,  1.2,  0.4,  0. ,  0.2, -0. , -0.9, -0.2, -1.3, -0.5,  0.8])
        logging.debug("llr vector: {0}".format(repr(llr)))
        ref_decoder.decode(llr)
        print('real: {0}'.format(ref_decoder.objectiveValue))
        decoder.decode(llr)
        print('solution: {0}'.format(decoder.objectiveValue))
    
        logging.debug('real solution: {0}'.format(ref_decoder.objectiveValue))