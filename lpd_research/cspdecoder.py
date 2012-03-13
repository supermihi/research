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
EPS = 1e-8
        
        
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
        for encoder in self.code.encoders:
            trellis = encoder.trellis
            print(encoder)
            for segment in trellis:
                try:
                    print(segment.g_coeffs)
                except AttributeError:
                    print('segment {0}: no coeffs'.format(segment.pos))

        
    def solve(self):
        self.code.setCost(self.llrVector)
        self.NDA()
    
    def NDA(self):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149."""
        # find the initial point: minimize over last axis (original c objective function)
        # (step 0)
        direction = numpy.zeros(self.k+1)
        direction[self.k] = 1
        min_z = self.solveScalarization(direction)
        print('step 0: P_J = {0}'.format(repr(min_z)))
        
        Y = numpy.zeros((self.k+1,1))
        Y[-1] = min_z[-1]
        print("Y = {0}".format(Y))
        P = min_z.reshape((self.k+1,1)) - Y

        w = numpy.ones((1,1),dtype=numpy.double)
        X = numpy.empty((self.k+1,1), dtype=numpy.double)
        oldnormx = numpy.inf
        e1 = numpy.zeros((self.k+2,1),dtype=numpy.double)
        e1[0,0] = 1
        majorCycles = minorCycles = 0
        for i in itertools.count():
            if i > 200:
                print('200')
                raw_input()
                break
            print('\n\n**STEP 1 [iteration {0}'.format(i))
            majorCycles += 1
            # step 1
            numpy.dot(P, w, X)
            print('step 1: X = {0}'.format(X))
            normx = numpy.linalg.norm(X)
            if normx > oldnormx:
                print("∥X∥ increased: {0} > {1}".format(normx, oldnormx))
                raw_input()
            print('step 1: ∥X∥ = {0}'.format(normx))
            oldnormx = normx
            print('step 1: P = {0}'.format(repr(P)))
            print('step 1: w = {0}'.format(repr(w)))
            P_J = self.solveScalarization(X.ravel()).reshape((self.k+1,1)) - Y
            
            ans, sol = zeroInConvexHull([p.ravel() for p in numpy.hsplit(P, P.shape[1])])                
            if numpy.dot(X.T, P_J) > numpy.dot(X.T, X) -EPS:
                print('stop in 1 (c)')
                if not ans:
                    print('WARNING CPLEX DOES NOT AGREE')
                break
            elif ans:
                print('break cplex')
                break

            P = numpy.hstack((P, P_J.reshape((self.k+1,1))))
            print('step 1 (b): P_J = {0}'.format(repr(P_J)))
            w = numpy.vstack((w, 0))
            print('step 1 (e): w = {0}'.format(repr(w)))
            for j in itertools.count():
                if j > 100:
                    print('100')
                    raw_input()
                    break
                print('\n**STEP 2 [iteration {0}]'.format(j))
                result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(P.shape[1]), P)), e1)
                u = result[0]
                v = u / numpy.sum(u)
                print("step 2 (a): v = {0}".format(repr(v)))
                assert numpy.allclose(numpy.sum(v), 1)
                if not numpy.allclose(numpy.dot( (numpy.ones((P.shape[1], P.shape[1])) + numpy.dot(P.T, P)), u), numpy.ones((P.shape[1],1))):
                    print(numpy.dot( (numpy.ones((P.shape[1], P.shape[1])) + numpy.dot(P.T, P)), u) - numpy.ones((P.shape[1],1)))
                    raw_input()
                 
                if numpy.all(v > EPS):
                    w = v
                    break
                
                else:
                    minorCycles += 1
                    POS = numpy.nonzero(v <= EPS) # 3 (a) corrected                    
                    print("step 3 (a): POS = {0}".format(repr(POS)))
                    #theta = min(1, numpy.min(w[POS]/(w[POS] - v[POS]))) # 3 (b)
                    theta = min(1, numpy.max(v[POS]/(v[POS] - w[POS]))) # 3 (b)
                    
                    print("step 3 (c): θ = {0}".format(theta))
                    w = theta*w + (1-theta)*v # 3 (c)
                    w[numpy.nonzero(w<=EPS)] = 0 # 3 (d)
                    print('step 3 (d): w = {0}'.format(repr(w)))
                    firstZeroIndex = numpy.nonzero(w == 0)[0][0]
                    w = numpy.delete(w, firstZeroIndex, 0)
                    P = numpy.delete(P, firstZeroIndex, 1)
        
            
            #raw_input()    
        print('major cycles: {0}    minor cycles: {1}'.format(majorCycles, minorCycles))
        if normx < EPS:
            print('solution*: {0}'.format(Y[-1][0]))
        else:
            print('solution: {0}'.format(Y[-1].ravel() + numpy.dot(X.T, X).ravel() / X[-1].ravel()))
        return majorCycles, minorCycles

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
    interleaver = interleaver.Interleaver(repr = [1,4,3,2,0] )
    encoder = trellis.TD_InnerEncoder() # 4 state encoder
    code = turbolike.StandardTurboCode(encoder, interleaver)
    decoder = NDFDecoder(code)
    ref_decoder =CplexTurboLikeDecoder(code, ip = False)
    
    gen = simulate.AWGNSignalGenerator(code, snr = -1, round=1)
    llr = next(gen)
    #llr = numpy.array([ 0.4, -0.6,  0.,  -0.1,  0.7,  1.8,  1.,   0.3, -2.2 , 0.7 , 2.5 , 1.8,  2.1,  0.5, -0.7, -0.4,  0.7,  0.6,  3.,  -0.6 , 2.4 ,-0.3 ,-0.8])
    print(llr)
    decoder.decode(llr)
    ref_decoder.decode(llr)
    print('real solution: {0}'.format(ref_decoder.objectiveValue))
    print(ref_decoder.solution)