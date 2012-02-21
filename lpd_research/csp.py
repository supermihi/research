#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdecoding.codes import turbolike, trellis, interleaver
import numpy, numpy.linalg
import itertools, random

SIZE = 40
#code = turbolike.StandardTurboCode(encoder = trellis.LTE_Encoder(), interleaver = interleaver.lte_interleaver(SIZE), name = 'test')

EPS = 1e-8

def orthogonalize(onb, newVector):
    """Returns a vector which is orthogonal to all vectors in *onb* and spans the same space
    as onb ∪ {*newVector*}."""
    result = newVector
    for basisVec in onb:
        result -= numpy.dot(newVector[:-1], basisVec[:-1])*basisVec
    return result

class DummySolver:
    def __init__(self):
        self.points = []
        for coordinates in itertools.product(range(-3,4),repeat=3):
            if random.random() > .5 and numpy.linalg.norm(coordinates) <= 3:
                self.points.append(numpy.array(coordinates + (random.randint(-3,3),), dtype = numpy.double))
        print("\n".join(map(str,self.points)))
    
    def __call__(self, direction):
        minPoint = None
        min = numpy.Inf
        for point in self.points:
            dotProd = numpy.dot(direction, point)
            if dotProd < min:
                min = dotProd
                minPoint = point
        return minPoint


class NondominatedFacetAlgorithm:
    
    def __init__(self, k, solver):
        """*k*: number of constraints
           *solver*: method taking a (k+1) dimensional vector that defines the weighted sum scalarization,
                     solves the underlying combinatorial optimization problem, and returns a (k+1) dimen-
                     sional vector which contains the values of the *k* constraints plus the original
                     objective value."""
        self.k = k
        self.solver = solver
        
    def phase1(self):
        multipliers = numpy.zeros(self.k+1)
        multipliers[self.k] = 1
        min_c = self.solver(multipliers)
        print('min_c: {0}'.format(min_c))
        if numpy.linalg.norm(min_c[:-1]) < EPS:
            # min c^T x fulfills all constraints -> done
            print('initial solution {0} is feasible'.format(min_c))
            return
        points = [min_c]
        direction = numpy.array(min_c)
        direction[-1] = EPS
        onb = []
        while True:
            print('searching in direction {0}'.format(direction))
            newPoint = self.solver(direction)
            print('found {0}. point: {1}'.format(len(points)+1, newPoint))
            if len(points) <= self.k:
                newVec = newPoint - points[0]
                newBasis = orthogonalize(onb, newVec) # direction vector
                
                newBasis /= numpy.linalg.norm(newBasis)
                print('new vector for ONB: {0}'.format(newBasis))
                onb.append(newBasis)
                points.append(newPoint)
                direction = orthogonalize(onb, points[0])
                if numpy.allclose( points[0][:-1]/numpy.linalg.norm(points[0][:-1]), direction[:-1]/numpy.linalg.norm(direction[:-1]) ):
                    print('oh no, zero in convex hull before finished!')
                    break
                direction[-1] = EPS
            else:
                break
        print(points)
        print(onb)
        
                
            
                
        
    
    def phase2(self):
        pass
    
if __name__ == "__main__":
    random.seed(1337)
    solver = DummySolver()
    nda = NondominatedFacetAlgorithm(3, solver)
    nda.phase1()
    
#===============================================================================
# mu = numpy.randn(SIZE)
# 
# done = False
# while not done:
#    # calculate mu
#    code.enc_1.trellis.clearArcCosts()
#    code.enc_2.trellis.clearArcCosts()
#    cost1, path1 = algorithms.path.shortestPath(code.enc_1.trellis, mu, ...)
#    path1, path2 = # shortes path in trellis1, trellis2 wrt. mu
#===============================================================================