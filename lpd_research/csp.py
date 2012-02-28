#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, division
from lpdecoding.codes import turbolike, trellis, interleaver
import numpy, numpy.linalg, cplex as cpx
import itertools, random, sys

SIZE = 40
#code = turbolike.StandardTurboCode(encoder = trellis.LTE_Encoder(), interleaver = interleaver.lte_interleaver(SIZE), name = 'test')

EPS = 1e-8

def orthogonalize(onb, newVector):
    """Returns a vector which is orthogonal to all vectors in *onb* and spans the same space
    as onb ∪ {*newVector*}."""
    result = numpy.array(newVector)
    for basisVec in onb:
        result -= numpy.dot(newVector, basisVec)*basisVec
    return result

class DummySolver:
    def __init__(self):
        self.points = []
        for coordinates in itertools.product(range(-3,4),repeat=5):
            if random.random() > .5 and numpy.linalg.norm(coordinates) <= 5:
                self.points.append(numpy.array(coordinates + (random.randint(-3,3),), dtype = numpy.double))
        #print("\n".join(map(str,self.points)))
    
    def __call__(self, direction):
        minPoint = None
        min = numpy.Inf
        for point in self.points:
            dotProd = numpy.dot(direction, point)
            if dotProd < min:
                min = dotProd
                minPoint = point
        return minPoint

class RandomLPSolver:
    def __init__(self, dimension, numFacets):
        self.cplex = cpx.Cplex()
        self.dimension = dimension
        self.x = ["x{0}".format(i) for i in range(dimension)]
        self.cplex.variables.add(types=[self.cplex.variables.type.continuous]*len(self.x), names = self.x, lb = [-cpx.infinity]*len(self.x));
        self.cplex.set_problem_type(self.cplex.problem_type.LP)
        facets = []
        rhss = []
        for i in range(numFacets):
            coefficients = numpy.random.randn(dimension)
            coefficients /= numpy.linalg.norm(coefficients)
            rhs = numpy.random.randint(2,8)
            facets.append(coefficients)
            rhss.append(rhs)
        self.cplex.linear_constraints.add(names = ["facet{0}".format(facet) for facet in range(numFacets)],
                                          rhs = rhss,
                                          senses = "L"*numFacets,
                                          lin_expr = [cpx.SparsePair(ind = self.x, val = facet) for facet in facets])
        
        # determine solution
        self.cplex.linear_constraints.add(names = ["side{0}".format(i) for i in range(dimension-1) ],
                                          rhs = [0]*(dimension-1),
                                          senses = "E"*(dimension-1),
                                          lin_expr = [cpx.SparsePair(ind=(self.x[i],), val=(1,)) for i in range(dimension-1)])
        self.cplex.objective.set_sense(self.cplex.objective.sense.minimize)
        self.cplex.objective.set_linear(self.x[-1], 1)
        self.cplex.solve()
        print("Real LP solution: {0}".format(self.cplex.solution.get_objective_value()))
        self.cplex.linear_constraints.delete("side{0}".format(i) for i in range(dimension-1))
        self.cplex.set_results_stream(None)
    
    def __call__(self, direction):
        self.cplex.objective.set_linear(zip(self.x, direction))
        self.cplex.solve()
        return numpy.array(self.cplex.solution.get_values(self.x))
            
        
        
def zAxisInConvexHull(points):
    cplex = cpx.Cplex()
    cplex.set_results_stream(None)
    #cplex.set_warning_stream(None)
    #cplex.set_error_stream(None)
    x = ["x{0}".format(i) for i in range(len(points)) ]
    dim = len(points[0])
    cplex.variables.add(types=[cplex.variables.type.continuous]*len(points),
                        lb = [0]*len(points),
                        names = x)
    cplex.variables.add(types=[cplex.variables.type.continuous], names = ["z"], lb = [-cpx.infinity])
    cplex.set_problem_type(cplex.problem_type.LP)
    cplex.linear_constraints.add(names = ["cons{0}".format(j) for j in range(dim-1)],
                                 rhs = [0]*(dim-1),
                                 senses = "E"*(dim-1),
                                 lin_expr = [cpx.SparsePair(ind = x, val = [ points[i][j] for i in range(len(x))]) for j in range(dim-1)],
                                 )
    cplex.linear_constraints.add(names = ["consZ"],
                                 rhs = [0],
                                 senses = "E",
                                 lin_expr = [cpx.SparsePair(ind = x + ["z"], val = [ points[i][len(points[0])-1] for i in range(len(x))] + [-1]) ]
                                 )
    cplex.linear_constraints.add(names = ["conv"],
                                 rhs = [1],
                                 senses = "E",
                                 lin_expr = [ cpx.SparsePair(ind = x, val = [1]*len(x))])
    cplex.solve()
    return cplex.solution.is_primal_feasible()
                                     
    

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
        print('1st point: {0}'.format(min_c))
        self.minus_infinity = numpy.zeros(self.k+1)
        self.minus_infinity[-1] = 2*min_c[-1]
        if numpy.linalg.norm(min_c[:-1]) < EPS:
            # min c^T x fulfills all constraints -> done
            print('initial solution {0} is feasible'.format(min_c))
            return False
        points = [min_c]
        points_p = [min_c[:-1]]
        direction = numpy.array(min_c)
        direction[-1] = EPS
        onb = []
        onb_p = []
        
        while True:
            print('  searching in direction {0}'.format(direction))
            for point in points_p:
                print('    scalar product: {0}'.format(numpy.dot(point, direction[:-1])))
            newPoint = self.solver(direction)
            print('found {0}. point: {1}'.format(len(points)+1, newPoint))

            newBasisVec = orthogonalize(onb, newPoint - points[0])
            norm = numpy.linalg.norm(newBasisVec)
            if norm < EPS:
                print('basis norm is zero -- linearly dependent!')
                break
            print('    new vector for ONB: {0}'.format(newBasisVec/norm))
            onb.append(newBasisVec/norm)
            points.append(newPoint)
            points_p.append(newPoint[:-1])
            cplexResult = zAxisInConvexHull(points)
            print('    cplex says: {0}'.format(cplexResult))
            
            newBasisVec_p = orthogonalize(onb_p, newPoint[:-1] - points[0][:-1])
            norm_p = numpy.linalg.norm(newBasisVec_p)
            print('  new basis_p norm: {0}'.format(norm))
            if norm < EPS:
                print('basis_p norm is zero -- wtf!!')
                break
            onb_p.append(newBasisVec_p/norm_p)
            direction = orthogonalize(onb_p, points_p[0])
            norm = numpy.linalg.norm(direction)
            if norm < EPS:
                print('zero in convex hull -- phase1 completed')
                break
            elif len(points) > self.k:
                print('k points and still no zero in convex hull -- problem infeasible?')
                return False
            else:
                direction = numpy.concatenate((direction/norm, [EPS]))
        self.points = points
        self.onb = onb
        print(points)
        print(onb)
        return True
        
    def phase2(self):
        assert len(self.points) == self.k+1
        direction = orthogonalize(self.onb, self.minus_infinity)
        direction /= numpy.linalg.norm(direction)
        b = numpy.dot(self.points[0], direction)
        z = b / direction[-1]
        print('current z: {0}'.format(z))
        newPoint = self.solver(direction)
        print('new z: {0}'.format(numpy.dot(newPoint, direction)))
        print('new point\'s z: {0}'.format(newPoint[-1]))
    
if __name__ == "__main__":
    #random.seed(1337)
    #solver = DummySolver()
    solver = RandomLPSolver(5, 25)
    #sys.exit(0)
    nda = NondominatedFacetAlgorithm(4, solver)
    if nda.phase1():
        nda.phase2()
    
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