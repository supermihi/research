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
    def __init__(self, k, radius):
        self.points = []
        for coordinates in itertools.product(range(-radius,radius+1),repeat=k):
            if random.random() > .5 and numpy.linalg.norm(coordinates) <= radius:
                self.points.append(numpy.array(coordinates + (random.randint(-10,10),), dtype = numpy.double))
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
            coefficients = numpy.random.randint(-5,5,dimension)
            #coefficients /= numpy.linalg.norm(coefficients)
            rhs = numpy.random.randint(dimension**2+1)
            print(coefficients, rhs)
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
        print("Real LP solution: {1} [{0}]".format(self.cplex.solution.get_values(self.x), self.cplex.solution.get_objective_value()))
        self.cplex.linear_constraints.delete("side{0}".format(i) for i in range(dimension-1))
        self.cplex.set_results_stream(None)
    
    def __call__(self, direction):
        self.cplex.objective.set_linear(zip(self.x, direction))
        self.cplex.solve()
        if self.cplex.solution.get_status() == 2: #unbounded
            raise ValueError("optimization problem unbounded")
        return numpy.array(self.cplex.solution.get_values(self.x))
            
        
        
def zeroInConvexHull(points):
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

def zeroInAffineHull(points):
    cplex = cpx.Cplex()
    cplex.set_results_stream(None)

    lamb = ["lamb{0}".format(i) for i in range(len(points)) ]
    dim = len(points[0])
    x = ["x{0}".format(j) for j in range(dim)]
    cplex.variables.add(types=[cplex.variables.type.continuous]*len(points),
                        lb = [-cpx.infinity]*len(points),
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
    distance = cplex.solution.get_objective_value()
    return distance < 1e-6  
                                     
    

class NondominatedFacetAlgorithm:
    
    def __init__(self, k, solver):
        """*k*: number of constraints
           *solver*: method taking a (k+1) dimensional vector that defines the weighted sum scalarization,
                     solves the underlying combinatorial optimization problem, and returns a (k+1) dimen-
                     sional vector which contains the values of the *k* constraints plus the original
                     objective value."""
        self.k = k
        self.solver = solver
        
    def phase1_old(self):
        multipliers = numpy.zeros(self.k+1)
        multipliers[self.k] = 1
        min_c = self.solver(multipliers)
        print('1st point: {0}'.format(min_c))
        self.minus_infinity = numpy.zeros(self.k+1)
        self.minus_infinity[-1] = 10*min_c[-1]
        if numpy.linalg.norm(min_c[:-1]) < EPS:
            # min c^T x fulfills all constraints -> done
            print('initial solution {0} is feasible'.format(min_c))
            return False

        points = [min_c]
        points_p = [min_c[:-1]]
        print('first point norm: {0}'.format(numpy.linalg.norm(points_p[0])))
        direction = numpy.array(min_c)
        direction /= numpy.linalg.norm(direction[:-1])
        direction[-1] = EPS
        onb = []
        onb_p = []
        
        while True:
            #print('  searching in direction {0}'.format(direction))
            print('    distance from z axis: {0}'.format(numpy.dot(points_p[0], direction[:-1])))
            newPoint = self.solver(direction)
            print('found {0}. point: {1}'.format(len(points)+1, newPoint))
            #print('{0}. point found'.format(len(points)+1))

            newBasisVec = orthogonalize(onb, newPoint - points[0])
            norm = numpy.linalg.norm(newBasisVec)
            if norm < EPS:
                print('basis norm is zero -- linearly dependent!')
                break
            #print('    new vector for ONB: {0}'.format(newBasisVec/norm))
            onb.append(newBasisVec/norm)
            points.append(newPoint)
            points_p.append(newPoint[:-1])
            
            newBasisVec_p = orthogonalize(onb_p, newPoint[:-1] - points[0][:-1])
            norm_p = numpy.linalg.norm(newBasisVec_p)
            if norm < EPS:
                print('basis_p norm is zero -- wtf!!')
                break
            onb_p.append(newBasisVec_p/norm_p)
            direction = orthogonalize(onb_p, points_p[0])
            cplexResult, cpxDirection = zeroInConvexHull(points_p)
            #direction = cpxDirection 
            #print('   unscaled direction: {0}'.format(repr(direction)))
            #print('   cplex direction: {0}'.format(cpxDirection))
            directionDifference = numpy.linalg.norm(direction-cpxDirection)
            if directionDifference > 1e-6:
                print('OH NO!: {0} ({1} < {2})'.format(directionDifference, numpy.linalg.norm(direction), numpy.linalg.norm(cpxDirection)))
                for point in points_p:
                    print(numpy.dot(point, direction))
                return "omgwtf"                    
                    
            norm = numpy.linalg.norm(direction)
            if norm < 1e-6:
                print('zero in convex hull -- phase1 completed')
                cplexResult = zeroInConvexHull(points_p)
                if cplexResult:
                    print('-->cplex agrees')
                else:
                    cpxRes2 = zeroInAffineHull(points_p)
                    if cpxRes2:
                        print('affine :(')
                break
            elif len(points) > self.k:
                print(norm)
                print('k points and still no zero in convex hull -- problem infeasible?')
                return False
            else:
                direction = numpy.concatenate((direction/norm, [EPS]))
        self.points = points
        self.onb = onb
        #print(points)
        #print(onb)
        return True
    
    def phase1(self):
        """The algorithm described in "Finding the Nearest Point in a Polytope" by P. Wolfe,
        Mathematical Programming 11 (1976), pp.128-149."""
        # find the initial point: minimize over last axis (original c objective function)
        # (step 0)
        direction = numpy.zeros(self.k+1)
        direction[self.k] = 1
        min_z = self.solver(direction)
        print('step 0: P_J = {0}'.format(repr(min_z)))
        P = min_z[:-1].reshape((self.k,1))

        w = numpy.ones((1,1),dtype=numpy.double)
        X = numpy.empty((self.k,1), dtype=numpy.double)
        e1 = numpy.zeros((self.k+1,1),dtype=numpy.double)
        e1[0,0] = 1
        while True:
            # step 1
            print(P)
            print(w)
            print(X)
            numpy.dot(P, w, X)
            print('step 1: X = {0}'.format(X))
            P_J = self.solver(numpy.hstack((X.ravel(), [EPS])))
            if numpy.dot(X.T, P_J[:-1]) > numpy.dot(X.T, X) -EPS:
                print('stop in 1 (c)')
                break
            P = numpy.hstack((P, P_J[:-1].reshape((self.k,1))))
            w = numpy.vstack((w, 0))
            while True:
                result = numpy.linalg.lstsq(numpy.vstack((numpy.ones(P.shape[1]), P)), e1)
                print(result)
                u = result[0]
                print("least-square solution: {0}".format(u))
                v = u / numpy.sum(u)
                if numpy.all(v > EPS):
                    w = v
                    break
                else:
                    POS = numpy.nonzero(w-v > EPS) # 3 (a)
                    theta = min(1, numpy.min(w[POS]/(w[POS] - v[POS]))) # 3 (b)
                    print(theta)
                    w = theta*w + (1-theta)*v # 3 (c)
                    w[numpy.nonzero(w<=EPS)] = 0 # 3 (d)
                    print('step 3 (d): w = {0}'.format(w))
                    firstZeroIndex = numpy.nonzero(w == 0)[0][0]
                    w = numpy.delete(w, firstZeroIndex, 0)
                    P = numpy.delete(P, firstZeroIndex, 1)
                    
        print(zeroInConvexHull([p.ravel() for p in numpy.hsplit(P, P.shape[1])]))
                    
                    
        
        
        
    def phase2(self):
        while len(self.points) < self.k+1:
            print('  warning: not full dimensional')
            direction = orthogonalize(self.onb, self.minus_infinity)
            direction /= numpy.linalg.norm(direction)
            b = numpy.dot(self.points[0], direction)
            z = b / direction[-1]
            print('current z: {0}'.format(z))
            newPoint = self.solver(direction)
            print('new z: {0}'.format(numpy.dot(newPoint, direction)))
            print('new point\'s z: {0}'.format(newPoint[-1]))
            newBasisVec = orthogonalize(self.onb, newPoint - self.points[0])
            norm = numpy.linalg.norm(newBasisVec)
            if norm < EPS:
                print('new vector on old facet -- FOUND OPTIMAL FACET!')
                break
            self.onb.append(newBasisVec/norm)
            self.points.append(newPoint)
            
    
if __name__ == "__main__":
    seed = numpy.random.randint(1,10000000)
    seed = 6417330
    numpy.random.seed(seed)
    print(seed)
    
    k = 5
    #solver = DummySolver(k, 3)
    solver = RandomLPSolver(k+1, k**2)
    #sys.exit(0)
    nda = NondominatedFacetAlgorithm(k, solver)
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