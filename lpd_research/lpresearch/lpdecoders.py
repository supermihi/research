# -*- coding: utf-8 -*-
# Copyright 2011 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, division
from lpdecoding.core import Decoder
import numpy
from numpy.linalg import norm
import cplex
from . import cplexhelpers

class DualLPDecoder(Decoder):
    """The classical LP decoder defined by Feldman, in its 'explicit' definition via Forbidden Set
    Inequalities. Note that the size of this formulation is exponential in the check node degree,
    so polynomial running time is guaranteed only for LDPC codes."""
    
    def __init__(self, code):
        self.code = code
        self.cplex = cplex.Cplex()
        matrix = code.parityCheckMatrix
        self.cplex.linear_constraints.add( names = [ 'c{0}'.format(i) for i in range(code.blocklength)], senses = 'G'*code.blocklength)
        for row in matrix.A:
            N_i = [ i for i in range(len(row)) if row[i] == 1]
            for i in N_i:
                self.cplex.variables.add( obj = [0],
                                          lb = [0],
                                          ub = [cplex.infinity],
                                          types = [self.cplex.variables.type.continuous],
                                          columns = [cplex.SparsePair( ind = N_i, val = [1 if pos == i else -1 for pos in N_i])])
        #for x in range(code.blocklength):
        #    self.cplex.variables.add( obj = [0], lb = [0], columns = [cplex.SparsePair(ind = [x], val = [-1])])
        self.cplex.objective.set_sense(self.cplex.objective.sense.minimize)
        self.cplex.set_results_stream(None)
        self.cplex.set_warning_stream(None)
        self.cplex.set_error_stream(None)
        self.solution = numpy.zeros((1,code.blocklength))
    
    def solve(self):
        for num,val in enumerate(self.llrVector):
            self.cplex.linear_constraints.set_rhs(num, -val)
        self.cplex.solve()
        cplexhelpers.checkKeyboardInterrupt(self.cplex)
        if self.cplex.solution.is_primal_feasible():
            self.objectiveValue = 0
            self.solution[0] = 0
        else:
            self.objectiveValue = -1
            self.solution[0] = 1
    
    def addForbiddenSetInequality(self, position, complement, lpRow):
        for col in complement:
            self.lpmatrix[lpRow,col] = -1
            self.lpmatrix[lpRow,position] = 1
    def __str__(self):
        return "test"

class CplexL1Maximizer(Decoder):
        
    def __init__(self, code):
        self.code = code
        self.cplex = cplex.Cplex()
        matrix = code.parityCheckMatrix
        self.xp = [ "xp" + str(num) for num in range(matrix.shape[1]) ] # one x-positive-var for each column
        self.xn = [ "xn" + str(num) for num in range(matrix.shape[1]) ] # one x-negative-var for each column
        self.cplex.variables.add(
            types=[self.cplex.variables.type.continuous]*2*code.blocklength,
            names= self.xp + self.xn,
            lb = numpy.zeros(2*code.blocklength),
            ub = numpy.ones(2*code.blocklength)*numpy.inf)
        for row in matrix:
            N_i = numpy.nonzero(row)[0]
            for i in N_i:
                self.addForbiddenSetInequality([i], list(set(N_i) - set([i])))
        
        self.cplex.objective.set_sense(self.cplex.objective.sense.maximize)
        
        self.cplex.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = self.xp + self.xn, val = 
                                                                       numpy.hstack((numpy.ones(code.blocklength), -numpy.ones(code.blocklength))))],
                                          names = ['cone'],
                                          senses = 'E',
                                          rhs = [0])
        #self.cplex.set_results_stream(None)
        #self.cplex.set_warning_stream(None)
        #self.cplex.set_error_stream(None)
        self.cplex.set_problem_type(self.cplex.problem_type.LP)
    
    def minimumPseudoWeight(self):
        self.cplex.objective.set_linear( zip(range(2*self.code.blocklength), numpy.ones(2*self.code.blocklength) ))
        self.cplex.write('test.lp')
        self.cplex.solve()
        print('omg')
        return self.cplex.solution.get_objective_value()
    
    def addForbiddenSetInequality(self, subset, complement):
        assert len(subset) == 1
        delta = len(subset) - len(complement)
        ind = [self.xp[i] for i in subset + complement] + [self.xn[i] for i in subset + complement]
        val = [1]*len(subset) + [-1]*len(complement) + [-1]*len(subset) + [1]*len(complement)
        self.cplex.linear_constraints.add( senses = 'L',
                                           rhs = [len(subset) - 1 - delta/self.code.blocklength],
                                           lin_expr = [cplex.SparsePair( ind, val)])
    
    def solve(self):
        self.cplex.objective.set_linear( zip(self.x, self.llrVector) )
        self.cplex.solve()
        cplexhelpers.checkKeyboardInterrupt(self.cplex)
        self.objectiveValue = self.cplex.solution.get_objective_value()
        self.solution = numpy.array(self.cplex.solution.get_values(self.x))
    
    def __str__(self):
        return 'L1Max'
        
class FastLPDecoder(Decoder):
    def __init__(self, code, dfrac):
        self.code = code
        self.dfrac = dfrac
        self.dint, self.drem = numpy.modf(dfrac)
        self.solution = numpy.zeros((1,code.blocklength))
    
    def solve(self):
        
        sortedLLR = numpy.sort(self.llrVector)
        theSum = sortedLLR[:self.dint].sum() + sortedLLR[self.dint]*self.drem
        if theSum > 0:
            self.solution[0] = 0
            self.objectiveValue = 0
        else:
            self.solution[0] = 1
            self.objectiveValue = -42
        
    def addForbiddenSetInequality(self, position, complement, lpRow):
        for col in complement:
            self.lpmatrix[lpRow,col] = -1
            self.lpmatrix[lpRow,position] = 1