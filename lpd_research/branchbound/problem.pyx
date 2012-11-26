# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function
import logging

import numpy as np
cimport numpy as np
import cplex

from lpdecoding.core cimport Code, Decoder
from lpresearch cimport cspdecoder
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
from lpdecoding.codes import turbolike, trellis

cdef class Problem(object):
    
    cdef public np.ndarray solution
    cdef public double objectiveValue
    
    def __init__(self):
        self.solution = None
        self.objectiveValue = np.inf 
    
    def solve(self):
        """Solve the current problem.
        
        After solving, attributes *solution* and *objectiveValue* are available. If the
        problem is infeasible, solution is None and objectiveValue=inf."""
        pass
    
    def fixVariable(self, var, value):
        pass
    
    def unfixVariable(self, var):
        pass
    
    def fixVariables(self, fixes):
        """Fix several variables in one step, defined by a list of (var, val) tuples.
        
        The default implementation calls fixVariable len(fixes) times. Subclasses may
        do this more efficiently.
        """
        for var, value in fixes:
            self.fixVariable(var, value)
            
    def unfixVariables(self, vars):
        """Unfix several variables at once.
        
        Like *fixVariables*, the default implenetation just calls unfixVariable for each
        variable separately.
        """
        for var in vars:
            self.unfixVariable(var)
    
    def setObjectiveFunction(self, c):
        """Set the objective function to vector *c*, a numpy array."""
        pass

cdef class CSPTurboLPProblem(Problem):
    
    cdef Code code
    cdef Decoder decoder
    
    def __init__(self, code):
        Problem.__init__(self)
#        self.checkProblem = CplexTurboLPProblem(code)
        self.decoder = cspdecoder.CSPDecoder(code)
        self.code = code
        
    def setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c):
#        self.checkProblem.setObjectiveFunction(c)
        self.decoder.llrVector = c
        self.unfixVariables(range(self.code.blocklength))
        
    def solve(self):
#        self.checkProblem.solve()        
        self.decoder.solve()
        self.objectiveValue = self.decoder.objectiveValue
        if self.objectiveValue == np.inf:
            self.solution = None
        else:
            self.solution = self.decoder.solution
#        if not np.isclose(self.objectiveValue, self.checkProblem.objectiveValue, rtol=1e-5):
#            print('unequal: {} != {}'.format(self.objectiveValue, self.checkProblem.objectiveValue))
#            print(self.printFixes())
#            print(self.checkProblem.printFixes())
#            print(self.solution)
#            self.decoder.printSolutionParts()
#            print(self.code.encoders[1].trellis[8].fix_parity)
#            raw_input() 
        
    def fixVariable(self, int var, int value):
#        self.checkProblem.fixVariable(var, value)
        self.code.fixCodeBit(var, value)
        
    def unfixVariable(self, int var):
#        self.checkProblem.unfixVariable(var)
        self.code.fixCodeBit(var, -1)       


class CplexTurboLPProblem(Problem):
    
    def __init__(self, code):
        Problem.__init__(self)
        self.code = code
        self.decoder = CplexTurboLikeDecoder(code, ip=False)
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        self.unfixVariables(self.decoder.x)
        
    def solve(self):
        try:
            self.decoder.solve()
            if not self.decoder.cplex.solution.is_primal_feasible():
                self.solution = None
                self.objectiveValue = np.inf
            else:
                self.solution = self.decoder.solution
                self.objectiveValue = self.decoder.objectiveValue
        except cplex.exceptions.CplexSolverError as e:
            self.solution = None
            self.objectiveValue = np.inf

    def fixVariable(self, var, value):
        if value == 0:
            self.decoder.cplex.variables.set_upper_bounds(self.decoder.x[var], 0)
        else:
            self.decoder.cplex.variables.set_lower_bounds(self.decoder.x[var], 1)

    def fixVariables(self, fixes):
        """Fix given variables, where *fixes* is a list of (var, value) pairs."""
        zeroFixes = [ tup for tup in fixes if tup[1] == 0 ]
        oneFixes = [ tup for tup in fixes if tup[1] == 1 ]
        if len(zeroFixes) > 0:
            self.decoder.cplex.variables.set_upper_bounds(zeroFixes)
        if len(oneFixes) > 0:
            self.decoder.cplex.variables.set_lower_bounds(oneFixes)

    def unfixVariable(self, var):
        self.decoder.cplex.variables.set_lower_bounds(self.decoder.x[var], 0)
        self.decoder.cplex.variables.set_upper_bounds(self.decoder.x[var], 1)

    def unfixVariables(self, vars):
        """Unfix given *vars*."""
        self.decoder.cplex.variables.set_lower_bounds([ (var, 0) for var in vars])
        self.decoder.cplex.variables.set_upper_bounds([ (var, 1) for var in vars])
        
    def printFixes(self):
        ret = ''
        for i in range(self.decoder.code.blocklength):
            #print(self.decoder.cplex.variables.get_upper_bounds(i))
            #print(self.decoder.cplex.variables.get_lower_bounds(i))
            if self.decoder.cplex.variables.get_upper_bounds(self.decoder.x[i]) == 0:
                ret += '{}=0, '.format(i)
            elif self.decoder.cplex.variables.get_lower_bounds(self.decoder.x[i]) == 1:
                ret += '{}=1, '.format(i)
        return ret[:-2] + '\n'