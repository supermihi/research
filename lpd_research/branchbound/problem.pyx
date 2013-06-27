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
    
    def __init__(self):
        self.solution = None
        self.objectiveValue = np.inf 
    
    cdef int solve(self, double lb=1, double ub=0):
        """Solve the current problem.
        
        After solving, attributes *solution* and *objectiveValue* are available. If the
        problem is infeasible, solution is None and objectiveValue=inf."""
        return 0
    
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
    
    def __init__(self, code):
        Problem.__init__(self)
        self.decoder = cspdecoder.CSPDecoder(code, heuristic=1, keepLP=True)
        self.code = code
        
    cpdef setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c):
        self.decoder.llrVector = c
        self.unfixVariables(range(self.code.blocklength))
        
    cdef int solve(self, double lb=1, double ub=0):
        self.decoder.threshold = ub
        self.decoder.solve(hint=None, lb=lb)
        if self.decoder.threshold == 1 and ub != 0:
            self.objectiveValue = ub
            self.solution = None
            return -2
        self.objectiveValue = self.decoder.objectiveValue
        if self.objectiveValue == np.inf:
            self.solution = None
            return -1
        else:
            self.solution = self.decoder.solution
            self.hSolution = self.decoder.hSolution
            self.hObjectiveValue = self.decoder.hObjectiveValue
        return 0 
        
    def fixVariable(self, int var, int value):
        self.code.fixCodeBit(var, value)
        
    def unfixVariable(self, int var):
        self.code.fixCodeBit(var, -1)       
