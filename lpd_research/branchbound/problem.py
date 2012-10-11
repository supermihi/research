# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder

class Problem(object):
    
    solution = None
    objectiveValue = np.inf 
    
    def solve(self):
        """Solve the current problem.
        
        After solving, attributes *solution* and *objectiveValue* are available. If the
        problem is infeasible, solution is None and objectiveValue=inf."""
        pass
    
    def fixVariable(self, var, value):
        pass
    
    def unfixVariable(self, var):
        pass
    
    def setObjectiveFunction(self, c):
        """Set the objective function to vector *c*, a numpy array."""
        pass

class CSPTurboLPProblem(Problem):
    def __init__(self, code):
        from lpresearch import cspdecoder
        Problem.__init__(self)
        self.decoder = cspdecoder.CSPDecoder(code)

class CplexTurboLPProblem(Problem):
    
    def __init__(self, code):
        Problem.__init__(self)
        self.decoder = CplexTurboLikeDecoder(code, ip=False)
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        self.unfixVariables(self.decoder.x)
        
    def solve(self):
        self.decoder.solve()
        if not self.decoder.cplex.solution.is_primal_feasible():
            self.solution = None
            self.objectiveValue = np.inf
        else:
            self.solution = self.decoder.solution
            self.objectiveValue = self.decoder.objectiveValue

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