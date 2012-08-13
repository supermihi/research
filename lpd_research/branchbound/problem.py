# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder

class Problem:
    
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


class CplexTurboLPProblem(Problem):
    
    def __init__(self, code):
        super().__init__()
        self.decoder = CplexTurboLikeDecoder(code, ip=False)
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        
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
            
    def unfixVariable(self, var):
        self.decoder.cplex.variables.set_lower_bounds(self.decoder.x[var], 0)
        self.decoder.cplex.variables.set_upper_bounds(self.decoder.x[var], 1)
