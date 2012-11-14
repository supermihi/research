# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np

class BranchingRule:
    
    def __init__(self, problem):
        """Initialize the branching rule for *problem*."""
        self.problem = problem
    
    def selectVariable(self):
        pass
    
class FirstFractional(BranchingRule):
    """Simple rule that picks the first non-integral variable to branch on."""
    
    def selectVariable(self):
        for (i, x) in enumerate(self.problem.solution):
            if x > 1e-10 and x < 1 - 1e-10:
                return i
        return None
    
class MostFractional(BranchingRule):
    """Rule that selects the variable maximizing x - [x]."""
    
    def __init__(self, decoder):
        BranchingRule.__init__(self, decoder)
        self.infolength = decoder.code.infolength

    def selectVariable(self):
        index = np.argmin(np.abs(self.problem.solution[:self.infolength]-0.5))
        if self.problem.solution[index] < 1e-10 or self.problem.solution[index] > 1-1e-10:
            return None
        return index
    
class LeastReliable(BranchingRule):
    
    def __init__(self, decoder):
        BranchingRule.__init__(self, decoder)
        self.infolength = decoder.code.infolength
    
    def selectVariable(self):
        for index in np.argsort(np.abs(self.problem.decoder.llrVector[:self.infolength])):
            x = self.problem.solution[index]
            if x > 1e-10 and x < 1 - 1e-10:
                return index
        return None