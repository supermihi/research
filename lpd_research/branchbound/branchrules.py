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
    
    def selectVariable(self, solution):
        for (i, x) in enumerate(solution):
            if x > 1e-10 and x < 1 - 1e-10:
                return i
        return None
    

    
class MostFractional(BranchingRule):
    """Rule that selects the variable maximizing x - [x]."""
    
    def selectVariable(self, solution):
        index = np.argmin(np.abs(solution-0.5))
        if solution[index] < 1e-10 or solution[index] > 1-1e-10:
            return None
        return index
    
    
class LeastReliable(BranchingRule):
    
    def selectVariable(self, solution):
        for index in np.argsort(np.abs(self.problem.decoder.llrVector)):
            x = solution[index]
            if x > 1e-10 and x < 1 - 1e-10:
                return index
        return None