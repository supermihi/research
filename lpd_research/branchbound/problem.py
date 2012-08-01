# -*- coding: utf-8 -*-
# Copyright 2012 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np

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
        pass