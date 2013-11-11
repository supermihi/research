# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from branchbound.problem import Problem
from lpdecoding.decoders.siegeldecoders import ZhangSeparationDecoder
from lpdecoding.decoders.iterative import SumProductDecoder

import numpy as np

class LDPCLPProblem(Problem):
    
    def __init__(self, code, pureLP=False):
        Problem.__init__(self)
        self.decoder = ZhangSeparationDecoder(code, pureLP=pureLP)
        self.code = code
        self.iterDecoder = SumProductDecoder(code, maxIterations=200)
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        self.iterDecoder.llrVector = c
        self.unfixVariables(range(self.code.blocklength))
        
    def solve(self, lb=1, ub=0):        
        self.decoder.solve(hint=None, lb=lb)
        self.decoder.fix()
        self.objectiveValue = self.decoder.objectiveValue
        if self.objectiveValue == np.inf:
            self.solution = None
            return -1
        else:
            self.solution = self.decoder.solution
        if self.decoder.foundCodeword:
            return 1
        return 0
    
    def solveHeuristic(self):
        self.iterDecoder.solve()
        if self.iterDecoder.foundCodeword:
            self.hSolution = self.iterDecoder.solution
            self.hObjectiveValue = self.iterDecoder.objectiveValue
            return 1
        return 0 
        
    def fixVariable(self, var, value):
        #print('fix {} {}'.format(var, value))
        self.decoder.fixes[var] = value
        self.iterDecoder.fix(var, value)
        
    def unfixVariable(self, var):
        #print('unfix {}'.format(var))
        self.decoder.fixes[var] = -1
        self.iterDecoder.unfix(var)