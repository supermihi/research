# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from branchbound.problem import Problem
from lpdecoding.decoders.zhangsiegel import ZhangSiegelACG
from lpdecoding.decoders.iterative import IterativeDecoder

import numpy as np


class LDPCLPProblem(Problem):
    
    def __init__(self, code, **kwargs):
        Problem.__init__(self)
        self.decoder = ZhangSiegelACG(code, **kwargs)
        self.code = code
        self.iterDecoder = IterativeDecoder(code, minSum=False, maxIterations=1000)
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        self.iterDecoder.llrVector = c
        self.unfixVariables(range(self.code.blocklength))
        
    def solve(self, lb=1, ub=0):        
        self.decoder.solve(hint=None, lb=lb)
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
        self.decoder.fix(var, value)
        self.iterDecoder.fix(var, value)
        
    def unfixVariable(self, var):
        self.decoder.release(var)
        self.iterDecoder.release(var)
