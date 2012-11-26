# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict

from lpdecoding.core import Decoder
from . import bnb, problem, nodeselection, branchrules

class BranchBoundTurboDecoder(Decoder):
    
    def __init__(self, code, name=None):
        Decoder.__init__(self, code)
        if name is None:
            name = "B&B Decoder"
        self.name = name
        self.problem = problem.CSPTurboLPProblem(code)
        self.mlCertificate = self.foundCodeword = True
        
    def solve(self, hint=None):
        self.problem.setObjectiveFunction(self.llrVector)
        self.bnb = bnb.BranchAndBound(self.problem, nodeselection.DSTMethod, branchrules.FirstFractional)
        self.solution = self.bnb.run()
        self.objectiveValue = self.bnb.optimalObjectiveValue
        return self.objectiveValue
    
    def params(self):
        return OrderedDict( [ ("name", self.name) ])