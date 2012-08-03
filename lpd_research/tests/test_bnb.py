# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np

from branchbound import problem, bnb
from lpdecoding import *
from lpdecoding.codes.trellis import TDInnerEncoder
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder

import unittest

class ExampleProblem(problem.Problem):
    
    def __init__(self):
        interleaver = Interleaver(permutation=[5, 0, 9, 7, 2, 1, 8, 6, 3, 4])
        self.code = StandardTurboCode(TDInnerEncoder(), interleaver, "smallTestCode")
        self.decoder = CplexTurboLikeDecoder(self.code, ip=False) # LP decoder
        llrVector = np.array([ 2.42016923, 1.38835975, 3.83660991, 0.58480805, -0.36203485,
                              -0.21801139, -2.92970653, 2.42626591, 1.19869679, -1.40685654,
                              2.08236401, 0.93360395, 2.93510648, 2.0791263, 0.86044521,
                              -1.15049985, -0.06952653, 1.8066568, 1.68665064, -0.12933509,
                              -0.47870521, 0.7124441, -0.20058202, 0.24883215, -1.85805523,
                              1.20155129, -1.99822707, 0.09290395, 4.38302952, 2.5280745,
                              1.13024014, 1.80107028, 2.28149111, 1.9960362, 1.63789898,
                              1.47294491, 8.38113826, 0.32048955])
        self.decoder.llrVector = llrVector
        self.ipSolution = np.array(map(float, ("0.  1.  0.  0.  1.  1.  1.  0.  1.  0.  1.  0.  0.  1.  0.  1.  1.  0. \
                                   0.  0.  1.  0.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.  1.  0. \
                                   0.  0.".split())))
        
    def setObjectiveFunction(self, c):
        self.decoder.llrVector = c
        
    def solve(self):
        self.decoder.solve()
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

class TestTurboDecoder(unittest.TestCase):
    
    def test_decoder(self):
        testProblem = ExampleProblem()
        algorithm = bnb.BranchAndBound(testProblem, depthFirst=True)
        sol = algorithm.run()
        self.assert_(np.all(sol==testProblem.ipSolution))
        
        
if __name__ == "__main__":
    unittest.main()