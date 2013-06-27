#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
import numpy as np

from branchbound import problem, bnb, nodeselection, branchrules
from lpdecoding.codes.interleaver import Interleaver
from lpdecoding.codes.turbolike import StandardTurboCode, LTETurboCode
from lpdecoding.codes.convolutional import TDInnerEncoder
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder

import unittest
from lpdecoding.channels import SignalGenerator, AWGNC

RANDOM_TRIALS = 100

class TestTurboDecoder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        interleaver = Interleaver.random(30)
        self.code = StandardTurboCode(TDInnerEncoder(), interleaver, "smallTestCode")
        self.checkDecoder = CplexTurboLikeDecoder(self.code, ip=True)
        self.problem = problem.CSPTurboLPProblem(self.code)
    
    def setUp(self):
        self.channel = AWGNC(snr=-1, coderate=self.code.rate, seed=3874)
        self.generator = SignalGenerator(self.code, self.channel, randomCodewords=True, wordSeed=39847) 
    
    def test_BreadthFirstSearch(self):
        self.runBNBTest(nodeselection.BFSMethod)
        
    def test_DepthFirstSearch(self):
        self.runBNBTest(nodeselection.MyDFSMethod)
        
    def test_DeepSeaTroll(self):
        self.runBNBTest(nodeselection.DSTMethod)
        
    def test_BestBoundMethod(self):
        self.runBNBTest(nodeselection.BBSMethod)

    def runBNBTest(self, selectionMethod):
        totalBranch, totalMove = 0, 0
        for i in range(RANDOM_TRIALS):
            llr = next(self.generator)
            self.problem.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(self.problem, selectionMethod, branchrules.FirstFractional, eps=1e-10)
            solution = algo.run()
            self.checkDecoder.decode(llr)
            self.assert_(np.allclose(self.checkDecoder.solution,
                                     solution),
                         "{} ({}) != {} ({})".format(solution,
                                                     algo.optimalObjectiveValue,
                                                     self.checkDecoder.solution,
                                                     self.checkDecoder.objectiveValue))
            totalBranch += algo.branchCount
            totalMove += algo.moveCount
    
    def dont_test_medium_code(self):
        code = LTETurboCode(40)
        checkDecoder = CplexTurboLikeDecoder(code, ip=True)
        channel = AWGNC(snr=-1, coderate=code.rate)
        prob = problem.CplexTurboLPProblem(code)
        gen = SignalGenerator(code, channel, randomCodewords=True)
        RANDOM_TRIALS = 1
        for i in range(RANDOM_TRIALS):
            llr = next(gen)
            prob.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(prob, depthFirst=True)
            sol = algo.run()
            checkDecoder.decode(llr)
            self.assert_(np.allclose(checkDecoder.solution, sol),
                         "{} ({}) != {} ({})".format(sol, algo.optimalObjectiveValue,
                                                     checkDecoder.objectiveValue, checkDecoder.solution))
        
if __name__ == "__main__":
    unittest.main()