#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
import numpy as np

from branchbound import problem, bnb
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
        self.problem = problem.CplexTurboLPProblem(self.code)
    
    def setUp(self):
        self.channel = AWGNC(snr=-1, coderate=self.code.rate, seed=3874)
        self.generator = SignalGenerator(self.code, self.channel, randomCodewords=True, wordSeed=39847) 
    
    def utest_BreadthFirstSearch(self):
        self.runBNBTest(bnb.BFSMethod)
        
    def utest_DepthFirstSearch(self):
        self.runBNBTest(bnb.DFSMethod)
        
    def utest_DeepSeaTroll(self):
        self.runBNBTest(bnb.DSTMethod)
        
    def test_BestBoundMethod(self):
        self.runBNBTest(bnb.BBSMethod)

    def runBNBTest(self, branchMethod):
        totalBranch, totalMove = 0, 0
        for i in range(RANDOM_TRIALS):
            llr = next(self.generator)
            self.problem.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(self.problem, eps=1e-10, branchMethod=branchMethod)
            solution = algo.run()
            self.checkDecoder.decode(llr)
            self.assert_(np.allclose(self.checkDecoder.solution,
                                     solution),
                         "{} ({}) != {} ({})".format(solution,
                                                     algo.optimalObjectiveValue,
                                                     self.checkDecoder.objectiveValue,
                                                     self.checkDecoder.solution))
            totalBranch += algo.branchCount
            totalMove += algo.moveCount
        print('{} average move count: {}'.format(branchMethod, totalMove / RANDOM_TRIALS))
        print('{} average branch count: {}'.format(branchMethod, totalBranch / RANDOM_TRIALS))
    
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