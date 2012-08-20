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
from lpdecoding.channels import SignalGenerator

class TestTurboDecoder(unittest.TestCase):

    def test_small_code(self):
        interleaver = Interleaver(permutation=[5, 0, 9, 7, 2, 1, 8, 6, 3, 4])
        code = StandardTurboCode(TDInnerEncoder(), interleaver, "smallTestCode")
        checkDecoder = CplexTurboLikeDecoder(code, ip=True) 
        channel = AWGNC(snr=-1, coderate=code.rate, seed=3874)
        prob = problem.CplexTurboLPProblem(code)
        gen = SignalGenerator(code, channel, randomCodewords=True, wordSeed=39847)
        RANDOM_TRIALS = 100
        for i in range(RANDOM_TRIALS):
            llr = next(gen)
            prob.setObjectiveFunction(llr)
            algoDFS = bnb.BranchAndBound(prob, branchMethod=bnb.DFSMethod)
            algoBFS = bnb.BranchAndBound(prob, branchMethod=bnb.DFSMethod)
            algoBBS = bnb.BranchAndBound(prob, branchMethod=bnb.BBSMethod)
            for algo in algoDFS, algoBFS, algoBBS:
                sol = algo.run()
                checkDecoder.decode(llr)
                self.assert_(np.allclose(checkDecoder.solution, sol),
                             "{} ({}) != {} ({})".format(sol, algo.optimalObjectiveValue,
                                                         checkDecoder.objectiveValue, checkDecoder.solution))

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