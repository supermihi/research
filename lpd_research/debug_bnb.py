#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import sys

import numpy as np

from branchbound import problem, bnb
from lpdecoding.codes.interleaver import Interleaver
from lpdecoding.codes.turbolike import StandardTurboCode, LTETurboCode
from lpdecoding.codes.convolutional import LTEEncoder
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder

from lpdecoding.channels import SignalGenerator, AWGNC
 
if __name__ == "__main__":
    size = int(sys.argv[1])
    interleaver = Interleaver.random(size)
    code = StandardTurboCode(LTEEncoder(), interleaver, "smallTestCode")
    checkDecoder = CplexTurboLikeDecoder(code, ip=True)
    problem = problem.CplexTurboLPProblem(code)
    methods = bnb.BFSMethod, bnb.DFSMethod, bnb.DSTMethod, bnb.BBSMethod
    np.random.seed(223948)
    llr = np.random.standard_normal(code.blocklength)
    branchCounts = {}
    moveCounts = {}
    fixCounts = {}
    for method in methods:
        problem.setObjectiveFunction(llr)
        algo = bnb.BranchAndBound(problem, eps=1e-10, branchMethod=method)
        solution = algo.run()
        checkDecoder.decode(llr)
        if np.allclose(checkDecoder.solution, solution):
            print("okay")
        else:
            print("wrong solution:")
            print("\tBNB optimum={}".format(algo.optimalObjectiveValue))
            print("\tCPX optimum={}".format(checkDecoder.objectiveValue))
            raw_input()
        print("branch count={}".format(algo.branchCount))
        print("move count={}".format(algo.moveCount))
        print("fix count={}".format(algo.fixCount))
        fixCounts[method.__name__] = algo.fixCount
        branchCounts[method.__name__] = algo.branchCount
        moveCounts[method.__name__] = algo.moveCount
    print("move counts: {}".format(moveCounts))
    print("fix counts: {}".format(fixCounts))
    print("branch counts: {}".format(branchCounts))
