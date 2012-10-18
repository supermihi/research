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
import logging
import random
 
if __name__ == "__main__":
    size = int(sys.argv[1])
    logging.basicConfig(level=logging.INFO)
    random.seed(192837)
    interleaver = Interleaver.random(size)
    code = StandardTurboCode(LTEEncoder(), interleaver, "smallTestCode")
    checkDecoder = CplexTurboLikeDecoder(code, ip=True)
    problem = problem.CplexTurboLPProblem(code)
    methods = bnb.BFSMethod, bnb.DFSMethod, bnb.DSTMethod, bnb.BBSMethod
    seed = np.random.randint(9999999)
    seed = 3812070
    np.random.seed(seed)
    
    llr = np.random.standard_normal(code.blocklength)
    print(llr)
    branchCounts = {}
    moveCounts = {}
    fixCounts = {}
    for method in methods:
        problem.setObjectiveFunction(llr)
        algo = bnb.BranchAndBound(problem, eps=1e-10, branchMethod=method)
        solution = algo.run()
        checkDecoder.decode(llr)
        if np.allclose(checkDecoder.solution, solution):
            print("method {} okay".format(method))
            print("\toptimal value={}".format(algo.optimalObjectiveValue))
        else:
            print("method {} wrong solution:".format(method))
            print("\tBNB optimum={}".format(algo.optimalObjectiveValue))
            print("\tCPX optimum={}".format(checkDecoder.objectiveValue))
            print('\tseed= {}'.format(seed))
            raw_input()
        print("\tbranch count={}".format(algo.branchCount))
        print("\tmove count={}".format(algo.moveCount))
        print("\tfix count={}".format(algo.fixCount))
        fixCounts[method.__name__] = algo.fixCount
        branchCounts[method.__name__] = algo.branchCount
        moveCounts[method.__name__] = algo.moveCount
    print("move counts: {}".format(moveCounts))
    print("fix counts: {}".format(fixCounts))
    print("branch counts: {}".format(branchCounts))
