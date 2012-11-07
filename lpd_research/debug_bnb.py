#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import itertools
import logging
import random
import sys

import numpy as np

from branchbound import branchrules, nodeselection, problem as bnbproblem, bnb
from lpdecoding.codes.interleaver import Interleaver
from lpdecoding.codes.turbolike import StandardTurboCode, LTETurboCode
from lpdecoding.codes.convolutional import LTEEncoder
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
from lpdecoding.utils import stopwatch
 
if __name__ == "__main__":
    size = int(sys.argv[1])
    logging.basicConfig(level=logging.INFO)
    random.seed(192837)
    interleaver = Interleaver.random(size)
    code = StandardTurboCode(LTEEncoder(), interleaver, "smallTestCode")
    checkDecoder = CplexTurboLikeDecoder(code, ip=True)
    
    nodeSelectionMethods = nodeselection.BFSMethod, nodeselection.DFSMethod, \
                           nodeselection.DSTMethod, nodeselection.BBSMethod
    branchingRules = branchrules.FirstFractional, branchrules.MostFractional, \
                     branchrules.LeastReliable
    #seed = np.random.randint(9999999)
    seed = 9864950
    #seed = 3977440
    np.random.seed(seed)
    llr = np.random.standard_normal(code.blocklength)
    print(llr)
    branchCounts = {}
    moveCounts = {}
    fixCounts = {}
    times = {}
    for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
        problem = bnbproblem.CplexTurboLPProblem(code)
        problem.setObjectiveFunction(llr)
        with stopwatch() as timer:
            algo = bnb.BranchAndBound(problem, eps=1e-10, branchRule=bRule, selectionMethod=nsMethod)
            solution = algo.run()
        times[(nsMethod.__name__, bRule.__name__)] = timer.duration
        checkDecoder.decode(llr)
        if np.allclose(checkDecoder.solution, solution):
            print("method {}/{} okay".format(nsMethod, bRule))
            print("\toptimal value={}".format(algo.optimalObjectiveValue))
        else:
            print("method {}/{} wrong solution:".format(nsMethod, bRule))
            print("\tBNB optimum={}".format(algo.optimalObjectiveValue))
            print("\tCPX optimum={}".format(checkDecoder.objectiveValue))
            print('\tseed= {}'.format(seed))
            raw_input()
        print("\tbranch count={}".format(algo.branchCount))
        print("\tmove count={}".format(algo.moveCount))
        print("\tfix count={}".format(algo.fixCount))
        print("\ttime={}".format(timer.duration))
        fixCounts[(nsMethod.__name__, bRule.__name__)] = algo.fixCount
        branchCounts[(nsMethod.__name__, bRule.__name__)] = algo.branchCount
        moveCounts[(nsMethod.__name__, bRule.__name__)] = algo.moveCount
    print("move counts: {}".format(moveCounts))
    print("fix counts: {}".format(fixCounts))
    print("branch counts: {}".format(branchCounts))
