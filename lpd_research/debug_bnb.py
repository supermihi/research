#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, division
import itertools
import logging
import random
import sys

import numpy as np

from branchbound import branchrules, nodeselection, problem as bnbproblem, bnb
from lpdecoding.codes.interleaver import Interleaver
from lpdecoding.codes.turbolike import StandardTurboCode
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
#                           nodeselection.DFSandBBSMethod
    branchingRules = branchrules.LeastReliable, branchrules.LeastReliableSystematic, \
                     branchrules.FirstFractional, \
                     branchrules.MostFractional, branchrules.MostFractionalSystematic
                     
    #seed = np.random.randint(9999999)
    seed = 9864950
    numberOfTrials = 10
    #seed = 3977440
    np.random.seed(seed)
    attrs = "fixCount", "moveCount", "branchCount", "time"
    for attr in attrs:
        locals()[attr+"s"] = {}
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] = 0
        
    for i in range(numberOfTrials):
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            llr = np.random.standard_normal(code.blocklength)
            #problem = bnbproblem.CplexTurboLPProblem(code)
            problem = bnbproblem.CSPTurboLPProblem(code)
            problem.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(problem, eps=1e-10, branchRule=bRule, selectionMethod=nsMethod)
            solution = algo.run()
            for attr in attrs:
                locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] += getattr(algo, attr)
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
    for attr in attrs:
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] /= numberOfTrials
    import pprint
    print("move counts:")
    pprint.pprint(moveCounts)
    print("fix counts:")
    pprint.pprint(fixCounts)
    print("branch counts:")
    pprint.pprint(branchCounts)
    print("times:")
    pprint.pprint(times)
