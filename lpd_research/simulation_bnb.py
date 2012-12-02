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
from xlwt import Workbook
 
if __name__ == "__main__":

    size = int(sys.argv[1])
    numberOfTrials = int(sys.argv[2])
    logging.basicConfig(level=logging.INFO)
    random.seed(192837)
    cplexTime= 0
    interleaver = Interleaver.random(size)
    code = StandardTurboCode(LTEEncoder(), interleaver, "smallTestCode")
    checkDecoder = CplexTurboLikeDecoder(code, ip=True)
    

    nodeSelectionMethods = nodeselection.BFSMethod, nodeselection.DFSMethod, \
                           nodeselection.DSTMethod, nodeselection.BBSMethod, \
                           nodeselection.DFSandBBSMethod
    branchingRules = branchrules.LeastReliable, branchrules.LeastReliableSystematic, \
                     branchrules.FirstFractional, \
                     branchrules.MostFractional, branchrules.MostFractionalSystematic
    #initialize xls
    stats = Workbook()
    branchCounts = stats.add_sheet('BranchCounts')
    fixCounts = stats.add_sheet('FixCounts')
    unfixCounts = stats.add_sheet('UnfixCounts')
    moveCounts = stats.add_sheet('moveCounts')
    times = stats.add_sheet('Times')
    sheets = [branchCounts, fixCounts, unfixCounts, times]
    
    for (i,nsMethod) in enumerate(nodeSelectionMethods):
#        for sheet in sheets:
#            sheet.write(0,i*len(branchingRules), nsMethod.__name__)
        branchCounts.write(0,i*len(branchingRules), nsMethod.__name__)
        fixCounts.write(0,i*len(branchingRules), nsMethod.__name__)
        unfixCounts.write(0,i*len(branchingRules), nsMethod.__name__)
        moveCounts.write(0,i*len(branchingRules), nsMethod.__name__)
        times.write(0,i*len(branchingRules), nsMethod.__name__)
        for (j,bRule) in enumerate(branchingRules):
#            for sheet in sheets:
#                sheet.write(0,i*len(branchingRules), nsMethod.__name__)
            branchCounts.write(1,i*len(branchingRules)+j, bRule.__name__)
            fixCounts.write(1,i*len(branchingRules)+j, bRule.__name__)
            unfixCounts.write(1,i*len(branchingRules)+j, bRule.__name__)
            moveCounts.write(1,i*len(branchingRules)+j, bRule.__name__)
            times.write(1,i*len(branchingRules)+j, bRule.__name__)
    times.write(0, len(nodeSelectionMethods)*len(branchingRules)+1, 'Cplex')
    #stats.save('stats.xls') 
                       
    seed = np.random.randint(9999999)
    #seed = 9864950
    #seed = 3977440
    np.random.seed(seed)
    attrs = "branchCount", "fixCount", "unfixCount", "moveCount", "time", "lpVsAll" 
    for attr in attrs:
        locals()[attr+"s"] = {}
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] = 0
    
    for i in range(numberOfTrials):
        llr = np.random.standard_normal(code.blocklength)
        with stopwatch() as timer:
            checkDecoder.decode(llr)
        cplexTime += timer.duration
        stats.get_sheet(4).write(i+2,len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(timer.duration))
        for (j, (nsMethod, bRule)) in enumerate(itertools.product(nodeSelectionMethods, branchingRules)):
            #llr = np.random.standard_normal(code.blocklength)
            #problem = bnbproblem.CplexTurboLPProblem(code)
            problem = bnbproblem.CSPTurboLPProblem(code)
            problem.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(problem, eps=1e-10, branchRule=bRule, selectionMethod=nsMethod)
            solution = algo.run()
            stats.get_sheet(0).write(i+2, j, "{}".format(algo.branchCount))
            stats.get_sheet(1).write(i+2, j, "{}".format(algo.fixCount))
            stats.get_sheet(2).write(i+2, j, "{}".format(algo.unfixCount))
            stats.get_sheet(3).write(i+2, j, "{}".format(algo.moveCount))
            stats.get_sheet(4).write(i+2, j, "{}".format(algo.time))
            for attr in attrs:
                locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] += getattr(algo, attr)
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
    j = numberOfTrials + 3
    for (i,(nsMethod, bRule)) in enumerate(itertools.product(nodeSelectionMethods, branchingRules)):
        stats.get_sheet(0).write(j,i, "{}".format(branchCounts))
        stats.get_sheet(1).write(j,i, "{}".format(fixCounts))
        stats.get_sheet(2).write(j,i, "{}".format(unfixCounts))
        stats.get_sheet(3).write(j, i, "{}".format(moveCounts))
        stats.get_sheet(4).write(j, i, "{}".format(times))
    cplexTime /= numberOfTrials
    stats.get_sheet(4).write(j, len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(cplexTime))
    import pprint
    print("move counts:")
    pprint.pprint(moveCounts)
    print("fix counts:")
    pprint.pprint(fixCounts)
    print("branch counts:")
    pprint.pprint(branchCounts)
    print("times:")
    pprint.pprint(times)
    print("lpVsAlls:")
    pprint.pprint(lpVsAlls)
    print("cplexTime: {}".format(cplexTime))
    stats.save('stats.xls')
