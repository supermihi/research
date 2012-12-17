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
    

    nodeSelectionMethods = nodeselection.BBSMethod, nodeselection.BFSMethod,  \
                           nodeselection.BFSRandom, nodeselection.BFSRound, \
                           nodeselection.DFSMethod, nodeselection.DFSRandom, \
                           nodeselection.DFSRound, nodeselection.DSTMethod, \
                           nodeselection.DFSandBBSMethod
                           
    branchingRules = branchrules.LeastReliable, branchrules.LeastReliableSystematic, \
                     branchrules.FirstFractional, \
                     branchrules.MostFractional, branchrules.MostFractionalSystematic
    #initialize xls
    stats = Workbook()
    branchCountsXls = stats.add_sheet('BranchCounts')
    fixCountsXls = stats.add_sheet('FixCounts')
    unfixCountsXls = stats.add_sheet('UnfixCounts')
    moveCountsXls = stats.add_sheet('moveCounts')
    timesXls = stats.add_sheet('Times')
    #sheets = [branchCounts, fixCounts, unfixCounts, times]
    
    for (i,nsMethod) in enumerate(nodeSelectionMethods):
#        for sheet in sheets:
#            sheet.write(0,i*len(branchingRules), nsMethod.__name__)
        branchCountsXls.write(0,i*len(branchingRules), nsMethod.__name__)
        fixCountsXls.write(0,i*len(branchingRules), nsMethod.__name__)
        unfixCountsXls.write(0,i*len(branchingRules), nsMethod.__name__)
        moveCountsXls.write(0,i*len(branchingRules), nsMethod.__name__)
        timesXls.write(0,i*len(branchingRules), nsMethod.__name__)
        for (j,bRule) in enumerate(branchingRules):
#            for sheet in sheets:
#                sheet.write(0,i*len(branchingRules), nsMethod.__name__)
            branchCountsXls.write(1,i*len(branchingRules)+j, bRule.__name__)
            fixCountsXls.write(1,i*len(branchingRules)+j, bRule.__name__)
            unfixCountsXls.write(1,i*len(branchingRules)+j, bRule.__name__)
            moveCountsXls.write(1,i*len(branchingRules)+j, bRule.__name__)
            timesXls.write(1,i*len(branchingRules)+j, bRule.__name__)
    timesXls.write(0, len(nodeSelectionMethods)*len(branchingRules)+1, 'Cplex')
    timesXls.write(0, len(nodeSelectionMethods)*len(branchingRules)+2, 'Seed')
    #stats.save('stats.xls') 
                       
    seed = np.random.randint(9999999)
    #seed = 9864950
    #seed = 3977440
    np.random.seed(seed)
    attrs = "branchCount", "fixCount", "unfixCount", "moveCount", "time", "lpTime", "lpVsAll" 
    for attr in attrs:
        locals()[attr+"s"] = {}
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] = 0
    
    for i in range(numberOfTrials):
        llr = np.random.standard_normal(code.blocklength)
        timesXls.write(2*i+2,len(nodeSelectionMethods)*len(branchingRules)+2, "{}".format(seed) )
        with stopwatch() as timer:
            checkDecoder.decode(llr)
        cplexTime += timer.duration
        timesXls.write(2*i+2,len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(timer.duration))
        for (j, (nsMethod, bRule)) in enumerate(itertools.product(nodeSelectionMethods, branchingRules)):
            #llr = np.random.standard_normal(code.blocklength)
            #problem = bnbproblem.CplexTurboLPProblem(code)
            problem = bnbproblem.CSPTurboLPProblem(code)
            problem.setObjectiveFunction(llr)
            algo = bnb.BranchAndBound(problem, eps=1e-10, branchRule=bRule, selectionMethod=nsMethod)
            solution = algo.run()
            branchCountsXls.write(i+2, j, "{}".format(algo.branchCount))
            fixCountsXls.write(i+2, j, "{}".format(algo.fixCount))
            unfixCountsXls.write(i+2, j, "{}".format(algo.unfixCount))
            moveCountsXls.write(i+2, j, "{}".format(algo.moveCount))
            timesXls.write(2*i+2, j, "{}".format(algo.time))
            timesXls.write(2*i+3, j, "{} ({})".format(algo.lpTime, round(algo.lpVsAll,2)))
            for attr in attrs:
                locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] += getattr(algo, attr)
            if np.allclose(checkDecoder.solution, solution):
                print("method {}/{} okay".format(nsMethod.__name__, bRule.__name__))
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
            print("\tunfix count={}".format(algo.unfixCount))
    for attr in attrs:
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] /= numberOfTrials
    j = numberOfTrials + 3
    for (i,(nsMethod, bRule)) in enumerate(itertools.product(nodeSelectionMethods, branchingRules)):
        #argument = '("{}", "{}")'.format(nsMethod.__name__, bRule.__name__)
        argument = ("{}".format(nsMethod.__name__), "{}".format(bRule.__name__))
        branchCountsXls.write(j,i, "{}".format(branchCounts[argument]))
        fixCountsXls.write(j,i, "{}".format(fixCounts[argument]))
        unfixCountsXls.write(j,i, "{}".format(unfixCounts[argument]))
        moveCountsXls.write(j, i, "{}".format(moveCounts[argument]))
        timesXls.write(2*numberOfTrials + 3, i, "{}".format(times[argument]))
        timesXls.write(2*numberOfTrials + 4, i, "{} ({})".format(lpTimes[argument], round(lpVsAlls[argument], 2)))
    cplexTime /= numberOfTrials
    timesXls.write(2*numberOfTrials + 3, len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(cplexTime))
    import pprint
    print("move counts:")
    pprint.pprint(moveCounts)
    print("fix counts:")
    pprint.pprint(fixCounts)
    print("branch counts:")
    pprint.pprint(branchCounts)
    print("times:")
    pprint.pprint(times)
    print("cplexTime: {}".format(cplexTime))
    print("lpVsAlls:")
    pprint.pprint(lpVsAlls)
    stats.save('stats.xls')
