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
                            nodeselection.DFSandBBSMethod, nodeselection.MyDFSMethod
    #nodeSelectionMethods = nodeselection.BBSMethod, nodeselection.MyBFSMethod,  \
    #nodeSelectionMethods = nodeselection.MyBFSRandom, nodeselection.MyBFSRound#, \
    #nodeSelectionMethods = nodeselection.MyDFSMethod, nodeselection.MyDFSRandom#, \
    #nodeSelectionMethods = nodeselection.MyDFSRound, nodeselection.MyDSTMethod#, \
#                           #nodeselection.DFSandBBSMethod
#    nodeSelectionMethods = [nodeselection.MyDFSMethod, nodeselection.DFSMethod]
                           
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
    attrs = "branchCount", "fixCount", "unfixCount", "moveCount", "time", "lpTime", "lpVsAll", \
            "boundTime", "boundVsAll", "refreshTime", "refreshVsAll", "getTime", "getVsAll", \
            "addTime", "addVsAll", "createTime", "createVsAll", "moveTime", "moveVsAll", \
            "selectionTime", "selectionVsAll"
    for attr in attrs:
        locals()[attr+"s"] = {}
        for nsMethod, bRule in itertools.product(nodeSelectionMethods, branchingRules):
            locals()[attr+"s"][(nsMethod.__name__, bRule.__name__)] = 0
    
    for i in range(numberOfTrials):
        llr = np.random.standard_normal(code.blocklength)
        timesXls.write(10*i+2,len(nodeSelectionMethods)*len(branchingRules)+2, "{}".format(seed) )
        with stopwatch() as timer:
            checkDecoder.decode(llr)
        cplexTime += timer.duration
        timesXls.write(10*i+2,len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(timer.duration))
        timesXls.write(10*i+3, len(nodeSelectionMethods)*len(branchingRules)+1, "lpTime")
        timesXls.write(10*i+4, len(nodeSelectionMethods)*len(branchingRules)+1, "boundTime")
        timesXls.write(10*i+5, len(nodeSelectionMethods)*len(branchingRules)+1, "refreshTime")
        timesXls.write(10*i+6, len(nodeSelectionMethods)*len(branchingRules)+1, "getTime")
        timesXls.write(10*i+7, len(nodeSelectionMethods)*len(branchingRules)+1, "addTime")
        timesXls.write(10*i+8, len(nodeSelectionMethods)*len(branchingRules)+1, "createTime")
        timesXls.write(10*i+9, len(nodeSelectionMethods)*len(branchingRules)+1, "moveTime")
        timesXls.write(10*i+10, len(nodeSelectionMethods)*len(branchingRules)+1, "selectionTime")
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
            timesXls.write(10*i+2, j, "{}".format(algo.time))
            timesXls.write(10*i+3, j, "{} ({})".format(algo.lpTime, round(algo.lpVsAll,2)))
            timesXls.write(10*i+4, j, "{} ({})".format(algo.boundTime, round(algo.boundVsAll, 2))) 
            timesXls.write(10*i+5, j, "{} ({})".format(algo.refreshTime, round(algo.refreshVsAll, 2)))
            timesXls.write(10*i+6, j, "{} ({})".format(algo.getTime, round(algo.getVsAll, 2)))
            timesXls.write(10*i+7, j, "{} ({})".format(algo.addTime, round(algo.addVsAll, 2)))
            timesXls.write(10*i+8, j, "{} ({})".format(algo.createTime, round(algo.createVsAll, 2)))
            timesXls.write(10*i+9, j, "{} ({})".format(algo.moveTime, round(algo.moveVsAll, 2)))
            timesXls.write(10*i+10, j, "{} ({})".format(algo.selectionTime, round(algo.selectionVsAll, 2)))
            
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
        timesXls.write(10*numberOfTrials + 11, i, "{}".format(times[argument]))
        timesXls.write(10*numberOfTrials + 12, i, "{} ({})".format(lpTimes[argument], round(lpVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 13, i, "{} ({})".format(refreshTimes[argument], round(refreshVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 14, i, "{} ({})".format(boundTimes[argument], round(boundVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 15, i, "{} ({})".format(getTimes[argument], round(getVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 16, i, "{} ({})".format(addTimes[argument], round(addVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 17, i, "{} ({})".format(createTimes[argument], round(createVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 18, i, "{} ({})".format(moveTimes[argument], round(moveVsAlls[argument], 2)))
        timesXls.write(10*numberOfTrials + 19, i, "{} ({})".format(selectionTimes[argument], round(selectionVsAlls[argument], 2)))
        
        
    timesXls.write(10*numberOfTrials + 12, len(nodeSelectionMethods)*len(branchingRules)+1, "lpTime")
    timesXls.write(10*numberOfTrials + 13, len(nodeSelectionMethods)*len(branchingRules)+1, "refreshTime")
    timesXls.write(10*numberOfTrials + 14, len(nodeSelectionMethods)*len(branchingRules)+1, "boundTime")
    timesXls.write(10*numberOfTrials + 15, len(nodeSelectionMethods)*len(branchingRules)+1, "getTime")
    timesXls.write(10*numberOfTrials + 16, len(nodeSelectionMethods)*len(branchingRules)+1, "addTime")
    timesXls.write(10*numberOfTrials + 17, len(nodeSelectionMethods)*len(branchingRules)+1, "createTime")
    timesXls.write(10*numberOfTrials + 18, len(nodeSelectionMethods)*len(branchingRules)+1, "moveTime")
    timesXls.write(10*numberOfTrials + 19, len(nodeSelectionMethods)*len(branchingRules)+1, "selectionTime")
    #cplexTime /= numberOfTrials
    #timesXls.write(2*numberOfTrials + 3, len(nodeSelectionMethods)*len(branchingRules)+1, "{}".format(cplexTime))
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
