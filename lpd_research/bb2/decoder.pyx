# -*- coding: utf-8 -*-
# Copyright 2013-2014 Michael Helmling
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function

import logging
import itertools
import random
from collections import OrderedDict

from libc.math cimport fabs
 
import numpy as np
cimport numpy as np

from lpdecoding.core cimport Decoder
from lpdecoding.core import Decoder
from lpdecoding.decoders.zhangsiegelc import ZhangSiegelACGC
from lpdecoding.decoders.zhangsiegelg import ZhangSiegelACGGLPK
from lpdecoding.decoders.iterative import IterativeDecoder
from lpdecoding.utils cimport StopWatch

from bb2.node cimport Node

logger = logging.getLogger(name="bb2")

cdef void move(Decoder lbProv, Decoder ubProv, Node node, Node newNode):
    cdef list fix = []
    while node.depth > newNode.depth:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        node = node.parent
    
    while newNode.depth > node.depth:
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        newNode = newNode.parent
    while node is not newNode:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        node = node.parent
        newNode = newNode.parent
    for var, value in fix:
        lbProv.fix(var, value)
        ubProv.fix(var, value)


cdef enum BranchMethod:
    mostFractional, leastReliable, eiriksPaper
    

cdef enum SelectionMethod:
    mixed2, dfs, bbs, bfs


cdef class BranchAndBoundLDPCDecoder(Decoder):

    cdef:
        bint glpk, minDistance, allZero, calcUb, ubBB, highSNR
        object childOrder
        SelectionMethod selectionMethod
        BranchMethod branchMethod
        Decoder lbProvider, ubProvider
        int mixParam, maxRPCspecial, maxRPCnormal, maxRPCorig
        double mixGap
        StopWatch timer
        int selectCnt
        Node root
    
    def __init__(self, code, branchMethod="mostFractional", selectionMethod="dfs",
                 childOrder="01",
                 glpk=False,
                 allZero=False,
                 minimumDistance=False,
                 highSNR=False,
                 name="BBDecoder", lpParams=None, iterParams=None):
        
        self.name = name
        if lpParams is None:
            lpParams = {}
        if iterParams is None:
            iterParams = dict(minSum=False, maxIterations=1000)
        if minimumDistance:
            #lpParams["minimumDistance"] = True
            iterParams["minimumDistance"] = True
        self.minDistance = minimumDistance
        DecoderClass = ZhangSiegelACGGLPK if glpk else ZhangSiegelACGC
        self.lbProvider = DecoderClass(code, **lpParams) 
        self.ubProvider = IterativeDecoder(code, **iterParams)
        self.glpk = glpk
        self.highSNR = highSNR
        self.allZero = allZero
        if branchMethod == "mostFractional":
            self.branchMethod = mostFractional
        elif branchMethod == "leastReliable":
            self.branchMethod = leastReliable
        else:
            assert branchMethod == "eiriksPaper"
            self.branchMethod = eiriksPaper
        self.childOrder = childOrder
        self.calcUb = True
        if selectionMethod.startswith("mixed2"):
            self.selectionMethod = mixed2
            if selectionMethod[6] == "U":
                self.ubBB = True
                selectionMethod = selectionMethod[8:]
            else:
                self.ubBB = False
                selectionMethod = selectionMethod[7:]
            a, b, c, d = selectionMethod.split("/")
            self.mixParam = int(a)
            self.maxRPCspecial = int(b)
            self.maxRPCnormal = int(c)
            self.mixGap = float(d)
            self.maxRPCorig = self.lbProvider.maxRPCrounds
        elif selectionMethod == "bfs":
            self.selectionMethod = "bfs"
        elif selectionMethod == "dfs":
            self.selectionMethod = "dfs"
        else:
            assert selectionMethod == "bbs"
            self.selectionMethod = bbs
        self.timer = StopWatch()
        Decoder.__init__(self, code)

    
    cpdef setStats(self, stats):
        for item in "lpTime", "heuristicTime", "nodes", "prBd1", "prBd2", "prInf", "prOpt", "termEx", "termGap":
            if item not in stats:
                stats[item] = 0
        if "nodesPerDepth" not in stats:
            stats["nodesPerDepth"] = {}
        if "lpStats" in stats:
            self.lbProvider.setStats(stats["lpStats"])
            del stats["lpStats"]
        else:
            self.lbProvider.setStats(dict())
        if "iterStats" in stats:
            self.ubProvider.setStats(stats["iterStats"])
            del stats["iterStats"]
        else:
            self.ubProvider.setStats(dict())
        Decoder.setStats(self, stats)
    

    cpdef stats(self):
        stats = self._stats.copy()
        stats["lpStats"] = self.lbProvider.stats().copy()
        stats["iterStats"] = self.ubProvider.stats().copy()
        return stats


    cdef int branchIndex(self):
        cdef:
            int index, i
            double minDiff = np.inf
            np.double_t[:] solution = self.lbProvider.solution
        if self.branchMethod == mostFractional:
            for i in range(solution.shape[0]):
                if fabs(solution[i] - .5) < minDiff:
                    index = i
                    minDiff = fabs(solution[i] - .5)
            if solution[index] < 1e-6 or solution[index] > 1-1e-6:
                for index in range(self.code.blocklength):
                    if not self.fixed(index):
                        return index
                return -1
            return index
        elif self.branchMethod == leastReliable:
            for i in np.argsort(np.abs(self.llrs)):
                if fabs(.5-solution[i]) < .499:
                    return i
            return -1
        elif self.branchMethod == eiriksPaper:
            matrix = self.code.parityCheckMatrix
            degrees = np.zeros(matrix.shape[0], dtype=np.int)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i,j] == 1 and not self.lbProvider.fixed(j):
                        degrees[i] += 1
            candidates = []
            for j in range(matrix.shape[1]):
                if self.lbProvider.solution[j] > 1e-6 and self.lbProvider.solution[j] < 1-1e-6:
                    cdegrees = [0] * max(degrees)
                    for i in np.flatnonzero(matrix[:, j]):
                        if degrees[i] > 0:
                            cdegrees[degrees[i]-1] += 1
                    candidates.append( (cdegrees, j) )
            if len(candidates) == 0:
                return -1
            degrees, j = max(candidates)
            return j
    
    
    cpdef setLLRs(self, np.double_t[:] llrs):
        self.ubProvider.setLLRs(llrs)
        if self.highSNR:
            self.ubProvider.foundCodeword = self.ubProvider.mlCertificate = False
        else:
            self.timer.start()
            self.ubProvider.solve()
            self._stats["heuristicTime"] += self.timer.stop()
            if self.ubProvider.foundCodeword:
                self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
        self.lbProvider.setLLRs(llrs)
        Decoder.setLLRs(self, llrs)
    
    
    cpdef fix(self, int index, int value):
        self.lbProvider.fix(index, value)
        self.ubProvider.fix(index,value)
        
        
    cpdef release(self, int index):
        self.lbProvider.release(index)
        self.ubProvider.release(index)
        
        
    cpdef fixed(self, int index):
        return self.lbProvider.fixed(index)


    cpdef solve(self, np.int_t[:] hint=None, double lb=1):
        cdef:
            Node node, root, newNode0, newNode1, newNode
            list activeNodes = []
            np.ndarray[dtype=np.double_t, ndim=1] candidate = np.zeros(self.code.blocklength, dtype=np.double)
            double ub = 0
            int i, branchIndex
            str depthStr
        for i in range(self.code.blocklength):
            self.lbProvider.release(i)
            self.ubProvider.release(i)
        self.foundCodeword = self.mlCertificate = True
        root = node = Node() # root node
        self.selectCnt = 0
        self._stats["nodes"] += 1
        if self.selectionMethod == mixed2:
            self.lbProvider.maxRPCrounds = self.maxRPCspecial
        for i in itertools.count(start=1):
            
            # statistic collection and debug output
            depthStr = str(node.depth)
            if i > 1 and i % 10 == 0:
                logger.info('{}/{}, d {}, n {}, c {}, it {}, lp {}, spa {}'.format(root.lb, ub, node.depth, len(activeNodes), self.lbProvider.numConstrs, i, self._stats["lpTime"], self._stats["heuristicTime"]))
            if depthStr not in self._stats["nodesPerDepth"]:
                self._stats["nodesPerDepth"][depthStr] = 0
            self._stats["nodesPerDepth"][depthStr] += 1
            
            # upper bound calculation
            if i > 1 and self.calcUb: # for first iteration this was done in setLLR
                self.timer.start()
                self.ubProvider.solve()
                self._stats["heuristicTime"] += self.timer.stop()
            if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                candidate = self.ubProvider.solution.copy()
                ub = self.ubProvider.objectiveValue
                if self.allZero and ub < 0:
                    self.mlCertificate = False
                    break
            
            # lower bound calculation
            self.timer.start()
            self.lbProvider.upperBound = ub
            if i == 1 or self.calcUb:
                if self.ubProvider.foundCodeword:
                    self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
                    self.lbProvider.solve(hint=self.lbProvider.hint)
                else:
                    self.lbProvider.hint = None
                    self.lbProvider.solve()
            else:
                self.lbProvider.solve()
            self._stats["lpTime"] += self.timer.stop()
            if self.lbProvider.objectiveValue > node.lb:
                node.lb = self.lbProvider.objectiveValue
        
            # pruning or branching
            if node.lb == np.inf:
                logger.debug("node pruned by infeasibility")
                self._stats["prInf"] += 1
            elif self.lbProvider.foundCodeword:
                # solution is integral
                logger.debug("node pruned by integrality")
                if self.lbProvider.objectiveValue < ub:
                    candidate = self.lbProvider.solution.copy()
                    ub = self.lbProvider.objectiveValue
                    logger.debug("ub improved to {}".format(ub))
                    self._stats["prOpt"] += 1
                    if self.allZero and ub < 0:
                        self.mlCertificate = False
                        break
            elif node.lb < ub-1e-6:
                # branch
                branchIndex = self.branchIndex()
                newNode0 = Node(parent=node, branchIndex=branchIndex, branchValue=0)
                newNode1 = Node(parent=node, branchIndex=branchIndex, branchValue=1)
                if    (self.childOrder == "random" and np.random.randint(0, 2) == 0) \
                   or (self.childOrder == "llr" and self.llrs[branchIndex] < 0) \
                   or (self.childOrder == "10"):
                    activeNodes.append(newNode1)
                    activeNodes.append(newNode0)
                else:
                    activeNodes.append(newNode0)
                    activeNodes.append(newNode1)
                self._stats["nodes"] += 2
            else:
                logger.debug("node pruned by bound 2")
                self._stats["prBd2"] += 1
            if node.parent is not None:
                node.parent.updateBound(node.lb, node.branchValue)
                if root.lb >= ub - 1e-6:
                    self._stats["termGap"] += 1
                    break
            if len(activeNodes) == 0:
                self._stats["termEx"] += 1
                break
            newNode = self.selectNode(activeNodes, node, ub)
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode
        if self.selectionMethod == mixed2:
            self.lbProvider.maxRPCrounds = self.maxRPCorig
        self.solution = candidate
        self.objectiveValue = ub
        self.lbProvider.upperBound = np.inf
    
        
    cdef Node popMinNode(self, list activeNodes):
        cdef int i, minIndex
        cdef double minValue = np.inf
        for i in range(len(activeNodes)):
            if activeNodes[i].lb < minValue:
                minIndex = i
                minValue = activeNodes[i].lb
        return activeNodes.pop(minIndex)
    
    
    cdef Node selectNode(self, list activeNodes, Node currentNode, double ub):
        if self.selectionMethod == mixed2:
            if ((self.selectCnt >= self.mixParam) or (self.minDistance and self.root.lb == 1)) and (ub - currentNode.lb) > self.mixGap:
                # best bound
                newNode = self.popMinNode(activeNodes)
                self.selectCnt = 1
                self.lbProvider.maxRPCrounds = self.maxRPCspecial #np.rint(ub-newNode.lb)
                if self.ubBB:
                    self.calcUb = True
                return newNode
            else:
                self.lbProvider.maxRPCrounds = self.maxRPCnormal
                self.selectCnt += 1
                if self.ubBB:
                    self.calcUb = False
                return activeNodes.pop()
        elif self.selectionMethod == dfs:
            return activeNodes.pop()
        elif self.selectionMethod == bbs:
            return self.popMinNode(activeNodes)
        elif self.selectionMethod == bfs:
            return activeNodes.pop(0)
    
    
    cpdef minimumDistance(self):
        assert self.minDistance
        llrs = np.ones(self.code.blocklength, dtype=np.double)
        randomizedMD = True
        
        if randomizedMD:
            delta = 0.001
            epsilon = delta/self.code.blocklength
            np.random.seed(239847)
            llrs += epsilon*np.random.random_sample(self.code.blocklength)
        else:
            delta = 1e-6
        self.setLLRs(llrs)
        self.selectCnt = 1
        from .node import Node
        root = self.root = node = Node()
        root.lb = 1
        activeNodes = []
        candidate = None
        ub = np.inf
        self._stats["nodes"] += 1
        for i in itertools.count(start=1):            
            # statistic collection and debug output
            depthStr = str(node.depth)
            if i % 1000 == 0:
                logger.info('MD {}/{}, d {}, n {}, c {}, it {}, lp {}, spa {}'.format(root.lb, ub, node.depth, len(activeNodes), self.lbProvider.numConstrs, i, self._stats["lpTime"], self._stats["heuristicTime"]))
            if depthStr not in self._stats["nodesPerDepth"]:
                self._stats["nodesPerDepth"][depthStr] = 0
            self._stats["nodesPerDepth"][depthStr] += 1
            #node.printFixes()
            if node.lb >= ub-1+delta:
                logger.debug('prune 1')
            # upper bound calculation
            if i > 1 and self.calcUb: # for first iteration this was done in setLLR
                self.timer.start()
                self.ubProvider.solve()
                self._stats["heuristicTime"] += self.timer.stop()
            if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                candidate = self.ubProvider.solution.copy()
                ub = self.ubProvider.objectiveValue
            
            # lower bound calculation
            self.lbProvider.upperBound = ub - 1 + delta
            self.timer.start()
            if (i == 1 or self.calcUb) and self.ubProvider.foundCodeword:
                self.lbProvider.solve(hint=self.ubProvider.solution.astype(np.int))
            else:
                self.lbProvider.solve()
            self._stats["lpTime"] += self.timer.stop()
            if self.lbProvider.objectiveValue > node.lb:
#                 fractVal = self.lbProvider.objectiveValue - np.trunc(self.lbProvider.objectiveValue)
#                 if fractVal > delta and fractVal < 1-1e-6:
#                     node.lb = np.ceil(self.lbProvider.objectiveValue)
#                 else:
                node.lb = self.lbProvider.objectiveValue
            if node.lb == np.inf:
                logger.debug("node pruned by infeasibility")
                self._stats["prInf"] += 1
            elif self.lbProvider.foundCodeword and self.lbProvider.objectiveValue > .5:
                # solution is integral
                logger.debug("node pruned by integrality")
                if self.lbProvider.objectiveValue < ub:
                    candidate = self.lbProvider.solution.copy()
                    print('cand lb')
                    print(candidate)
                    ub = self.lbProvider.objectiveValue
                    logger.debug("ub improved to {}".format(ub))
                    self._stats["prOpt"] += 1
            elif node.lb < ub-1+delta:
                # branch
                branchIndex = self.branchIndex()
                if branchIndex == -1:
                    node.lb = np.inf
                    print('********** PRUNE 000000 ***************')
                    #raw_input()
                else:
                    newNodes = [Node(parent=node, branchIndex=branchIndex, branchValue=i) for i in (0,1) ]
                    if self.childOrder == "random":
                        random.shuffle(newNodes)
                    elif (self.childOrder == "llr" and self.llrs[branchIndex] < 0) or self.childOrder == "10":
                        newNodes.reverse()
                    activeNodes.extend(newNodes)
                    self._stats["nodes"] += 2
            else:
                logger.debug("node pruned by bound 2")
                self._stats["prBd2"] += 1
            if node.parent is not None:
                node.parent.updateBound(node.lb, node.branchValue)
                if root.lb >= ub - 1 + delta:
                    self._stats["termGap"] += 1
                    break
            if len(activeNodes) == 0:
                self._stats["termEx"] += 1
                break
            newNode = self.selectNode(activeNodes, node, ub)
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode
        self.solution = candidate
        self.objectiveValue = np.rint(ub)
        return self.objectiveValue
        
    cpdef params(self):
        if self.selectionMethod == mixed2:
            method = "mixed2{}/{}/{}/{}/{}".format("U" if self.ubBB else "",
                                                   self.mixParam,
                                                   self.maxRPCspecial,
                                                   self.maxRPCnormal,
                                                   self.mixGap)
        else:
            methodNames = { dfs: "dfs", bfs: "bfs", bbs: "bbs"}
            method = methodNames[self.selectionMethod]
        branchMethodNames = {mostFractional: "mostFractional", leastReliable: "leastReliable", eiriksPaper: "eiriksPaper"}
        parms = [("name", self.name),
                 ("branchMethod", branchMethodNames[self.branchMethod]),
                 ("selectionMethod", method),
                 ("childOrder", self.childOrder),
                 ("lpParams", self.lbProvider.params()),
                 ("iterParams", self.ubProvider.params())]
        if self.glpk:
            parms.insert(2, ("glpk", True))
        if self.highSNR:
            parms.insert(2, ("highSNR", True) )
        if self.allZero:
            parms.insert(1, ("allZero", True))
        if self.minDistance:
            parms.insert(1, ("minimumDistance", True) )
        return OrderedDict(parms)

        
            
            
