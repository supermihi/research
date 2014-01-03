# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
import logging
import itertools
import random
from collections import OrderedDict

import numpy as np

from lpdecoding.core import Decoder
from lpdecoding.decoders.zhangsiegelc import ZhangSiegelACGC
from lpdecoding.decoders.zhangsiegelg import ZhangSiegelACGGLPK
from lpdecoding.decoders.iterative import IterativeDecoder
from lpdecoding.utils import StopWatch

logger = logging.getLogger(name="bb2")

def move(lbProv, ubProv, node, newNode):
    fix = []
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


class BranchAndBoundLDPCDecoder(Decoder):
    
    def __init__(self, code, branchMethod="mostFractional", selectionMethod="dfs", acgDepth=-1,
                 ubDepth=1,
                 childOrder="01",
                 glpk=False,
                 allZero=False,
                 minimumDistance=False,
                 name="BBDecoder", lpParams=None, iterParams=None):
        
        self.name = name
        if lpParams is None:
            lpParams = {}
        if iterParams is None:
            iterParams = dict(minSum=False, maxIterations=1000)
        if minimumDistance:
            lpParams["minimumDistance"] = True
        self.minDistance = minimumDistance
        DecoderClass = ZhangSiegelACGGLPK if glpk else ZhangSiegelACGC
        self.lbProvider = DecoderClass(code, **lpParams) 
        self.ubProvider = IterativeDecoder(code, **iterParams)
        self.glpk = glpk
        self.allZero = allZero
        self.branchMethod = branchMethod
        self.acgDepth = acgDepth
        self.ubDepth = ubDepth
        self.childOrder = childOrder
        self.selectionMethod = selectionMethod
        self.timer = StopWatch()
        Decoder.__init__(self, code)

    
    def setStats(self, stats):
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
    

    def stats(self):
        stats = self._stats.copy()
        stats["lpStats"] = self.lbProvider.stats().copy()
        stats["iterStats"] = self.ubProvider.stats().copy()
        return stats

    def branchIndex(self):
        if self.branchMethod == "mostFractional":
            index = np.argmin(np.abs(self.lbProvider.solution-0.5))
            if self.lbProvider.solution[index] < 1e-6 or self.lbProvider.solution[index] > 1-1e-6:
                return -1
            return index
        elif self.branchMethod == "leastReliable":
            for i in np.argsort(np.abs(self.llrs)):
                if np.abs(.5-self.lbProvider.solution[i]) < .499:
                    return i
            return -1
        elif self.branchMethod == "eiriksPaper":
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
            degrees, j = max(candidates)
            return j
        raise ValueError()
    
    def setLLRs(self, llrs):
        self.lbProvider.setLLRs(llrs)
        self.ubProvider.setLLRs(llrs)
        Decoder.setLLRs(self, llrs)
    
    def solve(self, hint=None, lb=1):
        from .node import Node
        for i in range(self.code.blocklength):
            self.lbProvider.release(i)
            self.ubProvider.release(i)
        self.foundCodeword = self.mlCertificate = True
        node = Node() # root node
        root = node
        activeNodes = []
        candidate = None
        ub = np.inf
        maxDepth = 0
        self._stats["nodes"] += 1
        for i in itertools.count(start=1):
            depthStr = str(node.depth)
            if i > 1 and i % 10 == 0:
                logger.info('gap {}/{}, d {}, n {}, c {}, iteration {}'.format(root.lb, ub, node.depth, len(activeNodes), self.lbProvider.numConstrs, i))
            #print('gap {}/{}, d {}, n {}, iteration {}'.format(root.lb, ub, node.depth, len(activeNodes), i))
            if depthStr not in self._stats["nodesPerDepth"]:
                self._stats["nodesPerDepth"][depthStr] = 0
            self._stats["nodesPerDepth"][depthStr] += 1
            if node.depth % self.ubDepth == 0:
                self.timer.start()
                self.ubProvider.solve()
                self._stats["heuristicTime"] += self.timer.stop()
                if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                    candidate = self.ubProvider.solution.copy()
                    ub = self.ubProvider.objectiveValue
                    if self.allZero and ub < 0:
                        self.mlCertificate = False
                        break
            if node.depth > maxDepth:
                maxDepth = node.depth
            if node.lb >= ub - 1e-6:
                # prune by bound
                logger.info("node pruned by bound 1")
                self._stats["prBd1"] += 1
                if node.parent is not None:
                    node.parent.updateBound(node.lb, node.branchValue)
                    if root.lb >= ub - 1e-6:
                        self._stats["termGap"] += 1
                        break
            else:
                # solve
                self.timer.start()
                if self.acgDepth != -1 and node.depth % self.acgDepth == 0:
                    self.lbProvider.pureLP = False
                    self.lbProvider.solve()
                    self.lbProvider.pureLP = True
                elif node.depth % self.ubDepth == 0:
                    if self.ubProvider.foundCodeword:
                        self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
                        self.lbProvider.solve(hint=self.ubProvider.solution.astype(np.int))
                    else:
                        self.lbProvider.hint = None
                        self.lbProvider.solve()
                else:
                    self.lbProvider.solve()
                self._stats["lpTime"] += self.timer.stop()
                node.lb = self.lbProvider.objectiveValue
                if node.parent is not None:
                    node.parent.updateBound(node.lb, node.branchValue)
                    if root.lb >= ub - 1e-6:
                        self._stats["termGap"] += 1
                        break
                if node.lb == np.inf:
                    logger.info("node pruned by infeasibility")
                    self._stats["prInf"] += 1
                elif self.lbProvider.foundCodeword:
                    # solution is integral
                    logger.info("node pruned by integrality")
                    if self.lbProvider.objectiveValue < ub:
                        candidate = self.lbProvider.solution.copy()
                        ub = self.lbProvider.objectiveValue
                        logger.info("ub improved to {}".format(ub))
                        self._stats["prOpt"] += 1
                        if self.allZero and ub < 0:
                            self.mlCertificate = False
                            break
                else:
                    if node.lb < ub-1e-6:
                        # branch
                        branchIndex = self.branchIndex()
                        newNodes = [Node(parent=node, branchIndex=branchIndex, branchValue=i) for i in (0,1) ]
                        if self.childOrder == "random":
                            random.shuffle(newNodes)
                        elif self.childOrder == "llr" and self.llrs[branchIndex] < 0:
                            newNodes.reverse()
                        activeNodes.extend(newNodes)
                        self._stats["nodes"] += 2
                    else:
                        logger.info("node pruned by bound 2")
                        self._stats["prBd2"] += 1
            if len(activeNodes) == 0:
                break
            if self.selectionMethod == "dfs":
                newNode = activeNodes.pop()
            elif self.selectionMethod == "bbs":
                newNode = min(activeNodes, key=lambda n: -n.lb)
                activeNodes.remove(newNode)
            elif self.selectionMethod == "bfs":
                newNode = activeNodes.pop(0)
            elif self.selectionMethod.startswith("mixed/"):
                if node.depth >= int(self.selectionMethod[6:]):
                    newNode = activeNodes.pop(0)
                else:
                    newNode = min(activeNodes, key=lambda n: -n.lb)
                    activeNodes.remove(newNode)
            else:
                raise ValueError("wrong selectionMethod")
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode
        self.lbProvider.reset()
        self.solution = candidate
        self.objectiveValue = ub
        
    def minimumDistance(self):
        assert self.minDistance
        self.setLLR(np.ones(self.code.blocklength, dtype=np.double))
        self.solve()
        return self.objectiveValue
        
    def params(self):
        parms = [("name", self.name),
                 ("branchMethod", self.branchMethod),
                 ("method", self.selectionMethod),
                 ("acgDepth", self.acgDepth),
                 ("ubDepth", self.ubDepth),
                 ("childOrder", self.childOrder),
                 ("lpParams", self.lbProvider.params()),
                 ("iterParams", self.ubProvider.params())]
        if self.glpk:
            parms.insert(2, ("glpk", True))
        if self.allZero:
            parms.insert(1, ("allZero", True))
        if self.minDistance:
            parms.insert(1, ("minimumDistance", True) )
        return OrderedDict(parms)

        
            
            