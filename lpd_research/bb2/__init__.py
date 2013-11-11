# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import logging
import itertools

import numpy as np

from lpdecoding.core import Decoder

logger = logging.getLogger(name="bb2")

class Node:
    nodeCount = 0
    
    def __init__(self, **kwargs):
        self.parent = kwargs.get("parent", None)
        self.branchIndex = kwargs.get("branchIndex", None)
        self.branchValue = kwargs.get("branchValue", None)
        self.lb = -np.inf
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.lbChild = [-np.inf, -np.inf]
        Node.nodeCount += 1
        
    def updateBound(self, lbChild, childValue):
        if lbChild > self.lbChild[childValue]:
            self.lbChild[childValue] = lbChild
        newLb = min(self.lbChild)
        if newLb > self.lb:
            self.lb = newLb
            if self.parent is not None:
                self.parent.updateBound(newLb, self.branchValue)

    def __del__(self):
        Node.nodeCount -= 1


def move(problem, node, newNode):
    fix = []
    while node.depth > newNode.depth:
        problem.unfixVariable(node.branchIndex)
        node = node.parent
    
    while newNode.depth > node.depth:
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        newNode = newNode.parent
        
    while node is not newNode:
        problem.unfixVariable(node.branchIndex)
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        node = node.parent
        newNode = newNode.parent
    for var, value in fix:
        problem.fixVariable(var, value)

class BranchAndBoundLDPCDecoder(Decoder):
    
    def __init__(self, code, branchMethod="mostFractional", pureLP=False, method="dfs", name="BBDecoder"):
        self.code = code
        self.name = name
        import bbldpc
        self.problem = bbldpc.LDPCLPProblem(code, pureLP)
        self.branchMethod = branchMethod
        self.method = method

    def branchIndex(self):
        if self.branchMethod == "mostFractional":
            index = np.argmin(np.abs(self.problem.solution-0.5))
            if self.problem.solution[index] < 1e-10 or self.problem.solution[index] > 1-1e-10:
                return -1
            return index
        elif self.branchMethod == "leastReliable":
            for i in np.argsort(np.abs(self.llrVector)):
                if np.abs(.5-self.problem.solution[i]) < .499:
                    return i
            return -1 
            
    def solve(self, hint=None, lb=1):
        self.problem.setObjectiveFunction(self.llrVector)
        self.problem.unfixVariables(range(self.code.blocklength))
        node = Node()
        activeNodes = []
        candidate = None
        ub = np.inf
    
        for i in itertools.count():
            logger.info("iteration {:3d}, active nodes: {}".format(i, len(activeNodes)))
            if node.lb >= ub:
                # prune by bound
                logger.info("node pruned by bound 1")
                if node.parent is not None:
                    node.parent.updateBound(node.lb, node.branchValue)
            else:
                # solve
                ans = self.problem.solve()
                node.lb = self.problem.objectiveValue
                if node.parent is not None:
                    node.parent.updateBound(node.lb, node.branchValue)
                if ans == 1:
                    # solution is integral
                    logger.info("node pruned by integrality")
                    if self.problem.objectiveValue < ub:
                        candidate = self.problem.solution.copy()
                        ub = self.problem.objectiveValue
                        logger.info("ub improved to {}".format(ub))
                elif ans == -1:
                    logger.info("node pruned by infeasibility")
                elif ans == 0:
                    if node.lb < ub:
                        # branch
                        branchIndex = self.branchIndex()
                        activeNodes.append(Node(parent=node, branchIndex=branchIndex, branchValue=0))
                        activeNodes.append(Node(parent=node, branchIndex=branchIndex, branchValue=1))
                    else:
                        logger.info("node pruned by bound 2")
                ans = self.problem.solveHeuristic()
                if ans == 1:
                    if self.problem.hObjectiveValue < ub:
                        print('heuristic improved ub')
                        candidate = self.problem.hSolution.copy()
                        ub = self.problem.hObjectiveValue
            
            if len(activeNodes) == 0:
                break
            if self.method == "dfs":
                newNode = activeNodes.pop()
            elif self.method == "bbs":
                newNode = min(activeNodes, key=lambda n: -n.lb)
                activeNodes.remove(newNode)
            elif self.method == "wbs":
                newNode = min(activeNodes, key=lambda n: n.lb)
            elif self.method == "bfs":
                newNode = activeNodes.pop(0)
            else:
                raise ValueError("wrong method")
            move(self.problem, node, newNode)
            node = newNode
        self.solution = candidate
        self.objectiveValue = ub
        if "nodes" not in self.stats:
            self.stats["nodes"] = 0
        self.stats["nodes"] += i
        
    def params(self):
        return dict() #todo
            
        
            
            