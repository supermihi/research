# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import absolute_import

from collections import deque
import heapq
from .bnb import Node, NodesExhausted

class BranchMethod:
    
    def __init__(self, rootNode, problem):
        self.root = rootNode
        self.problem = problem
    
    def getActiveNode(self, activeOld):
        """Return next active node. If all nodes are exhausted, raises an NodeExhausted exception."""
        pass
    
    def addNodes(self, node0, node1):
        pass
    
    def createNodes(self, branchVariable, parent):
        pass
    
    def move(self, fromNode, toNode):
        """Moves problem from fromNode to toNode.
        """
        unfixCount = 0
        fixCount = 0
        fix = []
        #logging.debug('moving from {} to {}'.format(fromNode, toNode))
        while fromNode.depth > toNode.depth:
            self.problem.unfixVariable(fromNode.branchVariable)
            unfixCount = unfixCount + 1
            #logging.debug('unfix variable {}'.format(fromNode.branchVariable))
            fromNode = fromNode.parent
        
        while toNode.depth > fromNode.depth:
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            fixCount = fixCount + 1
            toNode = toNode.parent
            
        while toNode is not fromNode:
            #logging.debug('unfix variable* {}'.format(fromNode.branchVariable))
            self.problem.unfixVariable(fromNode.branchVariable)
            unfixCount = unfixCount +1
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            fromNode = fromNode.parent
            toNode = toNode.parent
        #logging.debug("Fix list: {}".format(fix))
        for var, value in fix:
            self.problem.fixVariable(var, value)
            fixCount = fixCount + 1
        return (fixCount, unfixCount)


class BFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque( [rootNode] )
        
    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        self.problem.solve()
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    
    def addNodes(self, node0, node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
        
        
class DFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        self.problem.solve()
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    def addNodes(self, node0, node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
    
        
class BBSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.problem.solve()
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.root.lowerb = self.problem.objectiveValue
        self.activeNodes = [ (rootNode.lowerb, rootNode) ]
        
    def getActiveNode(self, activeOld):
        try:
            activeNode = heapq.heappop(self.activeNodes)[1]
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        return (activeNode, fixC, unfixC)
         
    def addNodes(self,node0, node1):
        heapq.heappush(self.activeNodes, (node0.lowerb, node0))
        heapq.heappush(self.activeNodes, (node1.lowerb, node1))
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        self.problem.fixVariable(branchVariable, 0)
        self.problem.solve()
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        self.problem.solve()
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)              

#DeepSeaTroll Search Method        
class DSTMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque([rootNode])
        self.problem.solve()
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.problem.objectiveValue
        
    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        return (activeNode, fixC, unfixC)
        
    def addNodes(self, node0, node1):
        if node0.objectiveValue < node1.objectiveValue:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
            
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        self.problem.fixVariable(branchVariable, 0)
        self.problem.solve()
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        self.problem.solve()
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable) 