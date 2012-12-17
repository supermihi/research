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
from lpdecoding.utils import stopwatch
import numpy as np

cdef class BranchMethod:
    
    def __init__(self, rootNode, problem):
        self.root = rootNode
        self.problem = problem
        self.FirstSolutionExists = False
        self.lpTime = 0
        
    cdef (int, int, int) refreshActiveNodes(self, Node activeOld):
        return (0,0,0)
    
    def getActiveNode(self, activeOld):
        """Return next active node. If all nodes are exhausted, raises an NodeExhausted exception."""
        pass
    
    def addNodes(self, node0, node1):
        pass
    
    def createNodes(self, branchVariable, parent):
        pass
    
    cdef (int, int) move(self,Node fromNode, Node toNode):
        """Moves problem from fromNode to toNode.
        """
        cdef:
            int unfixCount, fixCount
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


cdef class BFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque( [rootNode] )
        
    cdef (Node, int, int) getActiveNode(self, Node activeOld):
        cdef:
            int fixC, unfixC
            Node activeNode
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef (int, int) createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        return (0, 0)
        
cdef class BFSRandom(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque( [rootNode] )
        
    cdef (Node, int, int) getActiveNode(self, Node activeOld):
        cdef:
            int fixC, unfixC
            Node activeNode
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    
    cdef void addNodes(self, Node node0, Node node1):
        if int np.random.randint(0, 2) == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    def (int, int) createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        return (0, 0)
    
cdef class BFSRound(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque( [rootNode] )
        
    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    
    def addNodes(self, node0, node1):
        k = node0.parent.solution[node0.branchVariable]
        if k > 0.5:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        elif k < 0.5:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        elif k == 0.5:
            if np.random.randint(0, 2) == 0:
                self.activeNodes.append(node1)
                self.activeNodes.append(node0)
            else:
                self.activeNodes.append(node0)
                self.activeNodes.append(node1)
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        return (0, 0) 
        
cdef class DFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
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
        return (0, 0)
        
cdef class DFSRandom(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    def addNodes(self, node0, node1):
        if np.random.randint(0, 2) == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        return (0, 0)
    
cdef class DFSRound(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self, activeOld):
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        (fixC, unfixC) = self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return (activeNode, fixC, unfixC)
        
    def addNodes(self, node0, node1):
        k = node0.parent.solution[node0.branchVariable]
        if k > 0.5:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        elif k < 0.5:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        elif k == 0.5:
            if np.random.randint(0, 2) == 0:
                self.activeNodes.append(node1)
                self.activeNodes.append(node0)
            else:
                self.activeNodes.append(node0)
                self.activeNodes.append(node1)
        
    def createNodes(self, branchVariable, parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        return (0, 0)
        
        
cdef class BBSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
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
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        return (2, 2)              

#DeepSeaTroll Search Method        
cdef class DSTMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque([rootNode])
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
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
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable) 
        return (2, 2)
        

cdef class DFSandBBSMethod(BranchMethod):

    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self, activeOld):
        if not self.FirstSolutionExists:
            try:
                activeNode = self.activeNodes.pop()
            except IndexError:
                raise NodesExhausted()
            (fixC, unfixC) = self.move(activeOld, activeNode)
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            activeNode.solution = self.problem.solution
            activeNode.objectiveValue = self.problem.objectiveValue
            #self.move(activeNode, activeOld)
            return (activeNode, fixC, unfixC)
        else:
            try:
                activeNode = heapq.heappop(self.activeNodes)[1]
            except IndexError:
                raise NodesExhausted()
            (fixC, unfixC) = self.move(activeOld, activeNode)
            return (activeNode, fixC, unfixC)
        
    def addNodes(self, node0, node1):
        if not self.FirstSolutionExists:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            heapq.heappush(self.activeNodes, (node0.lowerb, node0))
            heapq.heappush(self.activeNodes, (node1.lowerb, node1))
        
    def createNodes(self, branchVariable, parent):
        if not self.FirstSolutionExists:
            parent.child0 = Node(parent, branchVariable, 0)
            parent.child1 = Node(parent, branchVariable, 1)
            return (0,0)
        else:
            parent.child0 = Node(parent, branchVariable, 0)
            self.problem.fixVariable(branchVariable, 0)
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            parent.child0.solution = self.problem.solution
            parent.child0.objectiveValue = self.problem.objectiveValue
            self.problem.unfixVariable(branchVariable)
            parent.child1 = Node(parent, branchVariable, 1)
            self.problem.fixVariable(branchVariable, 1)
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            parent.child1.solution = self.problem.solution
            parent.child1.objectiveValue = self.problem.objectiveValue
            self.problem.unfixVariable(branchVariable)
            return (0,0)
            
    def refreshActiveNodes(self, activeOld):
        newNodes = []
        oldNode = activeOld
        unfixC = 0
        moveC = 0
        fixC = 0
        for i in self.activeNodes:
            (fixCount, unfixCount) = self.move(activeOld, i)
            unfixC += unfixCount
            fixC += fixCount
            moveC += 1
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            i.solution = self.problem.solution
            i.objectiveValue = self.problem.objectiveValue
            heapq.heappush(newNodes, (i.lowerb, i))
            activeOld = i
        (fixCount, unfixCount) = self.move(activeOld, oldNode)
        unfixC += unfixCount
        fixC += fixCount            
        moveC += 1
        return (fixC, unfixC, moveC)
    