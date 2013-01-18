# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import absolute_import

#from collections import deque
import heapq
from branchbound.bnb cimport Node
from branchbound.myList cimport myDeque
from branchbound.myList import MyIndexError
from .bnb import NodesExhausted
from lpdecoding.utils import stopwatch
import numpy as np

cdef class BranchMethod:
    
    def __init__(self, rootNode, problem):
        self.root = rootNode
        self.problem = problem
        self.FirstSolutionExists = False
        self.lpTime = 0
        self.moveCount = 0
        self.fixCount = 0
        self.unfixCount = 0
        
        
    cdef void refreshActiveNodes(self, Node activeOld):
        pass
    
    cdef Node getActiveNode(self, Node activeOld):
        """Return next active node. If all nodes are exhausted, raises an NodeExhausted exception."""
        raise NotImplementedError()
    
    cdef void addNodes(self, Node node0, Node node1):
        pass
    
    cdef void createNodes(self, int branchVariable, Node parent):
        pass
    
    cdef void move(self,Node fromNode, Node toNode):
        """Moves problem from fromNode to toNode.
        """
        fix = []
        #logging.debug('moving from {} to {}'.format(fromNode, toNode))
        while fromNode.depth > toNode.depth:
            self.problem.unfixVariable(fromNode.branchVariable)
            self.unfixCount += 1
            #logging.debug('unfix variable {}'.format(fromNode.branchVariable))
            fromNode = fromNode.parent
        
        while toNode.depth > fromNode.depth:
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            self.fixCount += 1
            toNode = toNode.parent
            
        while toNode is not fromNode:
            #logging.debug('unfix variable* {}'.format(fromNode.branchVariable))
            self.problem.unfixVariable(fromNode.branchVariable)
            self.unfixCount = self.unfixCount +1
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            fromNode = fromNode.parent
            toNode = toNode.parent
        #logging.debug("Fix list: {}".format(fix))
        for var, value in fix:
            self.problem.fixVariable(var, value)
            self.fixCount += 1


cdef class BFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.popleft()
        except MyIndexError():
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
cdef class BFSRandom(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
#        try:
        activeNode = self.activeNodes.popleft()
        if activeNode == None:
            return None
#        except MyIndexError():
#            raise NodesExhausted()
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    
    cdef void addNodes(self, Node node0, Node node1):
        cdef: 
            int l
        l = np.random.randint(0, 2)
        print("l: {}".format(l))
        if l == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
    
cdef class BFSRound(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
#        try:
        activeNode = self.activeNodes.popleft()
#        except MyIndexError():
#            raise NodesExhausted()
        if activeNode == None:
            return None
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    
    cdef void addNodes(self, Node node0, Node node1):
        cdef:
            int k
        k = node0.parent.solution[node0.branchVariable]
        if k > 0.5:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        elif k < 0.5:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        elif k == 0.5:
            k = np.random.randint(0, 2)
            if k == 0:
                self.activeNodes.append(node1)
                self.activeNodes.append(node0)
            else:
                self.activeNodes.append(node0)
                self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
         
        
cdef class DFSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except MyIndexError():
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
        
cdef class DFSRandom(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except MyIndexError():
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        cdef:
            int l
        l = np.random.randint(0, 2)
        if l == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
    
cdef class DFSRound(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except MyIndexError():
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        cdef: 
            int k
        k = node0.parent.solution[node0.branchVariable]
        if k > 0.5:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        elif k < 0.5:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        elif k == 0.5:
            k = np.random.randint(0, 2)
            if k == 0:
                self.activeNodes.append(node1)
                self.activeNodes.append(node0)
            else:
                self.activeNodes.append(node0)
                self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
        
cdef class BBSMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.root.lowerb = self.problem.objectiveValue
        self.activeNodes = [ (rootNode.lowerb, rootNode) ]
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = heapq.heappop(self.activeNodes)[1]
        except IndexError:
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        return activeNode
         
    cdef void addNodes(self, Node node0, Node node1):
        heapq.heappush(self.activeNodes, (node0.lowerb, node0))
        heapq.heappush(self.activeNodes, (node1.lowerb, node1))
        
    cdef void createNodes(self, int branchVariable, Node parent):
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
        self.unfixCount += 2
        self.fixCount += 2              

#DeepSeaTroll Search Method        
cdef class DSTMethod(BranchMethod):
    
    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.problem.objectiveValue
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except MyIndexError():
            raise NodesExhausted()
        self.move(activeOld, activeNode)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        if node0.objectiveValue < node1.objectiveValue:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
            
    cdef void createNodes(self, int branchVariable, Node parent):
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
        self.unfixCount += 2
        self.fixCount += 2
        

cdef class DFSandBBSMethod(BranchMethod):

    def __init__(self, rootNode, problem):
        BranchMethod.__init__(self, rootNode, problem)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        if not self.FirstSolutionExists:
            try:
                activeNode = self.activeNodes.pop()
            except MyIndexError():
                raise NodesExhausted()
            self.move(activeOld, activeNode)
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            activeNode.solution = self.problem.solution
            activeNode.objectiveValue = self.problem.objectiveValue
            #self.move(activeNode, activeOld)
            return activeNode
        else:
            try:
                activeNode = heapq.heappop(self.activeNodes)[1]
            except IndexError:
                raise NodesExhausted()
            self.move(activeOld, activeNode)
            return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        if not self.FirstSolutionExists:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            heapq.heappush(self.activeNodes, (node0.lowerb, node0))
            heapq.heappush(self.activeNodes, (node1.lowerb, node1))
        
    cdef void createNodes(self, int branchVariable, Node parent):
        if not self.FirstSolutionExists:
            parent.child0 = Node(parent, branchVariable, 0)
            parent.child1 = Node(parent, branchVariable, 1)
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
            
    cdef void refreshActiveNodes(self, Node activeOld):
        cdef:
            #heap newNodes
            Node oldNode, moveNode
            int i, leng
        newNodes = []
        oldNode = activeOld
        leng = self.activeNodes.length
        for i from 0 <= i < leng:
            moveNode = self.activeNodes.pop()
            self.move(activeOld, moveNode)
            self.moveCount += 1
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            moveNode.solution = self.problem.solution
            moveNode.objectiveValue = self.problem.objectiveValue
            heapq.heappush(newNodes, (moveNode.lowerb, moveNode))
            activeOld = moveNode
#        for i in self.activeNodes:
#            (fixCount, unfixCount) = self.move(activeOld, i)
#            unfixC += unfixCount
#            fixC += fixCount
#            moveC += 1
#            with stopwatch() as timer:
#                self.problem.solve()
#            self.lpTime += timer.duration
#            i.solution = self.problem.solution
#            i.objectiveValue = self.problem.objectiveValue
#            heapq.heappush(newNodes, (i.lowerb, i))
#            activeOld = i
        self.move(activeOld, oldNode)            
        self.moveCount += 1
    
