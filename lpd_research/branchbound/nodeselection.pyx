# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import absolute_import

from collections import deque
import heapq
from libc.math cimport fmin
from branchbound.bnb cimport Node
from branchbound import branchrules
from branchbound.myList cimport myDeque
from lpdecoding.utils import stopwatch
import numpy as np


cdef class SelectionMethod:
    
    def __init__(self, rootNode, problem, branchRule):
        self.root = rootNode
        self.branchRule = branchRule
        self.problem = problem
        self.FirstSolutionExists = False
        self.lpTime = 0
        self.moveTime = 0
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
        self.moveCount += 1
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

    cdef void updBound(self, Node node):
        """Updates lower and upper bounds for node and all parent nodes, if possible.
        """
        cdef:
            double ub, lb, ubp, lbp, ubb, lbb, upper, lower
        if node.parent is None:
            return
        ub = node.upperb
        lb = node.lowerb
        #upper bound parent => ubp; lower bound parent => lbp
        ubp = node.parent.upperb
        lbp = node.parent.lowerb
        if node.branchValue == 1:
            #upper bound brother => ubb; lower bound brother => lbb
            ubb = node.parent.child0.upperb
            lbb = node.parent.child0.lowerb
        elif node.branchValue == 0:
            ubb = node.parent.child1.upperb
            lbb = node.parent.child1.lowerb
        upper = fmin(ubb, ub)
        lower = fmin(lbb, lb)
        #update of upper and lower bound
        if upper < ubp and lower > lbp:
            node.parent.lowerb = lower
            node.parent.upperb = upper
            self.updBound(node.parent)
        elif upper < ubp:
            node.parent.upperb = upper
            self.updBound(node.parent)
        elif lower > lbp:
            node.parent.lowerb = lower
            self.updBound(node.parent)

cdef class BFSMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
#        print("root upper: {}, root lower: {}".format(self.root.upperb, self.root.lowerb))
#        print("root objectiveValue: {}".format(self.root.objectiveValue))
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)

cdef class MyBFSMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = myDeque(rootNode)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        #try:
        activeNode = self.activeNodes.popleft()
        #except MyIndexError():
        #    raise NodesExhausted()
        if activeNode == None:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
        #self.move(activeNode, activeOld)
        return activeNode
           
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        
        
cdef class BFSRandom(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
        #self.move(activeNode, activeOld)
        return activeNode
          
    cdef void addNodes(self, Node node0, Node node1):
        cdef: 
            int l
        l = np.random.randint(0, 2)
        #print("l: {}".format(l))
        if l == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
             
cdef class MyBFSRandom(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
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
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
        #self.move(activeNode, activeOld)
        return activeNode
         
    cdef void addNodes(self, Node node0, Node node1):
        cdef: 
            int l
        l = np.random.randint(0, 2)
        #print("l: {}".format(l))
        if l == 0:
            self.activeNodes.append(node1)
            self.activeNodes.append(node0)
        else:
            self.activeNodes.append(node0)
            self.activeNodes.append(node1)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        

cdef class BFSRound(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.popleft()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
             
cdef class MyBFSRound(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
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
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
         

cdef class DFSMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            if activeNode.depth > 0 and activeNode.parent.lowerb > -np.inf:
                self.problem.solve(activeNode.parent.lowerb)
            else:
                self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
                
cdef class MyDFSMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
            double lb=1
        #try:
        activeNode = self.activeNodes.pop()
        if activeNode == None:
            return None
        #except MyIndexError():
            #raise NodesExhausted()
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            if activeNode.parent is not None:
                lb = activeNode.parent.lowerb
            if self.problem.solve(lb, self.root.upperb) == -2:
                activeNode.upperb = activeNode.lowerb = self.root.upperb
                return activeNode
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
        #self.move(activeNode, activeOld)
        return activeNode
        
    cdef void addNodes(self, Node node0, Node node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
    cdef void createNodes(self, int branchVariable, Node parent):
        parent.child0 = Node(parent, branchVariable, 0)
        parent.child1 = Node(parent, branchVariable, 1)
        

cdef class DFSRandom(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
                
cdef class MyDFSRandom(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        #try:
        activeNode = self.activeNodes.pop()
        #except MyIndexError():
        #    raise NodesExhausted()
        if activeNode == None:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
        

cdef class DFSRound(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
            
cdef class MyDFSRound(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = myDeque(rootNode)

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        #try:
        activeNode = self.activeNodes.pop()
        #except MyIndexError():
        #    raise NodesExhausted()
        if activeNode == None:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        activeNode.solution = self.problem.solution
        activeNode.objectiveValue = self.problem.objectiveValue
        activeNode.lowerb = activeNode.objectiveValue
        if activeNode.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
            if activeNode.varToBranch == -1:
                activeNode.upperb = activeNode.objectiveValue
                if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = activeNode.upperb
                    self.root.solution = activeNode.solution
            else:
                if self.problem.hSolution is not None:
                    activeNode.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(activeNode)
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
        
        
cdef class BBSMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.problem.objectiveValue
        self.root.lowerb = self.root.objectiveValue
        if self.root.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            self.root.varToBranch = self.branchRule.selectVariable(self.root.solution)
            if self.root.varToBranch == -1:
                self.root.upperb = self.root.objectiveValue
                if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = self.root.upperb
            else:
                if self.problem.hSolution is not None:
                    self.root.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = self.root.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(self.root)
        self.activeNodes = [ (rootNode.lowerb, rootNode) ]
        
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = heapq.heappop(self.activeNodes)[1]
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
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
        parent.child0.lowerb = parent.child0.objectiveValue
        if parent.child0.solution is not None:
            parent.child0.varToBranch = self.branchRule.selectVariable(parent.child0.solution)
            if parent.child0.varToBranch == -1:
                parent.child0.upperb = parent.child0.objectiveValue
                if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child0.upperb
                    self.root.solution = parent.child0.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child0.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child0.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child0)
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        parent.child1.lowerb = parent.child1.objectiveValue
        if parent.child1.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            parent.child1.varToBranch = self.branchRule.selectVariable(parent.child1.solution)
            if parent.child1.varToBranch == -1:
                parent.child1.upperb = parent.child1.objectiveValue
                if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child1.upperb
                    self.root.solution = parent.child1.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child1.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child1.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child1)
        self.problem.unfixVariable(branchVariable)
        self.unfixCount += 2
        self.fixCount += 2              


#DeepSeaTroll Search Method        
cdef class DSTMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.problem.objectiveValue
        self.root.lowerb = self.root.objectiveValue
        if self.root.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            self.root.varToBranch = self.branchRule.selectVariable(self.root.solution)
            if self.root.varToBranch == -1:
                self.root.upperb = self.root.objectiveValue
                if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = self.root.upperb
            else:
                if self.problem.hSolution is not None:
                    self.root.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = self.root.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(self.root)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        try:
            activeNode = self.activeNodes.pop()
        except IndexError:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
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
            if self.problem.solve(parent.lowerb, parent.upperb) == -2:
                parent.child0.upperb = self.problem.objectiveValue
                parent.child0.solution = parent.solution
            else:
                parent.child0.solution = self.problem.solution
        self.lpTime += timer.duration
        parent.child0.objectiveValue = parent.child0.lowerb = self.problem.objectiveValue
        if parent.child0.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            parent.child0.varToBranch = self.branchRule.selectVariable(parent.child0.solution)
            if parent.child0.varToBranch == -1:
                parent.child0.upperb = parent.child0.objectiveValue
                if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child0.upperb
                    self.root.solution = parent.child0.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child0.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child0.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child0)
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        with stopwatch() as timer:
            if self.problem.solve(parent.lowerb, parent.upperb) == -2:
                parent.child1.upperb = self.problem.objectiveValue
                parent.child1.solution = parent.solution
            else:
                parent.child1.solution = self.problem.solution
        self.lpTime += timer.duration
        parent.child1.lowerb = parent.child1.objectiveValue = self.problem.objectiveValue
        if parent.child1.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            parent.child1.varToBranch = self.branchRule.selectVariable(parent.child1.solution)
            if parent.child1.varToBranch == -1:
                parent.child1.upperb = parent.child1.objectiveValue
                if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child1.upperb
                    self.root.solution = parent.child1.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child1.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child1.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child1)
        self.problem.unfixVariable(branchVariable) 
        self.unfixCount += 2
        self.fixCount += 2
        
#DeepSeaTroll Search Method        
cdef class MyDSTMethod(SelectionMethod):
    
    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = myDeque(rootNode)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        self.root.solution = self.problem.solution
        self.root.objectiveValue = self.problem.objectiveValue
        self.root.lowerb = self.root.objectiveValue
        if self.root.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            self.root.varToBranch = self.branchRule.selectVariable(self.root.solution)
            if self.root.varToBranch == -1:
                self.root.upperb = self.root.objectiveValue
                if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = self.root.upperb
            else:
                if self.problem.hSolution is not None:
                    self.root.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > self.root.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = self.root.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(self.root)
        
    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        #try:
        activeNode = self.activeNodes.pop()
        #except MyIndexError():
        #    raise NodesExhausted()
        if activeNode == None:
            return None
        with stopwatch() as moveTimer:
            self.move(activeOld, activeNode)
        self.moveTime += moveTimer.duration
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
        parent.child0.lowerb = parent.child0.objectiveValue
        if parent.child0.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            parent.child0.varToBranch = self.branchRule.selectVariable(parent.child0.solution)
            if parent.child0.varToBranch == -1:
                parent.child0.upperb = parent.child0.objectiveValue
                if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child0.upperb
                    self.root.solution = parent.child0.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child0.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child0.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child0)
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node(parent, branchVariable, 1)
        self.problem.fixVariable(branchVariable, 1)
        with stopwatch() as timer:
            self.problem.solve()
        self.lpTime += timer.duration
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        parent.child1.lowerb = parent.child1.objectiveValue
        if parent.child1.solution is not None:
            #print("{}".format(self.branchRule.selectVariable()))
            parent.child1.varToBranch = self.branchRule.selectVariable(parent.child1.solution)
            if parent.child1.varToBranch == -1:
                parent.child1.upperb = parent.child1.objectiveValue
                if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                    self.root.objectiveValue = parent.child1.upperb
                    self.root.solution = parent.child1.solution
            else:
                if self.problem.hSolution is not None:
                    parent.child1.upperb = self.problem.hObjectiveValue 
                    if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child1.upperb
                        self.root.solution = self.problem.hSolution
        self.updBound(parent.child1)
        self.problem.unfixVariable(branchVariable) 
        self.unfixCount += 2
        self.fixCount += 2
        

cdef class DFSandBBSMethod(SelectionMethod):

    def __init__(self, rootNode, problem, branchRule):
        SelectionMethod.__init__(self, rootNode, problem, branchRule)
        self.activeNodes = deque([rootNode])

    cdef Node getActiveNode(self, Node activeOld):
        cdef:
            Node activeNode
        if not self.FirstSolutionExists:
            try:
                activeNode = self.activeNodes.pop()
            except IndexError:
                return None
            with stopwatch() as moveTimer:
                self.move(activeOld, activeNode)
            self.moveTime += moveTimer.duration
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            activeNode.solution = self.problem.solution
            activeNode.objectiveValue = self.problem.objectiveValue
            activeNode.lowerb = activeNode.objectiveValue
            if activeNode.solution is not None:
                #print("{}".format(self.branchRule.selectVariable()))
                activeNode.varToBranch = self.branchRule.selectVariable(activeNode.solution)
                if activeNode.varToBranch == -1:
                    activeNode.upperb = activeNode.objectiveValue
                    if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = activeNode.upperb
                        self.root.solution = activeNode.solution
                else:
                    if self.problem.hSolution is not None:
                        activeNode.upperb = self.problem.hObjectiveValue 
                        if self.root.objectiveValue > activeNode.upperb or not self.FirstSolutionExists:
                            self.root.objectiveValue = activeNode.upperb
                            self.root.solution = self.problem.hSolution
            self.updBound(activeNode)
            #self.move(activeNode, activeOld)
            return activeNode
        else:
            try:
                activeNode = heapq.heappop(self.activeNodes)[1]
            except IndexError:
                return None
            with stopwatch() as moveTimer:
                self.move(activeOld, activeNode)
            self.moveTime += moveTimer.duration
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
            parent.child0.lowerb = parent.child0.objectiveValue
            if parent.child0.solution is not None:
                #print("{}".format(self.branchRule.selectVariable()))
                parent.child0.varToBranch = self.branchRule.selectVariable(parent.child0.solution)
                if parent.child0.varToBranch == -1:
                    parent.child0.upperb = parent.child0.objectiveValue
                    if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child0.upperb
                        self.root.solution = parent.child0.solution
                else:
                    if self.problem.hSolution is not None:
                        parent.child0.upperb = self.problem.hObjectiveValue 
                        if self.root.objectiveValue > parent.child0.upperb or not self.FirstSolutionExists:
                            self.root.objectiveValue = parent.child0.upperb
                            self.root.solution = self.problem.hSolution
            self.updBound(parent.child0)
            self.problem.unfixVariable(branchVariable)
            parent.child1 = Node(parent, branchVariable, 1)
            self.problem.fixVariable(branchVariable, 1)
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            parent.child1.solution = self.problem.solution
            parent.child1.objectiveValue = self.problem.objectiveValue
            parent.child1.lowerb = parent.child1.objectiveValue
            if parent.child1.solution is not None:
                #print("{}".format(self.branchRule.selectVariable()))
                parent.child1.varToBranch = self.branchRule.selectVariable(parent.child1.solution)
                if parent.child1.varToBranch == -1:
                    parent.child1.upperb = parent.child1.objectiveValue
                    if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = parent.child1.upperb
                        self.root.solution = parent.child1.solution
                else:
                    if self.problem.hSolution is not None:
                        parent.child1.upperb = self.problem.hObjectiveValue 
                        if self.root.objectiveValue > parent.child1.upperb or not self.FirstSolutionExists:
                            self.root.objectiveValue = parent.child1.upperb
                            self.root.solution = self.problem.hSolution
            self.updBound(parent.child1)
            self.problem.unfixVariable(branchVariable)
            
    cdef void refreshActiveNodes(self, Node activeOld):
        cdef:
            #heap newNodes
            Node oldNode, moveNode
            #int i, leng
        newNodes = []
        oldNode = activeOld
#        leng = len(self.activeNodes)
#        for i from 0 <= i < leng:
#            moveNode = self.activeNodes.pop()
#            self.move(activeOld, moveNode)
#            #self.moveCount += 1
#            with stopwatch() as timer:
#                self.problem.solve()
#            self.lpTime += timer.duration
#            moveNode.solution = self.problem.solution
#            moveNode.objectiveValue = self.problem.objectiveValue
#            heapq.heappush(newNodes, (moveNode.lowerb, moveNode))
#            activeOld = moveNode
        for i in self.activeNodes:
            with stopwatch() as moveTimer:
                self.move(activeOld, i)
            self.moveTime += moveTimer.duration
            with stopwatch() as timer:
                self.problem.solve()
            self.lpTime += timer.duration
            i.solution = self.problem.solution
            i.objectiveValue = self.problem.objectiveValue
            i.lowerb = i.objectiveValue
            if i.solution is not None:
                #print("{}".format(self.branchRule.selectVariable()))
                i.varToBranch = self.branchRule.selectVariable(i.solution)
                if i.varToBranch == -1:
                    i.upperb = i.objectiveValue
                    if self.root.objectiveValue > i.upperb or not self.FirstSolutionExists:
                        self.root.objectiveValue = i.upperb
                        self.root.solution = i.solution
                else:
                    if self.problem.hSolution is not None:
                        i.upperb = self.problem.hObjectiveValue 
                        if self.root.objectiveValue > i.upperb or not self.FirstSolutionExists:
                            self.root.objectiveValue = i.upperb
                            self.root.solution = self.problem.hSolution
            self.updBound(i)
            heapq.heappush(newNodes, (i.lowerb, i))
            activeOld = i
        with stopwatch() as moveTimer:
            self.move(activeOld, oldNode)            
        self.moveTime += moveTimer.duration
        self.activeNodes = newNodes
        #self.moveCount += 1
    
