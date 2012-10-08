# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import print_function
import numpy as np
from collections import deque, set
import heapq

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
        #fix = []
        print('moving from {} to {}'.format(fromNode, toNode))
        fix = toNode.copy() - fromNode.copy()
        unfix = fromNode.copy() - toNode.copy()
        self.problem.fixVariables(fix)
        self.problem.unfixVariables(unfix)
        return (len(fix), len(unfix))
        
        
        
        
#        while fromNode.depth > toNode.depth:
#            self.problem.unfixVariable(fromNode.branchVariable)
#            unfixCount = unfixCount + 1
#            print('unfix variable {}'.format(fromNode.branchVariable))
#            fromNode = fromNode.parent
#        
#        while toNode.depth > fromNode.depth:
#            fix.append( (toNode.branchVariable, toNode.branchValue) )
#            fixCount = fixCount + 1
#            toNode = toNode.parent
#            
#        while toNode is not fromNode:
#            print('unfix variable* {}'.format(fromNode.branchVariable))
#            self.problem.unfixVariable(fromNode.branchVariable)
#            unfixCount = unfixCount +1
#            fix.append( (toNode.branchVariable, toNode.branchValue) )
#            fromNode = fromNode.parent
#            toNode = toNode.parent
#        print("Fix list: {}".format(fix))
#        for var, value in fix:
#            self.problem.fixVariable(var, value)
#            fixCount = fixCount + 1
#        return (fixCount, unfixCount)
    
    
    

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
        parent.child0 = Node2(parent, branchVariable, 0, parent.branchVarVal)
        parent.child1 = Node2(parent, branchVariable, 1, parent.branchVarVal)
        
        
        
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
        parent.child0 = Node2(parent, branchVariable, 0, parent.branchVarVal)
        parent.child1 = Node2(parent, branchVariable, 1, parent.branchVarVal)
        
    
        
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
        parent.child0 = Node2(parent, branchVariable, 0, parent.branchVarVal)
        self.problem.fixVariable(branchVariable, 0)
        self.problem.solve()
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node2(parent, branchVariable, 1, parent.branchVarVal)
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
        parent.child0 = Node2(parent, branchVariable, 0, parent.branchVarVal)
        self.problem.fixVariable(branchVariable, 0)
        self.problem.solve()
        parent.child0.solution = self.problem.solution
        parent.child0.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable)
        parent.child1 = Node2(parent, branchVariable, 1, parent.branchVarVal)
        self.problem.fixVariable(branchVariable, 1)
        self.problem.solve()
        parent.child1.solution = self.problem.solution
        parent.child1.objectiveValue = self.problem.objectiveValue
        self.problem.unfixVariable(branchVariable) 

class BranchAndBound:
    
    def __init__(self, problem, eps=1e-6, branchMethod=DFSMethod): 
        self.problem = problem
        self.eps = eps
        self.root = Node2()
        #the method used to branch at Nodes
        self.bMethod = branchMethod(self.root, self.problem)
        #self.activeNodes = deque([self.root])
        self.optimalSolution = None
        self.optimalObjectiveValue = np.inf
        #self.getActiveNode = selectionMethod
    
    def run(self):
        """Builds tree and runs a branch and bound algorithm.
        """
        activeOld = self.root
        branchCount = 0
        fixCount = 0
        unfixCount = 0
        moveCount = 0
        while True:
            print('starting main loop')
            #select one of the active nodes, move there and (solve the corresponding problem)
            try:
                (activeNew, fixC, unfixC) = self.bMethod.getActiveNode(activeOld)
            except NodesExhausted:
                break
            print("active node: {}".format(activeNew))
            
            #(fixC, unfixC) = self.bMethod.move(activeOld, activeNew)
            fixCount += fixC
            unfixCount += unfixC
            moveCount += 1
            
            if activeNew.solution is not None:
                #find the Variable to be branched in this node
                branchVariable = self.findVariable(activeNew.solution)
                if branchVariable is not None:
                    print("Variable x{}={} is not integral"
                          .format(branchVariable, activeNew.solution[branchVariable]))
                #update bounds of all nodes if neccesary
                activeNew.lowerb = activeNew.objectiveValue
                
                
                #hier eventuell heuristik für upperbound einfügen
                
                
                
                if branchVariable is None:
                    # have a feasible solution
                    if self.optimalObjectiveValue > activeNew.objectiveValue:
                        self.optimalSolution = activeNew.solution
                        self.optimalObjectiveValue = activeNew.objectiveValue
                    activeNew.upperb = self.problem.objectiveValue
                self.updBound(activeNew)
            
                #create new children or close branch
                if activeNew.lowerb > self.root.upperb:
                    pass
                elif abs(activeNew.lowerb - activeNew.upperb) < self.eps:
                    pass
                elif branchVariable is None:
                    pass
                else:
                    #create children with branchValue and add them to the activeNodes-list
                    self.bMethod.createNodes(branchVariable, activeNew)
                    #activeNew.child0 = Node(activeNew, branchVariable, 0)
                    #activeNew.child1 = Node(activeNew, branchVariable, 1)
                    self.bMethod.addNodes(activeNew.child0,activeNew.child1)
                    branchCount += 1
            else:
                activeNew.lowerb = np.inf
                self.updBound(activeNew)
            activeOld = activeNew
        
        #match fix and unfix count to the selected branchMethod
        if self.bMethod == BBSMethod or self.bMethod == DSTMethod:
            fixCount += 2*branchCount
            unfixCount += 2*branchCount
            
        self.moveCount = moveCount
        self.fixCount = fixCount
        self.unfixCount = unfixCount
        self.branchCount = branchCount
        print("******* optimal solution found *******")
        print(self.optimalSolution)
        print(self.optimalObjectiveValue)
        print("BranchCount: {count}; FixCount: {fix}, UnfixCount: {unfix}".format(count=branchCount, fix=fixCount, unfix=unfixCount))
        print("MoveCount: {move}".format(move=moveCount))
        #print("Bei DSTMethod und BBSMethod sind fixCount und UnfixCount je um 2 mal den BranchCount erhöht.")
        #print("Bei DFSMethod und BFSMethod ist der moveCount verdreifacht.")
        return self.optimalSolution
             
            
    

    def updBound(self, node):
        """Updates lower and upper bounds for node and all parent nodes, if possible.
        """
        if node.parent == None:
            pass
        else:
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
            upper = min(ubb, ub)
            lower = min(lbb, lb)
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
        
        
    
    def findVariable(self, vec):
        """Specifies the variable to be branched, or None if *vec* is integral.
        """
        for (i,x) in enumerate(vec):
            if x > 0 + self.eps and x < 1 - self.eps:
                return i
        return None
        
    #===========================================================================
    # def getActiveNode(self):
    #    """Gets one active node depending on searchrule from the activeNode-list.
    #    """
    #    if self.depthFirst:
    #        return self.activeNodes.pop()
    #    else:
    #        return self.activeNodes.popleft()
    #===========================================================================

class NodesExhausted(Exception):
    pass


        
            
         
#===============================================================================
# def getActiveDFS(bnb):
#    return bnb.activeNodes.pop()       
#  
#            
# def getActiveBFS(bnb):
#    return bnb.activeNodes.popleft()
#===============================================================================

class Node2:
    
    def __init__(self, parent=None, branchVariable=None, branchValue=None, branchVarVal=None):
        assert branchVariable is None or branchVariable >= 0
        self.parent = parent
        self.child0 = None
        self.child1 = None
        #nec to use BestBound
        self.solution = None
        self.objectiveValue = None
        
        self.branchVarVal = branchVarVal.copy().add((branchVariable, branchValue))
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth+1
        self.lowerb = -np.inf
        self.upperb = np.inf
        
    def __str__(self):
        return "Node({}/{} with lb={},ub={})".format(self.branchVariable, self.branchValue, self.lowerb, self.upperb)
    
    
#TODO: Node2class with (BranchVariabl, BranchValue) - List