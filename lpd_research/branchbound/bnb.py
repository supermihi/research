# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function
import numpy as np
from collections import deque

class BranchAndBound:
    
    def __init__(self, problem, eps=1e-6, BranchMethod=DFSMethod): 
        self.problem = problem
        self.eps = eps
        self.root = Node()
        #the method used to branch at Nodes
        self.bMethod = BranchMethod
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
        #moveCount = 0
        while True:
            
            #select one of the active nodes, move there and (solve the corresponding problem)
            try:
                activeNew = self.bMethod.getActiveNode(self)
            except NodesExhausted:
                break
            print("active node: {}".format(activeNew))
            
            (fixC, unfixC) = self.move(activeOld, activeNew)
            fixCount = fixCount + fixC
            unfixCount = unfixCount + unfixC
            
            self.problem.solve()
            
            if self.problem.solution != None:
                #find the Variable to be branched in this node
                branchVariable = self.findVariable(self.problem.solution)
                if branchVariable is not None:
                    print("Variable x{}={} is not integral".format(branchVariable, self.problem.solution[branchVariable]))
                #update bounds of all nodes if neccesary
                activeNew.lowerb = self.problem.objectiveValue
                
                
                #hier eventuell heuristik für upperbound einfügen
                
                
                
                if branchVariable is None:
                    if self.optimalObjectiveValue > self.problem.objectiveValue:
                        self.optimalSolution = self.problem.solution
                        self.optimalObjectiveValue = self.problem.objectiveValue
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
                    activeNew.child0 = Node(activeNew, branchVariable, 0)
                    activeNew.child1 = Node(activeNew, branchVariable, 1)
                    self.bMethod.addNodes(activeNew.child0,activeNew.child1)
                    branchCount = branchCount + 1
            else:
                activeNew.lowerb = np.inf
                self.updBound(activeNew)
            activeOld = activeNew
        
        print("******* optimal solution found *******")
        print(self.optimalSolution)
        print(self.optimalObjectiveValue)
        print("BranchCount: {count}; FixCount: {fix}, UnfixCount: {unfix}".format(count=branchCount, fix=fixCount, unfix=unfixCount))
        return self.optimalSolution
             
            
    
    def move(self, fromNode, toNode):
        """Moves problem from fromNode to toNode.
        """
        unfixCount = 0
        fixCount = 0
        fix = []
        print('moving from {} to {}'.format(fromNode, toNode))
        while fromNode.depth > toNode.depth:
            self.problem.unfixVariable(fromNode.branchVariable)
            unfixCount = unfixCount + 1
            print('unfix variable {}'.format(fromNode.branchVariable))
            fromNode = fromNode.parent
        
        while toNode.depth > fromNode.depth:
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            fixCount = fixCount + 1
            toNode = toNode.parent
            
        while toNode is not fromNode:
            print('unfix variable* {}'.format(fromNode.branchVariable))
            self.problem.unfixVariable(fromNode.branchVariable)
            unfixCount = unfixCount +1
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            fromNode = fromNode.parent
            toNode = toNode.parent
        print("Fix list: {}".format(fix))
        for var, value in fix:
            self.problem.fixVariable(var, value)
            fixCount = fixCount + 1
        return (fixCount, unfixCount)
    
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

class BranchMethod:
    
    def __init__(self, rootNode):
        self.root = rootNode
    
    def getActiveNode(self):
        """Return next active node. If all nodes are exhausted, raises an NodeExhausted exception."""
        pass
    
    def addNodes(self, node0, node1):
        pass

class BFSMethod(BranchMethod):
    
    def __init__(self, rootNode):
        BranchMethod.__init__(self, rootNode)
        self.activeNodes = deque( [rootNode] )
        
    def getActiveNode(self):
        try:
            return self.activeNodes.popleft()
        except IndexError:
            raise NodesExhausted()
    
    def addNodes(self, node0, node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
class DFSMethod(BranchMethod):
    
    def __init__(self, rootNode):
        BranchMethod.__init__(self, rootNode)
        self.activeNodes = deque ( [rootNode])

    def getActiveNode(self):
        try:
            return self.activeNodes.pop()
        except IndexError:
            raise NodesExhausted()
        
    def addNodes(self, node0, node1):
        self.activeNodes.append(node1)
        self.activeNodes.append(node0)
        
class BBSMethod(BranchMethod):
    
    def __init__(self, rootNode):
        BranchMethod.__init__(self, rootNode)
        self.activeNodes = [rootNode]
        
    def getActiveNode(self):
        mini = np.inf
        #in default the first node is returned
        i = 0
        for (i,x) in enumerate(self.activeNodes):
            if mini > x.lowerb:
                mini = x.lowerb
                place = i
        return self.activeNodes.pop(place)
    
    def addNodes(self,node0, node1):
        self.activeNodes.extend([node1, node0])
        
            
         
#===============================================================================
# def getActiveDFS(bnb):
#    return bnb.activeNodes.pop()       
#  
#            
# def getActiveBFS(bnb):
#    return bnb.activeNodes.popleft()
#===============================================================================

class Node:
    
    def __init__(self, parent=None, branchVariable=None, branchValue=None):
        assert branchVariable is None or branchVariable >= 0
        self.parent = parent
        self.child0 = None
        self.child1 = None
        #nec to use BestBound
        self.solution = None
        self.objectiveValue = None
        
        self.branchVariable = branchVariable
        self.branchValue = branchValue
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth+1
        self.lowerb = -np.inf
        self.upperb = np.inf
        
    def __str__(self):
        return "Node({}/{} with lb={},ub={})".format(self.branchVariable, self.branchValue, self.lowerb, self.upperb)