# -*- coding: utf-8 -*-
# Copyright 2012 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
import numpy as np
from collections import deque

class BranchAndBound:
    
    def __init__(self, problem, eps=1e-6, depthFirst=True): 
        self.problem = problem
        self.eps = eps
        self.root = Node(depth=0)
        self.activeNodes = deque([self.root])
        self.optimalSolution = None
        self.optimalObjectiveValue = np.inf
        self.depthFirst = depthFirst
    
    def run(self):
        """Builds tree and runs a branch and bound algorithm.
        """
        activeOld = self.root
        while len(self.activeNodes) != 0:
            #select one of the active nodes, move there and solve the corresponding problem
            activeNew = self.getActiveNode()
            self.move(activeOld, activeNew)
            self.problem.solve()
            
            if self.problem.solution != None:
                #find the Variable to be branched in this node
                activeNew.branchVariable = self.findVariable(self.problem.solution)
            
                #update bounds of all nodes if neccesary
                activeNew.lowerb = self.problem.objectiveValue
                
                
                #hier eventuell heuristik für upperbound einfügen
                
                
                
                if activeNew.branchVariable == -1:
                    if self.optimalObjectiveValue > self.problem.solution:
                        self.optimalSolution = self.problem.solution
                        self.optimalObjectiveValue = self.problem.objectiveValue
                    activeNew.upperb = self.problem.obejctiveValue
                self.updBound(activeNew)
            
                #create new children or close branch
                if activeNew.lowerb > self.root.upperb:
                    pass
                elif activeNew.lowerb == activeNew.upperb:
                    pass
                else:
                    #create children with branchValue and add them to the activeNodes-list
                    activeNew.child0 = Node(activeNew.depth+1, activeNew, 0)
                    activeNew.child1 = Node(activeNew.depth+1, activeNew, 1)
                    self.activeNodes.append(activeNew.child1)
                    self.activeNodes.append(activeNew.child0)
            else:
                activeNew.lowerb = np.inf
                self.updBound(activeNew)
            activeOld = activeNew
             
            
    
    def move(self, fromNode, toNode):
        """Moves problem from fromNode to toNode.
        """
        fix = []
        while fromNode.depth > toNode.depth:
            self.problem.unfixVariable(fromNode.branchVariable)
            fromNode = fromNode.parent
        
        while toNode.depth > fromNode.depth:
            fix.append( (toNode.branchVariable, toNode.branchValue) )
            toNode = toNode.parent
            
        while toNode is not fromNode:
            self.problem.unfixVariable(fromNode.branchVariable)
            fix.append(toNode.branchVariable)
            fromNode = fromNode.parent
            toNode = toNode.parent
            
        for var, value in fix:
            self.problem.fixVariable(var, value)
    
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
        """Specifies the variable to be branched.
        """
        for (i,x) in enumerate(vec):
            if x > 0+ self.eps and x < 1- self.esp:
                return i
        return -1
        
    def getActiveNode(self):
        """Gets one active node depending on searchrule from the activeNode-list.
        """
        if self.depthFirst:
            return self.activeNodes.pop()
        else:
            return self.activeNodes.popleft()
        
            
            
            
        

class Node:
    
    def __init__(self, depth, parent=None, branchValue = None):
        self.parent = parent
        self.child0 = None
        self.child1 = None
        self.branchVariable = None
        self.branchValue = branchValue
        self.depth = depth
        self.lowerb = -np.inf
        self.upperb = np.inf
        