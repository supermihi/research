# -*- coding: utf-8 -*-
# Copyright 2012 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
import numpy as np

class BranchAndBound:
    
    def __init__(self, problem, eps=1e-6):
        self.problem = problem
        self.eps = eps
        self.root = Node(depth=0)
        self.activeNodes = [ self.root ]
    
    def run(self):
        pass
    
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
    
    def acBound(self, node):
        """Updates lower and upper bounds for node and all parent nodes if possible
        """
        pass
    
    def findVariable(self):
        """Specifies the variable to be branched.
        """
        pass

class Node:
    
    def __init__(self, depth, parent=None):
        self.parent = parent
        self.child0 = None
        self.child1 = None
        self.branchVariable = None
        self.branchValue = None
        self.depth = depth
        self.lowerb = -np.inf
        self.upperb = np.inf
        