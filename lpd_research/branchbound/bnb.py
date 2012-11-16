# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import print_function
import numpy as np
import logging

from lpdecoding.utils import StopWatch

class BranchAndBound:
    
    def __init__(self, problem, selectionMethod, branchRule, eps=1e-6): 
        self.problem = problem
        self.eps = eps
        self.root = Node()
        #the method used to branch at Nodes
        self.selectionMethode = selectionMethod(self.root, self.problem)
        self.branchRule = branchRule(self.problem)
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
        timer = StopWatch()
        timer.start()
        while True:
            logging.debug('main loop iteration {}'.format(branchCount))
            logging.debug('lb={}, ub={}'.format(self.root.lowerb, self.root.upperb))
            logging.debug('#active nodes: {}\n'.format(len(self.selectionMethode.activeNodes)))
            #select one of the active nodes, move there and (solve the corresponding problem)
            try:
                (activeNew, fixC, unfixC) = self.selectionMethode.getActiveNode(activeOld)
            except NodesExhausted:
                break
            
            #(fixC, unfixC) = self.bMethod.move(activeOld, activeNew)
            fixCount += fixC
            unfixCount += unfixC
            moveCount += 1
            if activeNew.solution is not None:
                logging.debug('activeNew solution: {}'.format(activeNew.solution))
                #find the Variable to be branched in this node
                branchVariable = self.branchRule.selectVariable(activeNew.solution)
                logging.debug('branchVariable: {}'.format(branchVariable))
                #update bounds of all nodes if neccesary
                activeNew.lowerb = activeNew.objectiveValue

                #hier eventuell heuristik für upperbound einfügen
                if branchVariable is None:
                    # have a feasible solution
                    if self.optimalObjectiveValue > activeNew.objectiveValue:
                        if self.optimalSolution is None:
                            logging.info("first feasible solution after {} steps".format(branchCount))
                        # found new global optimum
                        self.optimalSolution = activeNew.solution
                        self.optimalObjectiveValue = activeNew.objectiveValue
                    logging.debug('objectiveValue: {}'.format(activeNew.objectiveValue))
                    activeNew.upperb = activeNew.objectiveValue
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
                    self.selectionMethode.createNodes(branchVariable, activeNew)
                    #activeNew.child0 = Node(activeNew, branchVariable, 0)
                    #activeNew.child1 = Node(activeNew, branchVariable, 1)
                    if np.random.randint(0, 2) == 0:
                        self.selectionMethode.addNodes(activeNew.child0,activeNew.child1)
                    else:
                        self.selectionMethode.addNodes(activeNew.child1, activeNew.child0)
                    branchCount += 1
            else:
                activeNew.lowerb = np.inf
                self.updBound(activeNew)
            activeOld = activeNew
        
        #match fix and unfix count to the selected branchMethod
        from .nodeselection import BBSMethod, DSTMethod
        if isinstance(self.selectionMethode, BBSMethod) or isinstance(self.selectionMethode, DSTMethod):
            fixCount += 2*branchCount
            unfixCount += 2*branchCount
            
        self.moveCount = moveCount
        self.fixCount = fixCount
        self.unfixCount = unfixCount
        self.branchCount = branchCount
        self.time = timer.stop()
        logging.debug("******* optimal solution found *******")
        logging.debug(self.optimalSolution)
        logging.debug(self.optimalObjectiveValue)
        logging.debug("BranchCount: {count}; FixCount: {fix}, UnfixCount: {unfix}".format(count=branchCount, fix=fixCount, unfix=unfixCount))
        logging.debug("MoveCount: {move}".format(move=moveCount))
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

class NodesExhausted(Exception):
    pass


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
    
    
