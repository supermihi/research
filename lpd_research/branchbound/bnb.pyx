# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, Philipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import print_function
from libc.math cimport fmin
import numpy as np
import logging
from lpdecoding.utils import StopWatch, stopwatch

cdef class BranchAndBound:
    
    def __init__(self, problem, selectionMethod, branchRule, eps=1e-6): 
        self.problem = problem
        self.eps = eps
        self.root = Node()
        #the method used to branch at Nodes
        self.selectionMethod = selectionMethod(self.root, self.problem)
        self.branchRule = branchRule(self.problem)
        #self.activeNodes = deque([self.root])
        self.optimalSolution = None
        self.optimalObjectiveValue = np.inf
        self.refreshTime = 0
        self.getTime = 0
        self.addTime = 0
        self.createTime = 0
        self.moveTime = 0
        self.selectionTime = 0
        self.boundTime = 0
        #self.getActiveNode = selectionMethod
    
    def run(self):
        """Builds tree and runs a branch and bound algorithm.
        """
        cdef:
            Node activeNew, activeOld = self.root
        branchCount = 0
        timer = StopWatch()
        timer.start()
        while True:
            logging.debug('main loop iteration {}'.format(branchCount))
            logging.debug('lb={}, ub={}'.format(self.root.lowerb, self.root.upperb))
            #logging.debug('#active nodes: {}\n'.format(len(self.selectionMethod.activeNodes)))
            #select one of the active nodes, move there and (solve the corresponding problem)
            #try:
            with stopwatch() as getTimer:
                activeNew = self.selectionMethod.getActiveNode(activeOld)
            self.getTime += getTimer.duration
            #except NodesExhausted:
            if not isinstance(activeNew, Node):
                #print("i shouldnt be here")
                break

            if activeNew.solution is not None:
                logging.debug('activeNew solution: {}'.format(activeNew.solution))
                #find the Variable to be branched in this node
                with stopwatch() as selectionTimer:
                    branchVariable = self.branchRule.selectVariable(activeNew.solution)
                self.selectionTime += selectionTimer.duration
                #print("branchVariable: {}".format(branchVariable))
                logging.debug('branchVariable: {}'.format(branchVariable))
                #update bounds of all nodes if neccesary
                activeNew.lowerb = activeNew.objectiveValue

                #hier eventuell heuristik für upperbound einfügen
                if branchVariable is None:
                    # have a feasible solution
                    if self.optimalObjectiveValue > activeNew.objectiveValue:
                        if self.optimalSolution is None:
                            logging.info("first feasible solution after {} steps".format(branchCount))
                            self.selectionMethod.firstSolutionExists = True
                            with stopwatch() as refreshTimer:
                                self.selectionMethod.refreshActiveNodes(activeNew)
                            self.refreshTime += refreshTimer.duration
                        # found new global optimum
                        self.optimalSolution = activeNew.solution
                        self.optimalObjectiveValue = activeNew.objectiveValue
                    logging.debug('objectiveValue: {}'.format(activeNew.objectiveValue))
                    activeNew.upperb = activeNew.objectiveValue
                with stopwatch() as boundTimer:
                    self.updBound(activeNew)
                self.boundTime += boundTimer.duration
                
                #create new children or close branch
                if activeNew.lowerb > self.root.upperb:
                    pass
                elif abs(activeNew.lowerb - activeNew.upperb) < self.eps:
                    pass
                elif branchVariable is None:
                    pass
                else:
                    #create children with branchValue and add them to the activeNodes-list
                    with stopwatch() as createTimer:
                        self.selectionMethod.createNodes(branchVariable, activeNew)
                    self.createTime += createTimer.duration
                    with stopwatch() as addTimer:
                        self.selectionMethod.addNodes(activeNew.child0, activeNew.child1)
                    self.addTime += addTimer.duration
                    branchCount += 1
            else:
                activeNew.lowerb = np.inf
                self.updBound(activeNew)
            activeOld = activeNew
           
        self.moveCount = self.selectionMethod.moveCount
        self.fixCount = self.selectionMethod.fixCount
        self.unfixCount = self.selectionMethod.unfixCount
        self.branchCount = branchCount
        self.time = timer.stop()
        self.lpTime = self.selectionMethod.lpTime
        self.moveTime = self.selectionMethod.moveTime 
        #to measure the time used for solving lps / functions in percent
        print(self.time)
        print(self.lpTime)
        self.lpVsAll = self.lpTime / self.time if self.time > 0 else 0
        self.boundVsAll = self.boundTime / self.time if self.time > 0 else 0
        self.refreshVsAll = self.refreshTime / self.time if self.time > 0 else 0
        self.getVsAll = self.getTime / self.time if self.time > 0 else 0
        self.addVsAll = self.addTime / self.time if self.time > 0 else 0
        self.createVsAll = self.createTime / self.time if self.time > 0 else 0
        self.moveVsAll = self.moveTime / self.time if self.time > 0 else 0
        self.selectionVsAll = self.selectionTime / self.time if self.time > 0 else 0
        #self.lpVsAll = self.lpTime / (self.time + self.lpTime)
        logging.debug("******* optimal solution found *******")
        logging.debug(self.optimalSolution)
        logging.debug(self.optimalObjectiveValue)
        logging.debug("BranchCount: {count}; FixCount: {fix}, UnfixCount: {unfix}"\
                      .format(count=self.branchCount, fix=self.fixCount, unfix=self.unfixCount))
        logging.debug("MoveCount: {move}".format(move=self.moveCount))
        return self.optimalSolution
             
            
    

    cdef void updBound(self, Node node):
        """Updates lower and upper bounds for node and all parent nodes, if possible.
        """
        cdef:
            double ub, lb, ubp, lbp, ubb, lbb, upper, lower
        if node.parent is None:
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

class NodesExhausted(Exception):
    pass


cdef class Node:
    
    def __init__(self, Node parent=None, int branchVariable=-1, int branchValue=-1):
        assert branchVariable == -1 or branchVariable >= 0
        self.parent = parent
        self.child0 = None
        self.child1 = None
        #nec to use BestBound
        self.solution = None
        self.objectiveValue = np.inf
        
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
    
    
