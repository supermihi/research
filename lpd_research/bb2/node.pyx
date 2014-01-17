# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function
cimport numpy as np
import numpy as np
from libc.math cimport fmin
cdef double inf = np.inf
cdef class Node:
    
    def __init__(self, **kwargs):
        self.parent = kwargs.get("parent", None)
        self.branchIndex = kwargs.get("branchIndex", -1)
        self.branchValue = kwargs.get("branchValue", -1)
        if self.parent is not None:
            self.depth = self.parent.depth + 1
            self.lb = self.parent.lb
        else:
            self.depth = 0
            self.lb = -inf
        self.lbChild0 = self.lbChild1 = -inf
        self.flag = False
        
    cpdef updateBound(self, double lbChild, int childValue):
        cdef double newLb, oldChild = self.lbChild0 if childValue == 0 else self.lbChild1
        if lbChild > oldChild:
            if childValue == 0:
                self.lbChild0 = lbChild
            else:
                self.lbChild1 = lbChild
        newLb = fmin(self.lbChild0, self.lbChild1)
        if newLb > self.lb:
            self.lb = newLb
            if self.parent is not None:
                self.parent.updateBound(newLb, self.branchValue)
                
    def printFixes(self):
        cdef Node node = self
        while node is not None:
            print('x{}={}, '.format(node.branchIndex, node.branchValue), end='')
            node = node.parent
        print()