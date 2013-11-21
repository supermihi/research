# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

cimport numpy as np
import numpy as np
from libc.math cimport fmin
cdef double inf = np.inf
cdef class Node:
    
    def __init__(self, **kwargs):
        self.parent = kwargs.get("parent", None)
        self.branchIndex = kwargs.get("branchIndex", -1)
        self.branchValue = kwargs.get("branchValue", -1)
        self.lb = -inf
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.lbChild = [-inf, -inf]
        
    cpdef updateBound(self, double lbChild, int childValue):
        cdef double newLb
        if lbChild > self.lbChild[childValue]:
            self.lbChild[childValue] = lbChild
        newLb = fmin(self.lbChild[0], self.lbChild[1])
        if newLb > self.lb:
            self.lb = newLb
            if self.parent is not None:
                self.parent.updateBound(newLb, self.branchValue)