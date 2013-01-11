# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation



from branchbound.bnb cimport Node

cdef class dequeNode:
    
    def __init__(self, Node pre=None, Node post=None, Node element=None):
        self.pre = pre
        self.element = element
        self.post = post

class MyIndexError(Exception):  
    pass

cdef class myDeque:
    
    def __init__(self, Node start=None):
        startNode = dequeNode(None, None, start)
        self.first = startNode
        self.last = startNode
        if start is not None:
            self.length = 1
        else:
            self.length = 0
            
    cdef Node popleft(self):
        cdef:
            Node returnValue
        if self.length > 0:
            returnValue = self.first.element
            if self.length > 1:
                self.first = self.first.post
                self.length += -1
            else:
                self.first = None
                self.last = None
                self.length = 0
            return returnValue
        else:
            raise MyIndexError()
    
    cdef Node pop(self):
        cdef:
            Node returnValue
        if not self.length == 0:
            returnValue = self.last.element
            if self.length > 1:
                self.last = self.last.pre
                self.length += -1
            else:
                self.first = None
                self.last = None
                self.length = 0
            return returnValue
        else:
            raise MyIndexError()
        
    cdef void append(self, Node nextNode):
        self.last.post = nextNode
        self.last = dequeNode(self.last, None, nextNode)
        self.length += 1

            
    