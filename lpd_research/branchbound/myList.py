# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation



from branchbound.bnb import Node

class dequeNode:
    
    def __init__(self, pre=None, post=None, element=None):
        self.pre = pre
        self.element = element
        self.post = post

class MyIndexError(Exception):  
    pass

class myDeque:
    
    def __init__(self, start=None):
        start = dequeNode(None, None, start)
        self.first = start
        self.last = start
        if start is not None:
            self.length = 1
        else:
            self.length = 0
            
    def popleft(self):
        if not self.length == 0:
            returnValue = self.first
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
    
    def pop(self):
        if not self.length == 0:
            returnValue = self.last
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
        
    def append(self, nextNode=None):
        self.last.post = nextNode
        myNextNode = dequeNode(self.last, None, nextNode)
        self.last = myNextNode
        self.length += 1
            
    