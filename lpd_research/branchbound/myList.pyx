# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling, PHilipp Reichling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation



from branchbound.bnb cimport Node

cdef class dequeNode:
    
    def __init__(self, dequeNode pre=None, dequeNode post=None, Node element=None):
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
        #if self.length > 0:
        #returnValue = self.first.element
        if self.length == 0:
            return None
        else:
#            print("i am here")
            returnValue = self.first.element
#            print("here i go again with length: {}".format(self.length))
            if self.length > 1:
#                print("blub")
                self.first = self.first.post
#                print("do i still blubber?")
                self.length += -1
            else:
                self.first = None
                self.last = None
                self.length = 0
            return returnValue
#        else:
#            return None
#    
    cdef Node pop(self):
        cdef:
            Node returnValue
        if self.length == 0:
            return None
        else:
            returnValue = self.last.element
            if self.length > 1:
                self.last = self.last.pre
                self.length += -1
            else:
                self.first = None
                self.last = None
                self.length = 0
            return returnValue
        #else:
        #    return None
        
    cdef void append(self, Node nextNode):
#        print("append function used")
#        print("self.last: {}".format(self.last))
        nextDequeNode = dequeNode(self.last, None, nextNode)
#       print("i got a nextDequeNode")
#       print("length: {}".format(self.length))
        if self.length == 0:
            self.first = nextDequeNode
            self.last = nextDequeNode
        else:
#            print("i am here")
            self.last.post = nextDequeNode
#            print("i got so far")
            self.last = nextDequeNode
#            print("almost there")
        self.length += 1
#        print("append function successfull")
            
    