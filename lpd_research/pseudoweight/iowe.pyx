#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation


"""Methods for computing pseudoweight IOWEs for 3-D turbo codes."""

import numpy as np
import math
import itertools
import collections
import functools
cimport numpy as np

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

@memoized
def multinom(K, w1, w2):
    """Return the log of the multinomial (K over w1,w2)"""
    return logFac(K) - logFac(w1) - logFac(w2) - logFac(K-w1-w2)

@memoized
def logFac(k):
    """Return an approximation of log(k!) by stirlings formula"""
    if k == 0:
        return 0
    return k*math.log(k) + 0.5*math.log(2*math.pi*k)-k

log2 = math.log(2)


cdef class IOWE:
    
    cdef public int length
    cdef public np.ndarray itable, ftable
    cdef public int maxW1, maxW2, maxQ1, maxQ2
    
    def __init__(self, table):
        cdef int i, w1, w2, q1, q2
        cdef double logn
        self.length = len(table)
        self.itable = np.empty((self.length, 4), dtype=np.int)
        self.ftable = np.empty(self.length, dtype=np.double)
        self.maxW1 = 0
        self.maxW2 = 0
        self.maxQ1 = 0
        self.maxQ2 = 0
        for i, (w1, w2, q1, q2, logn) in enumerate(table):
            self.itable[i,:] = (w1, w2, q1, q2)
            self.ftable[i] = logn
            if w1 > self.maxW1:
                self.maxW1 = w1
            if w2 > self.maxW2:
                self.maxW2 = w2
            if q1 > self.maxQ1:
                self.maxQ1 = q1
            if q2 > self.maxQ2:
                self.maxQ2 = q2


def readIOWE(path):
    table = []
    with open(path, "rt") as file:
        for line in file:
            spl = line.split()
            w1, w2, q1, q2 = (int(i) for i in spl[:-1])
            logn = float(spl[-1])
            table.append( (w1,w2,q1,q2,logn) )
    return IOWE(table[:13077])    


def computeOuterIOWE(IOWE inner):
    out = collections.OrderedDict()
    cdef np.int_t currentW1, currentW2, jLow, jHigh, i, j, k, numTotal
    cdef np.int_t w1, w2, qa1, qa2, qb1, qb2, q1, q2
    cdef double logn
    cdef np.int_t[:,:] itableOut
    cdef np.double_t[:] ftableOut
    currentW1 = -1
    currentW2 = -1
    jLow = jHigh = 0
    numTotal = 0
    for i in range(inner.length):
        w1, w2, qa1, qa2 = inner.itable[i,:]
        if w2 != currentW2 or w1 != currentW1:
            currentW1 = w1
            currentW2 = w2
            out[(w1,w2)] = outw1w2 = {}
            print("w1={},w2={}".format(w1, w2))
            jLow = i
            try:
                for j in itertools.count(i):
                    if inner.itable[j,1] != currentW2 or inner.itable[j,0] != currentW1:
                        jHigh = j
                        break
            except IndexError:
                jHigh = j
            print('jLow={}, jHigh={}'.format(jLow, jHigh))
        for j in range(jLow, jHigh):
            qb1 = inner.itable[j,2]
            qb2 = inner.itable[j,3]
            q1 = qa1 + qb1
            q2 = qa2 + qb2
            if (q1, q2) not in outw1w2:
                outw1w2[(q1,q2)] = inner.ftable[i] + inner.ftable[j]
                numTotal += 1
            else:
                old = outw1w2[(q1,q2)]
                outw1w2[(q1,q2)] = max(old, inner.ftable[j]+inner.ftable[i])\
                     + math.log(1+math.exp(-abs(old-inner.ftable[j]-inner.ftable[i])))
    itableOut = np.empty((numTotal, 4), dtype=np.int)
    ftableOut = np.empty(numTotal, dtype=np.double)
    k = 0
    for (w1, w2), outw1w2 in out.items():
        for (q1, q2), logn in sorted(outw1w2.items()):
            itableOut[k,0] = w1
            itableOut[k,1] = w2
            itableOut[k,2] = q1
            itableOut[k,3] = q2
            ftableOut[k] = logn #- multinom(100, w1, w2) #- w1*log2 
            k += 1
    return itableOut, ftableOut
