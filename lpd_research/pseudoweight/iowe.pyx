#!/usr/bin/python2
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation


"""Methods for computing pseudoweight IOWEs for 3-D turbo codes."""

import numpy as np
import math
import itertools, collections, functools
import os

cimport numpy as np
cimport cython
from libc.math cimport log, exp, fmax, fabs, floor
cdef np.double_t log2 = log(2)
cdef np.double_t mininf = -np.inf

cdef class IOWE:
    
    cdef public int length, K
    cdef public np.ndarray itable, ftable
    
    def __init__(self, K, itable, ftable):
        cdef int i, w1, w2, q1, q2
        cdef double logn
        self.length = ftable.size
        self.itable = itable
        self.ftable = ftable
        self.K = K

    @staticmethod
    def fromFile(path, K):
        cdef:
            int i = 0
            np.ndarray[ndim=2,dtype=np.int_t] itable
            np.ndarray[ndim=1,dtype=np.double_t] ftable
        with open(path, "rt") as file:
            lines = file.readlines()
        length = len(lines)
        itable = np.empty((length, 4), dtype=np.int)
        ftable = np.empty(length, dtype=np.double)
        
        for line in lines:
            spl = line.split()
            itable[i,0] = int(spl[0])
            itable[i,1] = int(spl[1])
            itable[i,2] = int(spl[2])
            itable[i,3] = int(spl[3])
            ftable[i] = float(spl[4])
            i += 1
        return IOWE(K, itable, ftable)
    
    def write(self, path):
        cdef int i
        with open(path, 'wt') as outfile:
            for i in range(self.length):
                outfile.write("{} {} {} {} {:.6f}\n".format(self.itable[i,0], self.itable[i,1], self.itable[i,2], self.itable[i,3], self.ftable[i]))


cdef double logbinom(int n, int k):
    cdef double result = 0
    cdef int i
    for i in range(n-k+1, n+1):
        result += log(i)
    for i in range(1, k+1):
        result -= log(i)
    return result

cdef class BinomTable:
    
    cdef int minTop, maxTop, minBot, maxBot
    cdef np.double_t[:,:] values
    
    def __init__(self, int minTop, int maxTop, int minBot, int maxBot):
        cdef int i, j
        self.minTop = minTop
        self.maxTop = maxTop
        self.minBot = minBot
        self.maxBot = maxBot
        self.values = np.empty((maxTop-minTop+1, maxBot-minBot+1), dtype=np.double)
        for i in range(self.minTop, self.maxTop+1):
            for j in range(self.minBot, self.maxBot+1):
                self.values[i-self.minTop,j-self.minBot] = logbinom(i,j)
    
    @cython.wraparound(False)
    cdef double get(self, int top, int bot):
        return self.values[top-self.minTop, bot-self.minBot]
        

cdef np.double_t[:] correctionterms = np.empty(2000000)
def initlogplus():
    cdef int i
    for i in range(2000000):
        correctionterms[i] = log(1+exp(-i/100000.0))

cdef inline np.double_t logPlus(np.double_t v1, np.double_t v2):
    cdef double absdiff = fabs(v1-v2)
    if absdiff < 20:
        return fmax(v1, v2) + correctionterms[<int>(absdiff*100000)] #log(1 + exp(-fabs(v1-v2)))
    return fmax(v1, v2)


cdef inline np.double_t logtrinom(int K, np.int_t w1, np.int_t w2):
    cdef:
        np.double_t res = 0
        int i
    for i in range(K-w1-w2+1, K+1):
        res += log(i)
    for i in range(1, w1+1):
        res -= log(i)
    for i in range(1, w2+1):
        res -= log(i)
    return res

cdef class TrinomTable:
    
    cdef np.double_t[:,:] values
    
    def __init__(self, int top, int maxbot):
        cdef int i, j
        self.values = np.empty((maxbot+1, maxbot+1), dtype=np.double)
        for i in range(0, maxbot+1):
            for j in range(i, maxbot+1):
                self.values[i, j] = self.values[j, i] = logtrinom(top, i, j)
                
    @cython.wraparound(False)
    cdef double get(self, int w1, int w2):
        return self.values[w1, w2]
    

@cython.wraparound(False)
def concatenatedIOWE(IOWE inner, int MAXW=50):
    cdef:
        np.ndarray[ndim=2, dtype=np.int_t] out_ind_d = np.empty((0,4), dtype=np.int, order='C')
        np.ndarray[ndim=1, dtype=np.double_t] out_val_d = np.empty(0, dtype=np.double, order='C')
        np.int_t[:,:] out_ind
        np.double_t[:] out_val
        np.ndarray[ndim=2, dtype=np.int_t] itable = inner.itable
        np.ndarray[ndim=1, dtype=np.double_t] ftable = inner.ftable
        np.double_t[:] tmpout = mininf*np.ones((MAXW+1)*(MAXW+1), dtype=np.double)
        int i = 0, j, qb1, qb2, q1, q2, itableSize = itable.shape[0]
        int jLow = 0, jHigh = 1, tmpIndex, outSize = 0, numTotal = 0, numTmp = 0
        int currentW1 = 0, currentW2 = 0, w1 = 0, w2 = 0, qa1 = 0, qa2 = 0
        np.double_t newVal, oldVal, tmpVal, normalization = 0
        bint newBlock
    tmp = os.times()
    time = tmp[0] + tmp[2]
    while True:
        for j in range(jLow, jHigh):
            qb1 = itable[j,2]
            qb2 = itable[j,3]
            q1 = qa1 + qb1
            q2 = qa2 + qb2
            if q1 > MAXW or q2 > MAXW:
                continue
            newVal = ftable[i] + ftable[j]
            tmpIndex = (MAXW+1)*q1+q2
            oldVal = tmpout[tmpIndex]
            if oldVal == mininf:
                tmpout[tmpIndex] = newVal
                numTmp += 1
            else:
                tmpout[tmpIndex] = logPlus(oldVal, newVal)
        newBlock = False
        i += 1
        if i < inner.length:
            w1 = itable[i,0]
            w2 = itable[i,1]
            qa1 = itable[i,2]
            qa2 = itable[i,3]
            if w2 != currentW2 or w1 != currentW1:
                newBlock = True
        else:
            newBlock = True
        if newBlock:
            if numTmp > 0:
                # finished a block of common w1/w2 values
                out_ind_d.resize((outSize+numTmp,4), refcheck=False)
                out_val_d.resize(outSize+numTmp, refcheck=False)
                out_ind = out_ind_d
                out_val = out_val_d
                tmpIndex = outSize
                normalization = logtrinom(inner.K, currentW1, currentW2) + currentW1*log2
                for j in range(tmpout.size):
                    tmpVal = tmpout[j]
                    if tmpVal != mininf:
                        numTmp -= 1
                        q1 = j // (MAXW+1)
                        q2 = j % (MAXW+1)
                        out_ind[tmpIndex,0] = currentW1
                        out_ind[tmpIndex,1] = currentW2
                        out_ind[tmpIndex,2] = q1
                        out_ind[tmpIndex,3] = q2
                        out_val[tmpIndex] = tmpVal - normalization
                        tmpIndex += 1
                        if numTmp == 0:
                            break
                outSize = out_ind.shape[0]
            currentW1 = w1
            currentW2 = w2
            if i == inner.length:
                break
            for j in range(tmpout.size):
                tmpout[j] = mininf
            jLow = i
            for jHigh in range(i, itable.shape[0]):
                if itable[jHigh,1] != currentW2 or itable[jHigh,0] != currentW1:
                    break
    tmp = os.times()
    print(tmp[0] + tmp[2] - time)
    return IOWE(inner.K, out_ind_d, out_val_d)

@cython.wraparound(False)
cpdef completeWE(IOWE pcc, IOWE inner, outfile, int MAXW=50, bint codewords=False):
    cdef:
        np.double_t[:] out = mininf*np.ones((MAXW+1)**2, dtype=np.double)
        np.int_t[:,:] itable_pcc = pcc.itable
        np.int_t[:,:] itable_inner = inner.itable
        np.double_t[:] ftable_pcc = pcc.ftable
        np.double_t[:] ftable_inner = inner.ftable
        int i, j, index, n_out = 0
        int K = pcc.K, twoLambdaK = pcc.K//2, twoK = 2*pcc.K
        int w1, w2, h1, h2, n1, n2, q1, q2, currentN1 = -1, currentN2 = -1
        double newVal, oldVal, middleterm, middledenom = logbinom(twoK, twoLambdaK)
        BinomTable t1 = BinomTable(0, MAXW, 0, MAXW)
        BinomTable t2 = BinomTable(twoK-2*MAXW, twoK, twoLambdaK-2*MAXW, twoLambdaK)
        TrinomTable tt = TrinomTable(twoLambdaK, MAXW)
    tmp = os.times()
    time = tmp[0] + tmp[2]
    for i in range(pcc.length):
        if i % 1000 == 0:
            print('{:8d}/{}'.format(i, pcc.length))
        w1 = itable_pcc[i,0]
        if codewords and w1 > 0:
            continue
        w2 = itable_pcc[i,1]
        q1 = itable_pcc[i,2]
        if codewords and q1 > 0:
            continue
        q2 = itable_pcc[i,3]
        currentN1 = currentN2 = -1
        for j in range(inner.length):
            n1 = itable_inner[j,0]
            if codewords and n1 > 0:
                continue
            n2 = itable_inner[j,1]
            if codewords an n2 > 0:
                continue
            if n2 != currentN2 or n1 != currentN1:
                if n1 > q1 or n2 > q2:
                    continue
                if n1+n2 > twoLambdaK:
                    continue
                currentN1 = n1
                currentN2 = n2
                middleterm = t1.get(q1, n1) + t1.get(q2, n2) + t2.get(twoK-q1-q2, twoLambdaK-n1-n2) - tt.get(n1, n2) - n1*log2
            h2 = itable_inner[j,3] + w2 + q2 - n2
            if h2 < 0 or h2 > MAXW:
                continue
            h1 = itable_inner[j,2] + w1 + q1 - n1
            if h1 < 0 or h1 > MAXW:
                continue
            index = (MAXW+1)*h1+h2
            newVal = ftable_pcc[i] + ftable_inner[j] + middleterm
            oldVal = out[index]
            if oldVal == mininf:
                out[index] = newVal
                n_out += 1 
            else:
                out[index] = logPlus(oldVal, newVal)
    tmp = os.times()
    print('computation time: {}'.format(tmp[0] + tmp[2] - time))
    with open(outfile, 'wt') as f:
        j = 0
        for h1 in range(MAXW+1):
            for h2 in range(MAXW+1):
                if out[j] != mininf:
                    f.write('{} {} {:.6f}\n'.format(h1, h2, out[j] - middledenom))
                j += 1
        

def readWE(path):
    with open(path, 'rt') as infile:
        lines = infile.readlines()
    size = len(lines) - 1
    result = np.empty((size,2), dtype=np.double)
    for i, line in enumerate(lines[1:]):
        h1, h2, m = line.strip().split()
        h1 = int(h1)
        h2 = int(h2)
        result[i,0] = (h1 + 2*h2)**2/(h1+4*h2)
        result[i,1] = float(m)
    sortindices = np.argsort(result[:,0])
    return result[sortindices,:]

def makeBins(pseudos):
    out = []
    pw = 0
    enum = mininf
    for w, m in pseudos:
        if w != pw:
            out.append((pw, enum))
            pw = w
            enum = m
        else:
            enum = logPlus(enum, m)
    return np.array(out[1:])
        
            

def estimate(np.double_t[:,:] array, float threshold):
    sum = array[0,1]
    logthresh = log(threshold)
    i = 1
    while sum <= logthresh:
        sum = logPlus(sum, array[i,1])
        i += 1
    return array[i-1, 0]