# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from lpdecoding.core import Decoder
from lpdecoding.codes import trellis
from lpdecoding.algorithms.path import shortestPathScalarization

class DantzigWolfeTurboDecoder(Decoder):
    
    def __init__(self, code, name=None):
        Decoder.__init__(self, code)
        if name is None:
            name = "DantzigWolfeTurboDecoder"
        self.name = name
        self.pairs = self.code.prepareConstraintsData()
        self.k = len(self.pairs)
        self.m = self.k+1
        self.segAndLabForCodeBit = {}
        for i in range(code.blocklength):
            for seg, lab in code.segmentsForCodeBit(i):
                self.segAndLabForCodeBit[(seg,lab)] = i
    
    def fixConstraintValue(self, j, val=0):
        labelStr = { trellis.INFO : "info", trellis.PARITY: "parity"}
        (t1, s1, b1), (t2, s2, b2) = self.pairs[j]
        val1 = self.llrVector[self.segAndLabForCodeBit[(s1,b1)]]
        val2 = self.llrVector[self.segAndLabForCodeBit[(s2,b2)]]
        assert val1 == val2
        if val == 0:
            if val1 + val2 > 0:
                bit1 = bit2 = 0
            else:
                bit1 = bit2 = 1
        elif val == 1:
            bit1 = 1
            bit2 = 0
        else:
            bit1 = bit2 = -1
        setattr(s1, "fix_{}".format(labelStr[b1]), bit1)
        setattr(s2, "fix_{}".format(labelStr[b2]), bit2)
        
    def solve(self, hint=None, lb=1):
        # find starting basis
        B = np.zeros( (self.m, self.m) )
        B[0,:] = 1
        lamb = 1
        mu = np.zeros(self.k)
        codeword = np.empty(self.code.blocklength)
        for j in range(self.k+1):
            g_result = np.zeros(self.k)
            if j > 0:
                self.fixConstraintValue(j-1, 1)
            for jj in range(j, self.k+1):
                self.fixConstraintValue(jj-1, 0)
            for enc in self.code.encoders:
                shortestPathScalarization(enc.trellis, lamb, mu, g_result, codeword)
            B[j, 1:] = g_result
            if j > 0:
                self.fixConstraintValue(j, -1)

def transpositionMatrix(n, i, j):
    ret = np.eye(n)
    ret[i,i] = ret[j,j] = 0
    ret[i,j] = ret[j,i] = 1
    return ret

def LU(orig):
    mat = np.array(orig).copy()
    L = np.eye(mat.shape[0])
    print(L)
    P = np.eye(mat.shape[0])
    for i in range(mat.shape[0]):
        print('i={}'.format(i))
        pivot = np.argmax(np.abs(mat[i:,i])) + i
        if pivot != i:
            print('swapping rows {} and {}'.format(pivot, i))
            # swap rows
            tmp = mat[pivot].copy()
            mat[pivot] = mat[i]
            mat[i] = tmp
            tmp = P[pivot].copy()
            P[pivot] = P[i]
            P[i] = tmp
            print(mat)
            tmp = L[i,:i].copy()
            L[i,:i] = L[pivot,:i]
            L[pivot,:i] = tmp
        for k in range(i+1,mat.shape[0]):
            factor = -mat[k,i]/mat[i,i]
            L[k,i] = -factor
            print('factor: {}'.format(factor))
            print('before: {}'.format(mat[k]))
            mat[k] += factor*mat[i]
            print('after: {}'.format(mat[k]))
        print(mat)
    print(L)
    print(np.dot(L, mat))
    print(np.dot(P, orig))
    return mat
            