# -*- coding: utf-8 -*-
from __future__ import division, print_function

import itertools

import numpy as np

from lpdecoding import matrix
from lpdecoding.core import Decoder
from lpdecoding.codes import trellis
from lpdecoding.algorithms.path import shortestPathScalarization
from lpdecoding.codes.turbolike import LTETurboCode


def FTran(L, U, P, b):
    """Solves Ax = b for x by FTran operation where PA=LU."""
    k = L.shape[0]
    y = np.empty(k)
    b = np.dot(P, b)
    for i in range(k):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i]))/L[i,i]
    x = np.empty(k)
    for i in range(k-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:]))/U[i,i]
    return x

def BTran(L, U, P, b):
    """Solves A^Tx + b for x by BTran operation where PA=LU."""
    k = L.shape[0]
    y = np.empty(k)
    for i in range(k):
        y[i] = (b[i] - np.dot(U[:i,i], y[:i]))/U[i,i]
    x = np.empty(k)
    for i in range(k-1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:,i], x[i+1:]))/L[i,i]
    return np.dot(P.T, x)


class DantzigWolfeTurboDecoder(Decoder):
    
    def __init__(self, code, name=None):
        Decoder.__init__(self, code)
        if name is None:
            name = self.__class__.__name__
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
        val1 = self.llrVector[self.segAndLabForCodeBit[(t1[s1],b1)]]
        val2 = self.llrVector[self.segAndLabForCodeBit[(t2[s2],b2)]]
        if val == 0:
            if val1 + val2 > 0:
                bit1 = bit2 = 0
            else:
                bit1 = bit2 = 1
        elif val == 1:
            if np.random.randint(0, 2) == 0:
                bit1 = 1
                bit2 = 0
            else:
                bit2 = 1
                bit1 = 0
        else:
            bit1 = bit2 = -1
        setattr(t1[s1], "fix_{}".format(labelStr[b1]), bit1)
        setattr(t2[s2], "fix_{}".format(labelStr[b2]), bit2)

    
    def generateStartBasis(self):
        B = np.empty( (self.m, self.m) )
        c = np.zeros(self.m)
        B[0,:] = 1
        codewords = [np.zeros(self.code.blocklength) for _ in range(self.m)]
        for j in range(self.k):
            self.fixConstraintValue(j, 0)
        for j in range(self.m):
            if j > 0:
                self.fixConstraintValue(j-1, 1)
            g_result = np.zeros(self.k)
            for enc in self.code.encoders:
                c[j] += shortestPathScalarization(enc.trellis, 1, np.zeros(self.k), g_result, codewords[j])
            B[1:, j] = g_result
            if j > 0:
                self.fixConstraintValue(j-1, -1)
        return B, c, codewords 
    
    
    def solve(self, hint=None, lb=1):
        self.code.setCost(self.llrVector)
        B, c, codewords = self.generateStartBasis()
        L, U, P = LU(B)
        b = np.zeros(self.m)
        b[0] = 1
        x = FTran(L, U, P, b)
        z = np.dot(c, x)
        ki = np.zeros(self.m)
        K = 0
        # init done
        for iteration in itertools.count():
            for i in range(self.m):
                if x[i] < 1e-8:
                    ki[i] += 1
                    x[i] = np.random.randint(2,256)
                    if ki[i] > K:
                        K = ki[i]
            print('iteration {}'.format(iteration))
            L, U, P = LU(B)
            pi = BTran(L, U, P, c)
            ans = self.pricing(L, U, P, pi)
            if ans is None:
                self.objectiveValue = z
                for i in np.flatnonzero(ki > 0):
                    x[i] = 0
                self.solution = np.around(np.dot(x, codewords), 5)
                break
            Aj, cj_bar, cj, codeword = ans
            Ajbar = FTran(L, U, P, Aj)
            delta = np.inf
            delta_ind = -1
            while delta == np.inf:
                for j in np.flatnonzero(ki == K):
                    if Ajbar[j] > 1e-8:
                        if x[j] / Ajbar[j] < delta:
                            delta = x[j] / Ajbar[j]
                            delta_ind = j
                if delta == np.inf:
                    assert K > 0
                    for i in np.flatnonzero(ki == K):
                        x[i] = 0
                        ki[i] -= 1
                    K -= 1
            assert delta < np.inf
            if delta < 1e-8:
                print('D')
            for i in range(self.m):
                if ki[i] == K:
                    x[i] -= delta*Ajbar[i]
                    if np.abs(x[i]) < 1e-10:
                        x[i] = 0
            #x -= delta*Ajbar
            x[delta_ind] = delta
            if not np.all(x >= 0):
                print('x', x)
                return
            if K == 0:
                z += delta*cj_bar
            c[delta_ind] = cj
            B[:, delta_ind] = Aj
            codewords[delta_ind] = codeword
            #assert np.allclose(x, np.dot(np.linalg.inv(B), b))
            #print('basis exchange in index {}'.format(delta_ind))
            
              
            
    def pricing(self, L, U, P, pi):
        g_result = np.zeros(self.k)
        codeword = np.zeros(self.code.blocklength)
        c_orig = 0
        for enc in self.code.encoders:
            c_orig += shortestPathScalarization(enc.trellis, 1, -pi[1:], g_result, codeword)
        Aj = np.concatenate( ([1], g_result) )
        reducedCost = c_orig - np.dot(pi, Aj)
        #print("reduced cost: {}".format(reducedCost))
        if reducedCost < -1e-6:
            return Aj, reducedCost, c_orig, codeword
        return None
            
        

def transpositionMatrix(n, i, j):
    ret = np.eye(n)
    ret[i,i] = ret[j,j] = 0
    ret[i,j] = ret[j,i] = 1
    return ret

def LU(orig):
    mat = np.array(orig).copy()
    L = np.eye(mat.shape[0])
    P = np.eye(mat.shape[0])
    for i in range(mat.shape[0]):
        pivot = np.argmax(np.abs(mat[i:,i])) + i
        if pivot != i:
            # swap rows
            tmp = mat[pivot].copy()
            mat[pivot] = mat[i]
            mat[i] = tmp
            tmp = P[pivot].copy()
            P[pivot] = P[i]
            P[i] = tmp
            tmp = L[i,:i].copy()
            L[i,:i] = L[pivot,:i]
            L[pivot,:i] = tmp
        for k in range(i+1,mat.shape[0]):
            factor = -mat[k,i]/mat[i,i]
            L[k,i] = -factor
            mat[k] += factor*mat[i]
    return L, mat, P

if __name__ == "__main__":
    from lpdecoding.codes.interleaver import Interleaver
    import random
    random.seed(34)
    np.random.seed(2)
    interleaver = Interleaver.random(10)
    from lpdecoding.codes.convolutional import LTEEncoder
    from lpdecoding.codes.turbolike import StandardTurboCode
    from lpdecoding.channels import *
    from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
    #code = StandardTurboCode(LTEEncoder(), interleaver, "testcode")
    code = LTETurboCode(40)
    channel = AWGNC(coderate=code.rate, snr=1, seed=121)
    sig = SignalGenerator(code, channel, randomCodewords=True, wordSeed=247)
    llr = next(sig)
    print(llr)
    decoder = DantzigWolfeTurboDecoder(code)
    decoder.decode(llr)
    decoder2 = CplexTurboLikeDecoder(code, ip=False)
    decoder2.decode(llr)
    print(decoder2.objectiveValue, decoder.objectiveValue)
    print(decoder2.solution, decoder.solution)