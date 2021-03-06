#!/usr/bin/python2
# coding: utf-8

from __future__ import division
import numpy as np
import logging

logger = logging.getLogger("simplex")
EPS = 1e-10
degenEps = 1 #1/3
import itertools


def primalSimplexRevised(A, b, c, fixed=None):
    """Apply revised primal simplex to the problem min cx s.t. Ax <= b, x >= 0.

    """

    # construct the full tableau (introducing slacks)
    # c 0 0
    # A I b
    #
    # and the reduced tableau initialized as
    # 0 0
    # I b
    stats = {}
    if fixed:
        import fixedpoint as fp
        global EPS
        fp.setPrecision(*fixed)
        EPS = fp.fixed2float(2**(fixed[1]//3+1)-1)
    else:
        EPS = 1e-10
    A = np.hstack( (A, np.eye(A.shape[0])) )
    m, n = A.shape
    c = np.hstack( (c, np.zeros(m)) )
    B = np.arange(n-m, n) # basics
    N = np.arange(n-m) # nonbasics
    Tred = np.zeros( (m+1, m+2), dtype=np.double)
    Tred[1:,:m] = np.eye(m)
    Tred[1:,m] = b
    maxAbsEntry = 0 # ab 2. zeile
    minAbsEntry = 1 # ganzes Tableau
    maxNonzeroEntries = 0 # ganze tableau
    if fixed:
        Tred = fp.np_float2fixed(Tred)
        A = fp.np_float2fixed(A)
        maxFp = 2**(fixed[0]-fixed[1]-1)-2**-fixed[1]
        if np.max(np.abs(c)) > maxFp:
            print("scaling function by {}".format(2/3*maxFp/np.max(np.abs(c))))
            c *= 1/2*maxFp/np.max(np.abs(c))                
        c = fp.np_float2fixed(c)
        b = fp.np_float2fixed(b)
    ki = np.zeros(m+1, dtype=np.int) # Wolfe's ad hoc procedure

    def pivot(row, col, K=0):
        # make pivot operation with pivot element Tred[row,col]
        
        pivotElem = Tred[row,col]
        for i in xrange(m+1):
            if i != row:
                if np.abs(Tred[i, col]) > EPS:
                    Tred[i,:m] -= Tred[row,:m]*(Tred[i, col]/pivotElem)
                    if K == 0 or (i > 0 and ki[i] == K):
                        Tred[i,m] -= Tred[row,m]*(Tred[i, col]/pivotElem)
                    Tred[i,m+1] -= Tred[row,m+1]*(Tred[i, col]/pivotElem)
                else:
                    Tred[i, col] = fp.FixedPointNumber(0) if fixed else 0
        Tred[row,:] /= pivotElem
        Tred[row,col] = fp.FixedPointNumber(1) if fixed else 1
        #Tred[row,col] = 1 # fix numerical errors
        # maybe swap these steps
    
    for iteration in itertools.count(1):
        # update statistics
        nmax =  np.max(np.abs(Tred[1:,:]))
        if nmax > maxAbsEntry:
            maxAbsEntry = nmax
        nmin = np.min(np.abs(Tred[np.nonzero(Tred)]))
        if nmin < minAbsEntry:
            minAbsEntry = nmin
        nnonzero = np.sum(Tred!=0)/Tred.size
        if nnonzero > maxNonzeroEntries:
            maxNonzeroEntries = nnonzero 
        # ad hoc procedure, step (1)
        for i in range(1,m+1):
            if np.abs(Tred[i,m]) < EPS:
                ki[i] += 1
                if fixed:
                    Tred[i,m] = fp.FixedPointNumber(fpValue=np.random.randint(3,15))
                else:
                    Tred[i,m] = np.random.randint(2,17)/16
        
        found = -1
        # optimality test:
        # - set found = 0 if nonbasic at lb with neg reduced costs is found
        # - set found = 1 if nonbasic at ub with pos reduced costs is found
        for j_ind, j in enumerate(N):
            if iteration > 1000:
                break
            cj_bar = c[j] + np.dot(Tred[0,:m], A[:,j])
            if cj_bar < -EPS:
                found = 1
                # compute augmented column
                Tred[1:,m+1] = np.dot(Tred[1:, :m], A[:, j])
                Tred[0,m+1] = cj_bar
                delta = np.inf
                K = np.max(ki)
                min_row = 0
                while delta == np.inf:
                    for i in np.flatnonzero(ki == K):
                        if i == 0:
                            continue
                        if Tred[i,m+1] > EPS:
                            quotient = Tred[i,m]/Tred[i,m+1]
                            if quotient < delta:
                                delta = quotient
                                min_row = i
                    if delta == np.inf:
                        if K == 0:
                            found = -1
                            break
                        for i in np.flatnonzero(ki == K):
                            Tred[i,m] = fp.FixedPointNumber(0) if fixed else 0
                            ki[i] -= 1
                        K -= 1                    
                if delta == np.inf:
                    found = -1
                    break
                assert min_row > 0
                N[j_ind] = B[min_row-1]
                B[min_row-1] = j
                pivot(min_row, m+1, K)
                break
        if found == -1:
            logger.debug("{} iterations".format(iteration))
            stats["iterations"] = iteration
            stats["maxnonzeros"] = maxNonzeroEntries
            stats["maxabs"] = maxAbsEntry
            stats["minabs"] = minAbsEntry
            for i in np.flatnonzero(ki > 0):
                Tred[i, m] = fp.FixedPointNumber(0) if fixed else 0
            x = np.zeros(n-m, dtype=np.double)
            for r_ind, r in enumerate(B):
                if r < n-m:
                    if fixed:
                        x[r] = 0 if Tred[r_ind+1,m] < 0.5 else 1
                    else:
                        x[r] = Tred[r_ind+1,m]
            return -Tred[0,m], x, stats
        
        

if __name__ == "__main__":
    A = np.array( [ [1, 1],
                    [-1, 1]])
    b = np.array( [1.5, 0.5])
    c = np.array( [-2, -1] )
    print(primal01SimplexRevised(A,b,c))
    
