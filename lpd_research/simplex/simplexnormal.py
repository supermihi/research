#!/usr/bin/python2
# coding: utf-8

from __future__ import division
import numpy as np
import logging

logger = logging.getLogger("simplex")
EPS = 1e-10
degenEps = 1 #1/3
import itertools


def primalSimplexRevised(A, b, c, fixed=False):
    """Apply revised primal simplex to the problem min cx s.t. Ax <= b, x >= 0.

    """

    # construct the full tableau (introducing slacks)
    # c 0 0
    # A I b
    #
    # and the reduced tableau initialized as
    # 0 0
    # I b
    if fixed:
        import fixedpoint as fp
        global EPS
        EPS = fp.fixed2float(3)
    A = np.hstack( (A, np.eye(A.shape[0])) )
    m, n = A.shape
    print("c<norm>={}".format(c))
    c = np.hstack( (c, np.zeros(m)) )
    B = np.arange(n-m, n) # basics
    N = np.arange(n-m) # nonbasics
    Tred = np.zeros( (m+1, m+2), dtype=np.double)
    Tred[1:,:m] = np.eye(m)
    Tred[1:,m] = b
    if fixed:
        Tred = fp.np_float2fixed(Tred)
        A = fp.np_float2fixed(A)
        c = fp.np_float2fixed(c)
        b = fp.np_float2fixed(b)
        print("A<fp>={}".format(A))
        print("c<fp>={}".format(c))
    ki = np.zeros(m+1, dtype=np.int) # Wolfe's ad hoc procedure

    def pivot(row, col, K=0):
        # make pivot operation with pivot element Tred[row,col]
        
        pivotElem = Tred[row,col]
        logger.info("pivoting with [{},{}]={}".format(row, col, pivotElem))
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
    
    for iteration in itertools.count():
        # ad hoc procedure, step (1)
        for i in range(1,m+1):
            if np.abs(Tred[i,m]) < EPS:
                ki[i] += 1
                print("INCREASING k[{}] to {} (0)".format(i, ki[i]))
                if fixed:
                    Tred[i,m] = fp.FixedPointNumber(np.random.randint(2,4))
                else:
                    Tred[i,m] = np.random.randint(2,17)
        logger.debug('iteration {}'.format(iteration))
        logger.debug('objective value z={}'.format(Tred[0,m]))
        
        found = -1
        # optimality test:
        # - set found = 0 if nonbasic at lb with neg reduced costs is found
        # - set found = 1 if nonbasic at ub with pos reduced costs is found
        for j_ind, j in enumerate(N):
            cj_bar = c[j] + np.dot(Tred[0,:m], A[:,j])
            if c[j] < 0:
                print("c[j]={}".format(c[j]))
                print("Tred[0,:m]={}".format(Tred[0,:m]))
                print("A[:,j]={}".format(A[:,j]))
                print("cj_bar={}".format(cj_bar))
            if cj_bar < -EPS:
                print("cj_bar {} < -EPS".format(cj_bar))
                found = 1
                # compute augmented column
                Tred[1:,m+1] = np.dot(Tred[1:, :m], A[:, j])
                Tred[0,m+1] = cj_bar
                logger.debug("B={}".format(B))
                logger.debug("N={}".format(N))
                logger.debug("j={}".format(j))
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
                        logger.debug("K DECREASE")
                        assert K > 0
                        for i in np.flatnonzero(ki == K):
                            Tred[i,m] = fp.FixedPointNumber(0) if fixed else 0
                            ki[i] -= 1
                        K -= 1
                        logger.debug("ki={}".format(ki))                    
                assert delta < np.inf
                logger.debug("delta={}".format(delta))
                assert min_row > 0
                N[j_ind] = B[min_row-1]
                B[min_row-1] = j
                logger.debug("b={}".format(Tred[1:,m]))
                pivot(min_row, m+1, K)
                logger.debug("b={}".format(Tred[1:,m]))
                logger.debug("ki = {}".format(ki))
                break
        if found == -1:
            logger.debug('no reduced costs in iteration {}'.format(iteration))
            logger.debug("final ki={}".format(ki))
            logger.debug("final B={}".format(B))
            logger.debug("final b={}".format(Tred[1:,m]))
            logger.debug("{} iterations".format(iteration))
            for i in np.flatnonzero(ki > 0):
                Tred[i, m] = fp.FixedPointNumber(0) if fixed else 0
            x = np.zeros(n-m, dtype=np.double)
            for r_ind, r in enumerate(B):
                if r < n-m:
                    x[r] = Tred[r_ind+1,m]
                    print(x[r], Tred[r_ind+1,m])
            logger.debug("x={}".format(x))
            return -Tred[0,m], x 
        
        

if __name__ == "__main__":
    A = np.array( [ [1, 1],
                    [-1, 1]])
    b = np.array( [1.5, 0.5])
    c = np.array( [-2, -1] )
    print(primal01SimplexRevised(A,b,c))
    
