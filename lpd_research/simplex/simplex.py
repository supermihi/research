#!/usr/bin/python2
# coding: utf-8

from __future__ import division
import numpy as np
import logging

logger = logging.getLogger("simplex")
EPS = 1e-10
degenEps = 1e-2 #1/3
import itertools


def primal01SimplexRevised(A, b, c):
    """Apply revised primal simplex to the problem min cx s.t. Ax <= b, 0<=x<=1.

    """

    # construct the full tableau (introducing slacks)
    # c 0 0
    # A I b
    #
    # and the reduced tableau initialized as
    # 0 0
    # I b
    A = np.hstack( (A, np.eye(A.shape[0])) )
    m, n = A.shape
    c = np.hstack( (c, np.zeros(m)) )
    B = np.arange(n-m, n) # basics
    N = np.arange(n-m) # nonbasics
    LU = np.zeros(n-m, dtype=np.int) # LU[j] == 1 <=> N[j]'th nonbasic is at upper bound
    Tred = np.zeros( (m+1, m+2), dtype=np.double )
    Tred[1:,:m] = np.eye(m)
    Tred[1:,m] = b
    ki = np.zeros(m+1, dtype=np.int)
    kiLU = np.zeros(m+1, dtype=np.int)
    K = 0
    def pivot(row, col, K=0):
        # make pivot operation with pivot element Tred[row,col]
        pivotElem = Tred[row,col]
        logger.info("pivoting with [{},{}]={}".format(row, col, pivotElem))
        for i in xrange(m+1):
            if i != row:
                if np.abs(Tred[i, col]) > EPS:
                    Tred[i,:m] -= (Tred[i, col]/pivotElem)*Tred[row,:m]
                    if K == 0 or (i > 0 and ki[i] == K):
                        Tred[i,m] -= (Tred[i, col]/pivotElem)*Tred[row,m]
                    Tred[i,m+1] -= (Tred[i, col]/pivotElem)*Tred[row,m+1]
                else:
                    Tred[i,col] = 0
        Tred[row,:] /= pivotElem
        Tred[row,col] = 1 # fix numerical errors
        # maybe swap these steps
    
    def clearKi():
        global K
        for i in np.flatnonzero(ki):
            Tred[i,m] = 0 if kiLU[i] == 0 else 1
            ki[i] = 0
        K = 0
    def reduceKi(i):
        Tred[i, m] = 0 if Tred[i,m] < .5 or B[i-1] >= n-m else 1 # kiLU[i] == 0 else 1
        ki[i] -= 1
    
    for iteration in itertools.count():
        # ad hoc procedure, step (1)
        for i in range(1, m+1):
            if np.abs(Tred[i,m]) < EPS:
                #assert kiLU[i] == 0
                ki[i] += 1
                kiLU[i] = 0
                logger.debug("INCREASING k[{}] to {} (0)".format(i, ki[i]))
                Tred[i,m] = np.random.random()*degenEps
            elif B[i-1] < n-m and Tred[i,m] > 1 - EPS:
                ki[i] += 1
                kiLU[i] = 1
                Tred[i,m] = 1-np.random.random()*degenEps
                logger.debug("INCREASING k[{}] to {} (1)".format(i, ki[i]))
        K = np.max(ki)
        logger.debug('iteration {}'.format(iteration))
        logger.debug('objective value z={}'.format(Tred[0,m]))
        
        found = -1
        # optimality test:
        # - set found = 0 if nonbasic at lb with neg reduced costs is found
        # - set found = 1 if nonbasic at ub with pos reduced costs is found
        for j_ind, j in enumerate(N):
            cj_bar = c[j] + np.dot(Tred[0,:m], A[:,j])
            if cj_bar < -EPS and LU[j_ind] == 0:
                found = 0
            elif cj_bar > EPS and LU[j_ind] == 1:
                found = 1
            if found >= 0:
                # compute augmented column
                Tred[1:,m+1] = np.dot(Tred[1:, :m], A[:, j])
                Tred[0,m+1] = cj_bar
                logger.debug("B={}".format(B))
                logger.debug("N={}".format(N))
                logger.debug("LU={}".format(LU))
                logger.debug("j={}".format(j))
                case = -1
                delta = np.inf
                min_row = 0
                if found == 0:
                    # Situation 1 or 2
                    logger.debug("situation 1/2 with j={}, j_ind={} (cj_bar={})".format(j, j_ind, cj_bar))
                    while case == -1:
                        logger.debug("K={}".format(K))
                        #for i in xrange(1,m+1):
                        logger.debug("nonzero ki={}".format(np.flatnonzero(ki == K)))
                        for i in np.flatnonzero(ki == K):
                            if i == 0:
                                continue
                            if Tred[i,m+1] > EPS: # relevant for delta_2
                                quotient = Tred[i,m]/Tred[i,m+1]
                                if quotient < delta: # BLAND or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                    delta = quotient
                                    min_row = i
                                    case = 1
                            elif B[i-1] < n-m:
                                if Tred[i,m+1] < -EPS: # relevant for delta_3
                                    quotient = (Tred[i,m]-1)/Tred[i,m+1]
                                    if quotient < delta: # BLAND or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                        delta = quotient
                                        min_row = i
                                        case = 2
                        if case == -1:
                            logger.debug("K DECREASE")
                            logger.debug("ki={}".format(ki))
                            assert K > 0
                            for i in np.flatnonzero(ki == K):
                                reduceKi(i)
                            K -= 1
                            logger.debug("ki={}".format(ki))
                        elif j < n-m and delta >= 1:
                            delta = 1
                            case = 0
                            logger.debug("LU")
                            raw_input()
                        
                    assert delta < np.inf
                    assert case > -1
                    logger.debug("delta={}".format(delta))
                    if case == 0:
                        #assert delta_1 == 1
                        logger.debug("case delta_1(a)")
                        # case (3): L->U
                        Tred[:,m] -= Tred[:,m+1] # update \tilde b
                        assert Tred[1:,m] > -EPS
                        LU[j_ind] = 1 # move j from L to U
                    elif case == 1:
                        # normal basis exchange: case (1)
                        logger.debug("case delta_2(a)")
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        pivot(min_row, m+1, K)
                    else:
                        # case (5): L -> B, B-> U
                        logger.debug("case delta_3(a)")
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        LU[j_ind] = 1
                        print(ki[min_row])
                        if ki[min_row] > 0:
                            #ki[min_row] -= 1
                            Tred[min_row, m] = 0 if Tred[min_row,m] < .5 else 1
                            K = np.max(ki)
                        Tred[min_row, m] -= 1
                        print(Tred[min_row,m], ki[min_row])
                        pivot(min_row, m+1, K)
                        print(Tred[min_row,m])
                        print(Tred[1:,m])
                        print(ki)
                        #ki[min_row] -= 1
                        raw_input()
                else:
                    assert found == 1
                    logger.debug("situation 3 with j={}, j_ind={} (cj_bar={})".format(j, j_ind, cj_bar))
                    #delta_1 = 1
                    while case == -1:
                        logger.debug("K={}".format(K))
                        # for i in xrange(1, m+1):
                        for i in np.flatnonzero(ki == K):
                            if i == 0:
                                continue
                            if Tred[i, m+1] < -EPS:
                                quotient = -Tred[i,m]/Tred[i,m+1]
                                if quotient < delta: # BLAND or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                    delta = quotient
                                    min_row = i
                                    case = 1
                            elif B[i-1] < n-m:
                                if Tred[i,m+1] > EPS:
                                    quotient = (1 - Tred[i,m])/Tred[i,m+1]
                                    if quotient < delta: # BLAND or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                        delta = quotient
                                        min_row = i
                                        case = 2
                        if case == -1:
                            logger.debug("K DECREASE")
                            logger.debug("ki={}".format(ki))
                            assert K > 0
                            for i in np.flatnonzero(ki == K):
                                reduceKi(i)
                            K -= 1
                            logger.debug("ki={}".format(ki))
                        elif delta >= 1:
                            delta = 1
                            case = 0
                            logger.debug("UL")
                            raw_input()
                        
                    assert delta < np.inf
                    if case == 0:
                        logger.debug("case delta_1(b)")
                        # case (4): U -> L
                        Tred[:,m] += Tred[:,m+1] # update \tilde b
                        LU[j_ind] = 0
                        assert Tred[1:,m] > -EPS
                    elif case == 1:
                        # case (6): U -> B, B-> L
                        logger.info("case delta_2(b)")
                        logger.info(Tred[:,m+1])
                        logger.info(Tred[:,m])
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        LU[j_ind] = 0
                        #Tred[:,m] += Tred[:,m+1]
                        pivot(min_row, m+1, K)
                        Tred[min_row, m] += 1
                    else:
                        logger.info("case delta_3(b)")
                        assert min_row > 0
                        #Tred[:,m] += Tred[:,m+1]
                        Tred[min_row, m] -= 1
                        #if np.abs(Tred[min_row, m]) < EPS:
                        #    Tred[min_row, m] = 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        pivot(min_row, m+1, K)
                        Tred[min_row, m] += 1
                logger.debug("b = {}".format(Tred[:,m]))
                logger.debug("ki = {}".format(ki))
                break
        if found == -1:
            logger.debug('no reduced costs in iteration {}'.format(iteration))
            logger.debug("final ki={}".format(ki))
            for i in np.flatnonzero(ki > 0):
                reduceKi(i)
            x = np.zeros(n-m, dtype=np.double)
            for r_ind, r in enumerate(B):
                if r < n-m:
                    x[r] = Tred[r_ind+1,m]
            for r_ind, r in enumerate(N):
                if r < n-m and LU[r_ind] == 1:
                    x[r] = 1
            logger.debug("x={}".format(x))
            return -Tred[0,m], x 
        
        

if __name__ == "__main__":
    A = np.array( [ [1, 1],
                    [-1, 1]])
    b = np.array( [1.5, 0.5])
    c = np.array( [-2, -1] )
    print(primal01SimplexRevised(A,b,c))
    
