#!/usr/bin/python2
# coding: utf-8
import numpy as np
import logging

logger = logging.getLogger("simplex")
EPS = 1e-16
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
    
    def pivot(row, col):
        # make pivot operation with pivot element Tred[row,col]
        pivotElem = Tred[row,col]
        logger.info("pivoting with [{},{}]={}".format(row, col, pivotElem))
        for ii in xrange(m+1):
            if ii != row:
                if np.abs(Tred[ii, col]) > EPS:
                    Tred[ii, :] -= (Tred[ii, col]/pivotElem)*Tred[row,:]
        Tred[row,:] /= pivotElem
        #Tred[row,col] = 1 # fix numerical errors
        # maybe swap these steps
    
    for iteration in itertools.count():
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
                case = -1
                delta = np.inf
                min_row = 0
                if found == 0:
                    # Situation 1 or 2
                    logger.debug("situation 1/2 with j={}, j_ind={} (cj_bar={})".format(j, j_ind, cj_bar))
                    for i in xrange(1,m+1):
                        if Tred[i,m+1] > EPS: # relevant for delta_2
                            quotient = Tred[i,m]/Tred[i,m+1]
                            if quotient < delta or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                delta = quotient
                                min_row = i
                                case = 1
                        elif B[i-1] < n-m:
                            if Tred[i,m+1] < -EPS: # relevant for delta_3
                                quotient = (Tred[i,m]-1)/Tred[i,m+1]
                                if quotient < delta or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                    delta = quotient
                                    min_row = i
                                    case = 2
                            else:
                                Tred[i,m+1] = 0
                    if j < n-m and delta >= 1:
                        delta = 1
                        case = 0
                    assert delta < 1e10
                    assert case > -1
                    # case = np.argmin((delta_1, delta_2, delta_3))
                    #logger.debug("deltas={},{},{}".format(delta_1, delta_2, delta_3))
                    #logger.debug("min_rows={},{}".format(min_row_d2, min_row_d3))
                    if case == 0:
                        #assert delta_1 == 1
                        logger.debug("case delta_1")
                        # case (3): L->U
                        Tred[:,m] -= Tred[:,m+1] # update \tilde b
                        LU[j_ind] = 1 # move j from L to U
                    elif case == 1:
                        # normal basis exchange: case (1)
                        logger.debug("case delta_2")
                        #logger.debug("Tred[i,m+1]={}".format(Tred[min_row_d2,m+1]))
                        #logger.debug("Tred[i,m]={}".format(Tred[min_row_d2,m]))
                        #assert min_row_d2 > 0
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        pivot(min_row, m+1)
                    else:
                        # case (5): L -> B, B-> U
                        logger.debug("case delta_3")
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        LU[j_ind] = 1
                        Tred[min_row, m] -= 1
                        if np.abs(Tred[min_row, m]) < EPS:
                            Tred[min_row, m] = 0
                        pivot(min_row, m+1)
                else:
                    assert found == 1
                    logger.debug("situation 3 with j={}, j_ind={} (cj_bar={})".format(j, j_ind, cj_bar))
                    #delta_1 = 1
                    for i in xrange(1, m+1):
                        if Tred[i,m+1] < -EPS:
                            quotient = -Tred[i,m]/Tred[i,m+1]
                            if quotient < delta or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                delta = quotient
                                min_row = i
                                case = 1
                        elif B[i-1] < n-m:
                            if Tred[i,m+1] > EPS:
                                quotient = (1 - Tred[i,m])/Tred[i,m+1]
                                if quotient < delta or (quotient < delta+EPS and B[i-1] < B[min_row-1]):
                                    delta = quotient
                                    min_row = i
                                    case = 2
                            else:
                                Tred[i,m+1] = 0
                    if delta >= 1:
                        delta = 1
                        case = 0
                    assert delta < 1e10
                    #case = np.argmin((delta_1, delta_2, delta_3))
                    #logger.debug("deltas={},{},{}".format(delta_1, delta_2, delta_3))
                    #logger.debug("min_rows={},{}".format(min_row_d2, min_row_d3))
                    if case == 0:
                        logger.debug("case delta_1")
                        # case (4): U -> L
                        Tred[:,m] += Tred[:,m+1] # update \tilde b
                        LU[j_ind] = 0
                    elif case == 1:
                        # case (6): U -> B, B-> L
                        logger.info("case delta_2")
                        logger.info(Tred[:,m+1])
                        logger.info(Tred[:,m])
                        assert min_row > 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        LU[j_ind] = 0
                        #Tred[:,m] += Tred[:,m+1]
                        pivot(min_row, m+1)
                        Tred[min_row, m] += 1
                    else:
                        logger.info("case delta_3")
                        assert min_row > 0
                        #Tred[:,m] += Tred[:,m+1]
                        Tred[min_row, m] -= 1
                        if np.abs(Tred[min_row, m]) < EPS:
                            Tred[min_row, m] = 0
                        N[j_ind] = B[min_row-1]
                        B[min_row-1] = j
                        pivot(min_row, m+1)
                        Tred[min_row, m] += 1
                break
        if found == -1:
            logger.debug('no reduced costs in iteration {}'.format(iteration))
            x = np.zeros(n-m, dtype=np.double)
            for r_ind, r in enumerate(B):
                if r < n-m:
                    x[r] = Tred[r_ind+1,m]
            for r_ind, r in enumerate(N):
                if r < n-m and LU[r_ind] == 1:
                    x[r] = 1
            return -Tred[0,m], x 
        
        

if __name__ == "__main__":
    A = np.array( [ [1, 1],
                    [-1, 1]])
    b = np.array( [1.5, 0.5])
    c = np.array( [-2, -1] )
    print(primal01SimplexRevised(A,b,c))
    
