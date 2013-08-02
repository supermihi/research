# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

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
            