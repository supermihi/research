__author__ = 'michael'

from lpdec.codes.nonbinary import *
import numpy as np
from lpdecres.nonbinary import BuildingBlockClass
if __name__ == '__main__':

    for shifts in [0,1,1,0,0,1,0], [0,1,0,1,0,0,0], [0,1,0,0,0,0,0], [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0]:
        bb = BuildingBlockClass(len(shifts), shifts)
        print('sigma={}'.format(bb.sigma))
        q = bb.p
        sigma = bb.sigma
        cws = []
        for i in range(1, q):
            cws.append([i, 0, q-i])
            cws.append([0, i, q-i])
        for j in range(2, q):
            if j == sigma:
                continue
            cws.append([-j, j-sigma, sigma])
        countA = countB = 0
        for j in range(2, q-2):

            for i in range(2, min(q-j, j+1)):
                assert i + j < q
                if bb.m[i] == 0 and bb.m[j] == 0 and bb.m[i+j] == 0:
                    cws.append([-i, -j, i+j])
                    print('(',i,j,')', end=',')
                    countA +=1
                # if bb.shifts[i] + bb.shifts[j] == 1 and bb.shifts[i+j] == 1:
                #     cws.append([-i, -j, i+j])
            for i in range(q-j, q-2):
                if bb.m[i] == bb.m[j] == 0 and bb.m[i+j-q] == 1:
                    cws.append([-i, -j, i+j])
                    print('(',i,j,')', end=',')
                    countB += 1
                pass
        print(countA, countB)
        print()
        mat = np.array([flanaganEmbedding(cw, q) for cw in cws])

        #print(mat)
        rank = np.linalg.matrix_rank(mat)
        print(rank)
        #print(rank == (q-1)*3-1)