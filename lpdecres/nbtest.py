__author__ = 'michael'

from lpdec.codes.nonbinary import *
import numpy as np

if __name__ == '__main__':
    q = 3
    cws = []
    for i in range(1, q):
        cws.append(binaryEmbedding([i, 0, q-i], q))
        cws.append(binaryEmbedding([i, q-i, 0], q))
    for i in range(1, (q-1)//2+1):
        cws.append(binaryEmbedding([i, i, q-2*i], q))
        print(binaryEmbedding([i, i, q-2*i], q))
    for i in range((q-1)//2, q):
        cws.append(binaryEmbedding([0, i, q-i], q))
    mat = np.array(cws)
    print(mat)
    print(np.linalg.matrix_rank(mat))