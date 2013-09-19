#!/usr/bin/python2
import numpy as np
import sys
from lpdecoding.codes import linear
from lpdecoding.decoders.siegeldecoders import ZhangSeparationDecoder
from lpdecoding.algorithms.pseudoweight import pseudoWeight as pw
from pseudoweight import supportsearch
supportMatrix = np.array([
    [ (0, 1), (0, 1), (1, 1), (1, 1), (-11, 3), (-11, 3), (-8, 3),  (-8, 3), (-2, 3), (-2, 3), (2, 3), (2, 3),  (-4, 3), (-4, 3), (5, 3), (5, 3), (-3, 1), (-3, 1), (-2, 1), (-2, 1), (-1, 1), (-1, 1)],
    [ (0, 1), (1, 6), (3, 2), (5, 6), (-11, 6), (-2, 1) , (-11, 6), (-4, 3), (-1, 2), (1, 6),  (5, 6), (4, 3),  (0, 1),  (-2, 3), (4, 3), (3, 2), (-2, 1), (-4, 3), (-2, 3), (-1, 2), (0, 1),  (-1, 2)],
    [ (0, 1), (1, 3), (2, 1), (2, 3), (0, 1),   (-1, 3),  (-1, 1),  (0, 1),  (-1, 3), (1, 1),  (1, 1), (2, 1),  (4, 3),  (0, 1),  (1, 1), (4, 3), (-1, 1), (1, 3),  (2, 3),  (1, 1),  (1, 1),  (0, 1) ],
    [ (0, 1), (1, 2), (5, 2), (1, 2), (11, 6),  (4, 3),   (-1, 6),  (4, 3),  (-1, 6), (11, 6), (7, 6), (8, 3),  (8, 3),  (2, 3),  (2, 3), (7, 6), (0, 1),  (2, 1),  (2, 1),  (5, 2),  (2, 1),  (1, 2) ],
    [ (0, 1), (2, 3), (3, 1), (1, 3), (11, 3),  (3, 1),   (2, 3),   (8, 3),  (0, 1),  (8, 3),  (4, 3), (10, 3), (4, 1),  (4, 3),  (1, 3), (1, 1), (1, 1),  (11, 3), (10, 3), (4, 1),  (3, 1),  (1, 1) ],
    [ (0, 1), (5, 6), (7, 2), (1, 6), (11, 2),  (14, 3),  (3, 2),   (4, 1),  (1, 6),  (7, 2),  (3, 2), (4, 1),  (16, 3), (2, 1),  (0, 1), (5, 6), (2, 1),  (16, 3), (14, 3), (11, 2), (4, 1),  (3, 2) ]
], dtype=np.object)

def solveMod(ratio, q):
    a, b = ratio
    for x in range(q):
        if (b*x - a) % q == 0:
            return x
    raise ValueError("q is not prime")

def createSupportVector(code):
    q = code.q
    extendedMatrix = np.array([ [solveMod(ratio, q) for ratio in row] for row in supportMatrix ], dtype=np.int)
    submatrix = np.zeros((q*6, extendedMatrix.shape[1]), dtype=np.int)
    for i, row in enumerate(extendedMatrix):
        for j, value in enumerate(row):
            submatrix[q*i+value, j] = 1
            
    supportvec = np.zeros(code.blocklength, dtype=np.int)
    for i in range(submatrix.shape[1]):
        found=False
        column = submatrix[:, i]
        for j in range(code.parityCheckMatrix.shape[1]):
            if np.all(code.parityCheckMatrix[:,j] == column):
                supportvec[j] = 1
                found = True
                break
        if not found:
            print('wtf')
            print(column)
            print(i)
    return supportvec

def minimizeSupportIP(q, m):
    import cplex
    code = linear.ArrayLDPCCode(q, m)
    decoder = ZhangSeparationDecoder(code, pureLP=True, coneProjection=True)
    z = [ "z{}".format(i) for i in range(code.blocklength) ]
    decoder.cplex.set_problem_type(decoder.cplex.problem_type.MILP)
    decoder.cplex.variables.add(types=[decoder.cplex.variables.type.binary] * code.blocklength,
                                 names=z)
    decoder.cplex.objective.set_sense(decoder.cplex.objective.sense.minimize)
    decoder.cplex.linear_constraints.add(
              lin_expr=[cplex.SparsePair(ind=[z[i], decoder.x[i]], val=[code.blocklength, -1]) for i in range(code.blocklength)],
              names=['decision_{}'.format(i) for i in range(code.blocklength)],
              senses='G'*code.blocklength,
              rhs=np.zeros(code.blocklength))
    decoder.llrVector = np.zeros(code.blocklength)
    decoder.cplex.objective.set_linear(zip(z, np.ones(code.blocklength)))
    decoder.solve()
    print(decoder.solution)
    print(decoder.objectiveValue)
    
def searchWithSupportVector(q):
    code = linear.ArrayLDPCCode(q, 6)
    support = createSupportVector(code)
    decoder = ZhangSeparationDecoder(code, pureLP=True, coneProjection=True)
    for i in range(support.size):
        if support[i] == 0:
            decoder.cplex.variables.set_upper_bounds(decoder.x[i], 0)
    result = supportsearch.findLowSupportPCWs("test", decoder, code, 100)
    print(result.support)
    print(" ".join(str(x) for x in result.lowestSupportPCW))
    print(pw(result.lowestSupportPCW))
    
def searchRightmostZero(q, z):
    code = linear.ArrayLDPCCode(q, 6)
    decoder = ZhangSeparationDecoder(code, pureLP=True, coneProjection=True)
    for i in range(q):
        zeroPositions = list(range(q*(i+1) - z, q*(i+1)))
        print(zeroPositions)
        #zeroPositions = np.random.randint(0, 2, code.blocklength)
        if len(zeroPositions) > 0:
            decoder.cplex.variables.set_upper_bounds([ (decoder.x[j], 0) for j in zeroPositions ])
    result = supportsearch.findLowSupportPCWs("test", decoder, code, 100)
    print(result.support)
    print(" ".join(str(x) for x in result.lowestSupportPCW))
    print(pw(result.lowestSupportPCW))

if __name__ == "__main__":
    #minimizeSupportIP(*(int(arg) for arg in sys.argv[1:]))
    searchRightmostZero(*(int(arg) for arg in sys.argv[1:]))

def eea(u, v):
    u1 = 1
    u2 = 0
    u3 = u
    v1 = 0
    v2 = 1
    v3 = v
    while v3 != 0:
        q = u3 / v3
        t1 = u1 - q * v1
        t2 = u2 - q * v2
        t3 = u3 - q * v3
        u1 = v1
        u2 = v2
        u3 = v3
        v1 = t1
        v2 = t2
        v3 = t3
    return u1, u2, u3
