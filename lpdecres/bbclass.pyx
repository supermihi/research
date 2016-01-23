# -*- coding: utf-8 -*-
# cython: language_level=3

import itertools, functools
import fractions
import numpy as np
cimport numpy as np
from lpdec.codes.nonbinary import flanaganEmbedding

cdef class BuildingBlockClass:

    """Building block class for non-binary LP decoding valid inequalities."""

    def __init__(self, shifts):
        if isinstance(shifts, str):
            shifts = list(map(int, shifts))
        q = len(shifts)
        self.q = q
        self.shifts = list(shifts)
        self.vals = np.empty((q, q), dtype=np.int)
        self.computeVals()

    def computeVals(self):
        for i in range(self.q):
            self.vals[0, i] = i + self.q*self.shifts[i]
        self.sigma = np.argmax(self.vals[0, :])
        for j in range(1, self.q):
            for i in range(self.q):
                self.vals[j, i] = self.vals[0, (i + j) % self.q] - self.vals[0, j]

    def isValid(self):
        """Check if this building block class induces valid inequalities."""
        # qRange = np.arange(1, self.q)
        # for loVals in itertools.product((0,1), repeat=self.q-1):
        #     maskModQ = np.dot(qRange, loVals) % self.q
        #     if maskModQ != 0:
        #         rhs = -np.dot(loVals, self.vals[self.lo, 1:])
        #         lhs = self.vals[0, self.q - maskModQ]
        #         if lhs > rhs:
        #             #print(mask, maskModQ, lhs, rhs)
        #             return False
        # return True
        import gurobimh as g
        model = g.Model()
        model.setParam('OutputFlag', 0)
        zVars = [model.addVar(vtype=g.GRB.INTEGER, ub=self.vals[0, self.sigma], name='y{}'.format(i))
                 for i in range(1, self.q)]
        xVars = [model.addVar(vtype=g.GRB.BINARY, name='x{}'.format(i)) for i in range(1, self.q)]
        modQVar = model.addVar(vtype=g.GRB.INTEGER, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='z')
        model.update()
        model.addConstr(g.LinExpr(self.vals[self.sigma, 1:], zVars) + g.LinExpr(self.vals[0, 1:], xVars) >= 1)
        model.addConstr(g.quicksum(xVars) == 1)
        qRange = list(range(1, self.q))
        model.addConstr(g.LinExpr(qRange, zVars) + g.LinExpr(qRange, xVars) + self.q*modQVar == 0)
        model.optimize()
        if model.Status == g.GRB.INFEASIBLE:
            return True
        else:
            # print('invalid y/x:')
            # print([x.X for x in xVars])
            # print([y.X for y in zVars])
            return False

    def isSymmetric(self):
        q = self.q
        sortedVals = sorted(self.vals[0, :])
        maxVal = sortedVals[-1]
        if q == 2:
            return True
        if maxVal % 2 == 1:
            return False
        for i in range(q//2):
            if sortedVals[i] != maxVal - sortedVals[q-i-1]:
                #print(i, sortedVals[i], sortedVals[q-i-1])
                return False
        if sortedVals[q//2] != maxVal // 2:
            #print(':(')
            return False
        return True

    def isDoublySymmetric(self):
        if self.q == 2:
            return False
        Wproj = sorted(self.vals[0, :])[1:-1]
        maxVal = Wproj[-1] // 2
        cand = []
        for i in Wproj:
            if i <= maxVal:
                cand.append(i)
            else:
                break
        if len(cand) < (self.q - 3) // 2:
            return False
        for size in range((self.q - 3) // 2, len(cand) + 1):
            for subset in itertools.combinations(cand, size):
                isDS = True
                for i in subset:
                    if Wproj[-1] - i not in Wproj:
                        isDS = False
                        break
                if isDS:
                    #print('DS: {}'.format(c))
                    # print('vals={}'.format(self.vals[0,:]))
                    # print('Wproj={}'.format(Wproj))
                    # print('subset={}'.format(subset))
                    return True


    def dominatesSingle(self, other):
        assert other.q == self.q
        return np.all(self.vals[0, :] >= other.vals[0, :]) and \
               np.all(self.vals[self.sigma, :] >= other.vals[other.sigma, :]) and \
               self.shifts != other.shifts

    def isDominatedBy(self, others):
        """Strong dominance check using linear programming"""
        if len(others) == 0:
            return False
        import gurobimh as g
        model = g.Model()
        model.setParam('OutputFlag', 0)
        vars = [model.addVar(lb=0) for _ in range(3*len(others)*(self.q-1))]
        model.update()
        Ahi = np.empty((self.q - 1, len(others)*(self.q-1)))
        Alo = np.empty((self.q - 1, len(others)*(self.q-1)))

        for i, cls in enumerate(others):
            for automorphism in range(1, self.q):
                for j in range(1, self.q):
                    Ahi[j-1, i*(self.q-1) + automorphism - 1] = cls.vals[0, automorphism*j % self.q]
                    Alo[j-1, i*(self.q-1) + automorphism - 1] = cls.vals[cls.sigma, automorphism*j % self.q]

        # b = np.hstack((self.vals[0, 1:], self.vals[self.lo, 1:], self.vals[self.lo, 1:]))
        # A = np.array(np.bmat([[Ahi, Alo, Alo], [Alo, Ahi, Alo], [Alo, Alo, Ahi]]))
        # test simpler
        b = np.hstack((self.vals[0, 1:], self.vals[self.sigma, 1:]))
        A = np.vstack((Ahi, Alo))
        # print('A=',A)
        # print('b=',b)
        for i, row in enumerate(A):
            model.addConstr(g.LinExpr(row, vars) >= b[i])
        model.update()
        model.optimize()
        # if model.status != g.GRB.INFEASIBLE:
        #     print(self.vals)
        #     print([o.shifts for o in others])
        #     print([v.X for v in vars])
        return model.Status != g.GRB.INFEASIBLE

    def embedConstant(self, zeta):
        ret = [0] * self.q
        ret[zeta] = 1
        return ret

    def embedFlanagan(self, zeta):
        ret = [0] * (self.q - 1)
        if zeta != 0:
            ret[zeta-1] = 1
        return ret

    def hasBadGcd(self):
        return functools.reduce(fractions.gcd, self.vals[0]) != 1

    def rankOfTightCodewords(self, d=3):
        nCols = d*(self.q-1)
        nRows = (d-1)*(self.q-1)
        cws = []
        for firsts in itertools.product(list(range(self.q)), repeat=d-1):
            last = -sum(firsts) % self.q
            if last == 0:
                continue
            # if np.count_nonzero(firsts) <= 1:
            #     continue
            lhsChange = sum(self.vals[self.sigma, f] for f in firsts) + self.vals[0, last]
            if lhsChange == 0:
                line = []
                for f in firsts:
                    line.extend(self.embedFlanagan(f))
                line.extend(self.embedFlanagan(last))
                cws.append(line)
        if len(cws) == 0:
            return 0
        M = np.array(cws, dtype=np.double)
        return np.linalg.matrix_rank(M)

    def testRank(self):
        cws = []
        q = self.q
        sigma = self.sigma
        for i in range(1, q):
            cws.append([i, 0, q-i])
            cws.append([0, i, q-i])
        for j in range(1, q):
            if j == sigma:
                continue
            cws.append([-j, j-sigma, sigma])
        for j in range(2, q-1):
            if self.shifts[j] == 1:
                continue
            for i in range(2, j+1):
                if self.shifts[i] == 1:
                    continue
                if i + j == q:
                    continue
                if (i+j < q and self.shifts[i+j] == 0) or (i+j >= q and self.shifts[i+j-q] == 1):
                    cws.append([-i, -j, (i + j) % q])
        mat = np.array([flanaganEmbedding(cw, q) for cw in cws])
        return np.linalg.matrix_rank(mat)

    def hasFullRank(self, d=3):
        rank = self.rankOfTightCodewords(d)
        if rank == d*(self.q-1) - 1:
            return True
        else:
            #print('rank=', rank)
            return False


    @classmethod
    def validFacetDefining(cls, int q, d=3):
        """Return all valid facet-defining unique classes for given `q`."""
        classes = []
        for shifts in sorted(itertools.product((0,1), repeat=q-1), key=sum):
            if shifts == (1,0) * ((q-1)//2) or q == 2 and shifts == (1,):
                continue  # redundant class 0101010...
            c = BuildingBlockClass((0,) + shifts)
            if c.isSymmetric() and c.isValid() and c.hasFullRank(d):
                classes.append(c)
        return classes


    def __str__(self):
        return str(self.shifts)
