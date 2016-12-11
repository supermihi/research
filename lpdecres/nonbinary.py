# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import itertools, functools
import fractions
import numpy as np
import sys
from lpdec.codes.nonbinary import flanaganEmbedding


class BuildingBlockClass:

    """Building block class for non-binary LP decoding valid inequalities."""

    def __init__(self, m):
        if isinstance(m, str):
            m = list(map(int, m))
        p = len(m)
        self.p = p
        self.m = m[:]
        self.vals = np.empty((p, p), dtype=np.int)
        self.computeVals()

    def computeVals(self):
        for i in range(self.p):
            self.vals[0, i] = i + self.p * self.m[i]
        self.sigma = np.argmax(self.vals[0, :])
        self.max = np.max(self.vals[0, :])
        for j in range(1, self.p):
            for i in range(self.p):
                self.vals[j, i] = self.vals[0, (i + j) % self.p] - self.vals[0, j]

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
                 for i in range(1, self.p)]
        xVars = [model.addVar(vtype=g.GRB.BINARY, name='x{}'.format(i)) for i in range(1, self.p)]
        modQVar = model.addVar(vtype=g.GRB.INTEGER, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='z')
        model.update()
        model.addConstr(g.LinExpr(self.vals[self.sigma, 1:], zVars) + g.LinExpr(self.vals[0, 1:], xVars) >= 1)
        model.addConstr(g.quicksum(xVars) == 1)
        qRange = list(range(1, self.p))
        model.addConstr(g.LinExpr(qRange, zVars) + g.LinExpr(qRange, xVars) + self.p * modQVar == 0)
        model.optimize()
        if model.Status == g.GRB.INFEASIBLE:
            return True
        else:
            # print('invalid y/x:')
            # print([x.X for x in xVars])
            # print([y.X for y in zVars])
            return False


    def isValidGeneric(self):
        if np.all(self.m == 0):
            return True
        I = [i for i in range(1, self.p) if self.vals[self.sigma, i] >= -self.sigma]
        assignment = np.zeros(len(I), dtype=np.int)
        return self.validGenericHelper(I, 0, 0, assignment)

    def validGenericHelper(self, I, index, curSum, assignment):
        if index == len(I):
            r = -curSum
            return self.m[r] == 0
            if r <= 0 or r >= self.p:
                return True
            return self.m[r] == 0

        i = I[index]
        val = self.vals[self.sigma, i]
        maxN = (curSum + self.sigma) // (-val)
        for ni in range(maxN + 1):
            assignment[index] = ni
            if not self.validGenericHelper(I, index + 1, curSum + ni * val, assignment):
                return False
        return True


    def isSymmetric(self):
        if self.max == self.p - 1:
            # all-zero
            return True
        return self.sigma % 2 == 1 and all(self.m[i] + self.m[self.sigma - i] == 1 for i in range(0, self.sigma //2 + 1))
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
        if self.p == 2:
            return False
        Wproj = sorted(self.vals[0, :])[1:-1]
        maxVal = Wproj[-1] // 2
        cand = []
        for i in Wproj:
            if i <= maxVal:
                cand.append(i)
            else:
                break
        if len(cand) < (self.p - 3) // 2:
            return False
        for size in range((self.p - 3) // 2, len(cand) + 1):
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
        assert other.q == self.p
        return np.all(self.vals[0, :] >= other.vals[0, :]) and \
               np.all(self.vals[self.sigma, :] >= other.vals[other.sigma, :]) and \
               self.m != other.shifts

    def isDominatedBy(self, others):
        """Strong dominance check using linear programming"""
        if len(others) == 0:
            return False
        import gurobimh as g
        model = g.Model()
        model.setParam('OutputFlag', 0)
        vars = [model.addVar(lb=0) for _ in range(3 * len(others) * (self.p - 1))]
        model.update()
        Ahi = np.empty((self.p - 1, len(others) * (self.p - 1)))
        Alo = np.empty((self.p - 1, len(others) * (self.p - 1)))

        for i, cls in enumerate(others):
            for automorphism in range(1, self.p):
                for j in range(1, self.p):
                    Ahi[j-1, i*(self.p-1) + automorphism - 1] = cls.vals[0, automorphism*j % self.p]
                    Alo[j-1, i*(self.p-1) + automorphism - 1] = cls.vals[cls.sigma, automorphism*j % self.p]

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
        #     print('others', [o.m for o in others])
        #     print('solution', [v.X for v in vars])
        return model.Status != g.GRB.INFEASIBLE

    def embedConstant(self, zeta):
        ret = [0] * self.p
        ret[zeta] = 1
        return ret

    def embedFlanagan(self, zeta):
        ret = [0] * (self.p - 1)
        if zeta != 0:
            ret[zeta-1] = 1
        return ret

    def hasBadGcd(self):
        return functools.reduce(fractions.gcd, self.vals[0]) != 1

    def rankOfTightCodewords(self, d=3):
        nCols = d*(self.p - 1)
        nRows = (d-1)*(self.p - 1)
        cws = []
        for firsts in itertools.product(list(range(self.p)), repeat=d-1):
            last = -sum(firsts) % self.p
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
        q = self.p
        sigma = self.sigma
        for i in range(1, q):
            cws.append([i, 0, q-i])
            cws.append([0, i, q-i])
        for j in range(1, q):
            if j == sigma:
                continue
            cws.append([-j, j-sigma, sigma])
        for j in range(2, q-1):
            if self.m[j] == 1:
                continue
            for i in range(2, j+1):
                if self.m[i] == 1:
                    continue
                if i + j == q:
                    continue
                if (i+j < q and self.m[i+j] == 0) or (i+j >= q and self.m[i+j-q] == 1):
                    cws.append([-i, -j, (i + j) % q])
        mat = np.array([flanaganEmbedding(cw, q) for cw in cws])
        return np.linalg.matrix_rank(mat)

    def hasFullRank(self, d=3):
        rank = self.rankOfTightCodewords(d)
        if rank == d*(self.p-1) - 1:
            return True
        else:
            #print('rank=', rank)
            return False

    def __eq__(self, other):
        return self.m == other.m

    def __ne__(self, other):
        return self.m != other.m


    @classmethod
    def validFacetDefining(cls, q, d=3):
        """Return all valid facet-defining unique classes for given `q`."""
        classes = []
        for shifts in sorted(itertools.product((0,1), repeat=q-1), key=sum):
            if shifts == (1,0) * ((q-1)//2):
                continue  # redundant class 0101010...
            c = BuildingBlockClass((0,) + shifts)
            if c.isSymmetric() and c.isValid() and c.hasFullRank(d):
                classes.append(c)
        return classes

    @classmethod
    def uniqueClasses(cls, q):
        """returns all unique classes for given `q`."""
        return [BuildingBlockClass((0,) + shifts)
                for shifts in sorted(itertools.product((0, 1), repeat=q - 1), key=sum)
                if shifts != (1,0) * ((q-1)//2)]


    def __str__(self):
        return str(self.m)


if __name__ == '__main__':
    # c = BuildingBlockClass('0110000')
    # c = BuildingBlockClass('0010000')
    # print(c.rankOfTightCodewords(5))
    # sys.exit(0)
    import pprint
    for q in [2,3,5,7,11,13,17, 19]:
        print('Searching for q={} ...'.format(q))
        count = dict(symmetric=0, valid=0, unique=0, doublySymmetric=0, facet=0)
        for shifts in sorted(itertools.product((0,1), repeat=q-1), key=sum):
            c = BuildingBlockClass((0,) + shifts)
            if c.isValidGeneric():

                #print('{} is valid {}'.format(c, 'and symmetric' if c.isSymmetric() else ''))
                count['valid'] += 1
                if count['valid'] % 100 == 0:
                    print(count['valid'])
                if c.hasBadGcd():
                    continue# only m=0101010...
                count['unique'] += 1
                # if c.hasFullRank(4):
                #     count['facet'] += 1
                #     if not c.isSymmetric():
                #         raise Exception()
                if c.isSymmetric():
                    count['symmetric'] += 1

                    # if c.hasFullRank(3):
                    #     print('FAC', end='')
        print('Search for p={} finished. Stats:'.format(q))
        pprint.pprint(count)
        print('*******************************************')
    sys.exit(0)
    q = 13
    classes = set()
    # c1 = BuildingBlockClass(q, (0, 0, 0, 1, 1))
    # c2 = BuildingBlockClass(q, (0, 0, 1, 1, 0))
    # print(c1.vals)
    # print(c1.isValid(), c1.isSymmetric())
    # print(c2.vals)
    # print(c2.isValid(), c2.isSymmetric())
    #sys.exit(0)
    for shifts in sorted(itertools.product((0,1), repeat=q-1), key=sum):
        #shifts = (0,0,1,0,1,0)
        c = BuildingBlockClass(q, (0,) + shifts)
        print('testing {}'.format(c))
        print(c.vals)
        if c.hasBadGcd():
            print('{} bad gcd'.format(c))
        elif not c.isValid():
            print('{} invalid'.format(c))
        # elif not c.isSymmetric():
        #     print('{} unsymmetric'.format(c))
        # elif any(cls.dominatesSingle(c) for cls in classes):
        #     print("{} dominates {}".format(cls, c))
        # elif c.isDominatedBy(classes):
        #     print('strong domination {}'.format(c))
        elif not c.hasFullRank(3):
            print('{} does not have full rank'.format(c))
        else:
            print('{} ok'.format(c))
            classes.add(c)
            # for cls in list(classes):
            #     if c.dominatesSingle(cls):
            #         print('{} dominates {}!'.format(c, cls))
            #         classes.remove(cls)
        #break
    print('found {} classes'.format(len(classes)))

    for cls in classes:
        print(cls)
    # print('now performing dominance check')
    # for cls in classes:
    #     if cls.isDominatedBy(classes - set([cls])):
    #         print('oh, {} dominated!'.format(cls))
    #     else:
    #         print('{} ok'.format(cls))
    print('{} classes remaining'.format(len(classes)))
    for cls in classes:
        print(cls)