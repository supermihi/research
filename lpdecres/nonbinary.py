# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import itertools
import numpy as np
import sys

class BuildingBlockClass:

    def __init__(self, q, shifts):
        self.q = q
        self.shifts = shifts[:]
        self.vals = np.empty((q, q), dtype=np.int)
        self.computeVals()

    def computeVals(self):
        for i in range(self.q):
            self.vals[0, i] = i + self.q*self.shifts[i]
        self.sigma = np.argmax(self.vals[0, :])
        for j in range(1, self.q):
            for i in range(self.q):
                self.vals[j, i] = self.vals[0, (i + j) % self.q] - self.vals[0, j]
                #self.vals[j, (i - j) % self.q] = self.vals[0, i] - self.vals[0, j]


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
        sortedVals = sorted(self.vals[0, :])
        maxVal = sortedVals[-1]
        if maxVal % 2 == 1:
            return False
        for i in range(self.q//2):
            if sortedVals[i] != maxVal - sortedVals[q-i-1]:
                #print(i, sortedVals[i], sortedVals[q-i-1])
                return False
        if sortedVals[self.q//2] != maxVal // 2:
            #print(':(')
            return False
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
                    Ahi[j-1, i*(q-1) + automorphism - 1] = cls.vals[0, automorphism*j % self.q]
                    Alo[j-1, i*(q-1) + automorphism - 1] = cls.vals[cls.sigma, automorphism*j % self.q]

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
        if model.status != g.GRB.INFEASIBLE:
            print(self.vals)
            print([o.shifts for o in others])
            print([v.X for v in vars])
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
        for divisor in range(2, self.q-1):
            if np.all(self.vals[0] % divisor == 0):
                return True
        return False

    def hasFullRank(self, d=3):
        nCols = d*(self.q-1)
        nRows = (d-1)*(self.q-1)
        M = np.zeros((nRows, nCols))
        for i in range(nRows):
            M[i, i] = 1
        for k in range(d-1):
            for i in range(self.q - 1):
                M[i+(q-1)*k, (nCols-1) - i] = 1
        cws = []
        for firsts in itertools.product(list(range(q)), repeat=d-1):
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
            return False
        print(M)
        Mplus = np.array(cws, dtype=np.double)
        #M = np.concatenate((M, Mplus), axis=0)
        M = Mplus
        # print(M)
        rank = np.linalg.matrix_rank(M)
        if rank == d*(q-1) - 1:
            return True
        else:
            print('rank=', rank)
            return False


    def __str__(self):
        return str(self.shifts)



if __name__ == '__main__':
    for q in [2,3,5,7,11,13,17,19,23,29]:
        count = 0
        for shifts in sorted(itertools.product((0,1), repeat=q-1), key=sum):
            c = BuildingBlockClass(q, (0,) + shifts)
            if c.isValid():
                count += 1
                if count % 100 == 0:
                    print(count)
        print('q={}: {} valid classes'.format(q, count))
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
        elif not c.hasFullRank(6):
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