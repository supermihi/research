# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict
import numpy as np
from lpdec.decoders.base import Decoder
import gurobipy as gu


class AdaptiveTernaryLPDecoder(Decoder):

    def __init__(self, code):
        assert code.q == 3
        Decoder.__init__(self, code, name='AdaptiveTernaryLPDecoder')
        model = gu.Model()
        self.model = model
        self.matrix = code.parityCheckMatrix
        model.setParam('OutputFlag', 0)
        self.x = OrderedDict()
        for i in range(code.blocklength):
            for k in range(1, code.q):
                var = self.model.addVar(0, 1, name='x{},{}'.format(i, k))
                self.x[i, k] = var
        self.model.update()
        for i in range(self.code.blocklength):
            self.model.addConstr(self.x[i, 1] + self.x[i, 2], gu.GRB.LESS_EQUAL, 1,
                                 name='S{}'.format(i))
        self.model.update()

    def setStats(self, stats):
        if 'cuts' not in stats:
            stats['cuts'] = 0
        if 'totalLPs' not in stats:
            stats['totalLPs'] = 0
        Decoder.setStats(self, stats)

    def cutSearch(self, j):
        # print('cut search {}'.format(j))
        Nj = np.flatnonzero(self.matrix[j])
        # print('Nj={}'.format(Nj))
        hj = self.matrix[j, Nj]
        # print('hj={}'.format(hj))
        d = Nj.size
        k = np.empty((d, 2), dtype=np.int)
        xj = {}
        for i in range(d):
            if hj[i] == 1:
                xj[i, 1] = self.x[Nj[i], 1]
                xj[i, 2] = self.x[Nj[i], 2]
            else:
                # rotate
                assert hj[i] == 2
                # raise ValueError()
                xj[i, 1] = self.x[Nj[i], 2]
                xj[i, 2] = self.x[Nj[i], 1]
        # print(' ; '.join('{}, {}'.format(xj[i,1].X, xj[i,2].X) for i in range(d)))
        psiPlus = np.empty((d, 2))
        psiMinus = np.empty((d, 2))
        Psi = [0, 0]
        eta = [0, 0]
        for i in range(d):
            a = 1 - 3*xj[i, 2].X
            b = 2 - 3*xj[i, 1].X - 3*xj[i, 2].X
            if a <= 0 and b <= 0:
                k[i,0] = 0
                eta[0] += 2
                Psi[0] += 2 - xj[i,1].X - 2*xj[i,2].X
                psiPlus[i,0] = -b
                psiMinus[i,0] = -a
            elif a >= 0 and b <= a:
                k[i,0] = -1
                eta[0] += 1
                Psi[0] += 1 - xj[i,1].X + xj[i,2].X
                psiPlus[i,0] = a
                psiMinus[i,0] = a - b
            else:
                k[i,0] = -2
                Psi[0] += 2*xj[i,1].X + xj[i,2].X
                psiPlus[i,0] = b - a
                psiMinus[i,0] = b

            if b <= 0 and b <= a:
                k[i,1] = 0
                psiPlus[i,1] = -b
                psiMinus[i,1] = a - b
                eta[1] += 2
                Psi[1] += 2 - 2*xj[i,1].X - xj[i,2].X
            elif a <= 0 and a <= b:
                k[i,1] = -1
                psiPlus[i,1] = b - a
                psiMinus[i,1] = -a
                eta[1] += 1
                Psi[1] += 1 + xj[i,1].X - xj[i,2].X
            else:
                assert a >= 0 and b >= 0
                k[i,1] = -2
                psiPlus[i,1] = a
                psiMinus[i,1] = b
                Psi[1] += xj[i,1].X + 2*xj[i,2].X

        cut = [False, False]
        if sum(cut) == 2:
            print('two')
        for Theta in 0, 1:
            if Psi[Theta] < 2 - 1e-8 and eta[Theta] % 3 == 2:
                cut[Theta] = True
            elif Psi[Theta] < 2 - 1e-8:
                argsPlus = np.argsort(psiPlus[:, Theta])
                argsMinus = np.argsort(psiMinus[:, Theta])
                iPlus = argsPlus[0]
                jPlus = argsPlus[1]
                iMinus = argsMinus[0]
                jMinus = argsMinus[1]
                if eta[Theta] % 3 == 1:
                    if psiPlus[iPlus, Theta] < psiMinus[iMinus, Theta] + psiMinus[jMinus, Theta]:
                        k[iPlus, Theta] += 1
                        Psi[Theta] += psiPlus[iPlus, Theta]
                    else:
                        k[iMinus, Theta] -= 1
                        k[jMinus, Theta] -= 1
                        Psi[Theta] += psiMinus[iMinus, Theta] + psiMinus[jMinus, Theta]
                else:
                    if psiMinus[iMinus, Theta] < psiPlus[iPlus, Theta] + psiPlus[jPlus, Theta]:
                        k[iMinus, Theta] -= 1
                        Psi[Theta] += psiMinus[iMinus, Theta]
                    else:
                        k[iPlus, Theta] += 1
                        k[jPlus, Theta] += 1
                        Psi[Theta] += psiPlus[iPlus, Theta] + psiPlus[jPlus, Theta]
                if Psi[Theta] < 2 - 1e-8:
                    cut[Theta] = True
        if not any(cut):
            return False
        k = (k + 2) % 3 - 2
        if cut[0]:
            kk = k[:, 0]
            assert np.sum(kk) % 3 == (d-1) % 3
            kappa = 2*(d-1) + np.sum(kk)
            lhs = gu.LinExpr()
            for i in range(d):
                if kk[i] == -2:
                    lhs += -2 * xj[i, 1] - xj[i, 2]
                elif kk[i] == -1:
                    lhs += xj[i, 1] - xj[i, 2]
                else:
                    lhs += xj[i, 1] + 2*xj[i, 2]
            self.model.addConstr(lhs, gu.GRB.LESS_EQUAL, kappa)
            # print('insert Theta0 {} <= {}'.format(kk, kappa))
        if cut[1]:
            kk = k[:, 1]
            assert np.sum(kk) % 3 == (d-1) % 3
            kappa = 2*(d-1) + np.sum(kk)
            lhs = gu.LinExpr()
            for i in range(d):
                if kk[i] == -2:
                    lhs += - xj[i, 1] - 2* xj[i, 2]
                elif kk[i] == -1:
                    lhs += - xj[i, 1] + xj[i, 2]
                else:
                    lhs += 2*xj[i, 1] + xj[i, 2]
            self.model.addConstr(lhs, gu.GRB.LESS_EQUAL, kappa)
            # print('insert Theta1 {} <= {}'.format(kk, kappa))
        self._stats['cuts'] += sum(cut)
        return True


    def solve(self, lb=-np.inf, ub=np.inf):
        self.model.setObjective(gu.LinExpr(self.llrs, self.x.values()))
        for constr in self.model.getConstrs():
            if constr.ConstrName[0] != 'S':
                self.model.remove(constr)
        while True:
            self._stats['totalLPs'] += 1
            self.model.optimize()
            if self.model.Status != gu.GRB.OPTIMAL:
                raise RuntimeError('unknown Gurobi status {}'.format(self.model.Status))
            cutAdded = False
            for j in range(self.matrix.shape[0]):
                cutAdded |= self.cutSearch(j)
            if cutAdded:
                self.model.update()
            else:
                self.mlCertificate = self.foundCodeword = True
                for i in range(self.code.blocklength):
                    self.solution[i] = 0
                    for k in range(1, self.code.q):
                        if self.x[i, k].X > 1e-5:
                            if self.solution[i] != 0:
                                self.mlCertificate = self.foundCodeword = False
                                self.solution[i] = .5  # error
                            else:
                                self.solution[i] = k
                self.objectiveValue = self.model.ObjVal
                break

    def params(self):
        return OrderedDict(name=self.name)

if __name__ == '__main__':
    print('hello')
    from lpdec.imports import *
    code = TernaryGolayCode()
    print(code.parityCheckMatrix)
    decFL = FlanaganLPDecoder(code, ml=False)
    decTE = AdaptiveTernaryLPDecoder(code)
    simulation.ALLOW_DIRTY_VERSION = True
    simulation.ALLOW_VERSION_MISMATCH = True
    #simulation.DEBUG_SAMPLE = 8
    db.init('sqlite:///:memory:')
    for snr in frange(2, 2.1, .5):
        channel = AWGNC(snr, code.rate, seed=8374, q=3)
        simulator = Simulator(code, channel, [decFL, decTE], 'ternary')
        simulator.maxSamples = 1000
        simulator.maxErrors = 1000
        simulator.wordSeed = 1337
        simulator.outputInterval = 1
        simulator.dbStoreTimeInterval = 10
        simulator.revealSent = True
        simulator.concurrent = True
        simulator.run()
    print(decTE.stats())
