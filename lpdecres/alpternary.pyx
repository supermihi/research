# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict
import numpy as np
cimport numpy as np
from lpdec.decoders.base cimport Decoder
from lpdec.utils import Timer
import gurobipy as gu


cdef class AdaptiveTernaryLPDecoder(Decoder):

    cdef dict x
    cdef list Nj, hj, xj, xvars
    cdef object model, timer
    cdef double[:,:] xVals, psiPlus, psiMinus, xjVals
    cdef np.int_t[:,:] matrix
    cdef np.int_t[:,:] k

    def __init__(self, code):
        assert code.q == 3
        Decoder.__init__(self, code, name='AdaptiveTernaryLPDecoder')
        model = gu.Model()
        self.model = model
        self.matrix = code.parityCheckMatrix
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        self.x = dict()
        self.xvars = []
        for i in range(code.blocklength):
            for k in range(1, code.q):
                var = self.model.addVar(0, 1, name='x{},{}'.format(i, k))
                self.x[i, k] = var
                self.xvars.append(var)
        self.model.update()
        for i in range(self.code.blocklength):
            self.model.addConstr(self.x[i, 1] + self.x[i, 2], gu.GRB.LESS_EQUAL, 1,
                                 name='S{}'.format(i))
        self.model.update()
        self.Nj = []
        self.hj = []
        self.xj = []

        self.timer = Timer()
        for row in code.parityCheckMatrix:
            Nj = np.flatnonzero(row)
            self.Nj.append(Nj)
            hj = row[Nj]
            self.hj.append(hj)
            d = Nj.size
            xj = np.empty((d, 3), dtype=np.object)
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
            self.xj.append(xj)
        self.xVals = np.empty((code.blocklength, 3))
        self.psiPlus = np.empty((code.blocklength, 2))
        self.psiMinus = np.empty((code.blocklength, 2))
        self.xjVals = np.empty((code.blocklength, 3))
        self.k = np.empty((code.blocklength, 2), dtype=np.int)

    def setStats(self, stats):
        for stat in 'cuts', 'totalLPs', 'lpTime':
            if stat not in stats:
                stats[stat] = 0
        Decoder.setStats(self, stats)

    cdef cutSearch(self, int j):
        cdef:
            np.intp_t[:] Nj = self.Nj[j]
            np.int_t[:] hj = self.hj[j]
            int d = Nj.size
            double[:,:] xjVals = self.xjVals
            double[:,:] xVals = self.xVals
            int i
            np.int_t[:,:] k = self.k
            double[:,:] psiMinus = self.psiMinus
            double[:,:] psiPlus = self.psiPlus
        xj = self.xj[j]
        for i in range(d):
            if hj[i] == 1:
                xjVals[i, 1] = xVals[Nj[i], 1]
                xjVals[i, 2] = xVals[Nj[i], 2]
            else:
                xjVals[i, 1] = xVals[Nj[i], 2]
                xjVals[i, 2] = xVals[Nj[i], 1]
        Psi = [0, 0]
        eta = [0, 0]
        for i in range(d):
            a = 1 - 3*xjVals[i, 2]
            b = 2 - 3*xjVals[i, 1] - 3*xjVals[i, 2]
            if a <= 0 and b <= 0:
                k[i,0] = 0
                eta[0] += 2
                Psi[0] += 2 - xjVals[i, 1] - 2*xjVals[i, 2]
                psiPlus[i,0] = -b
                psiMinus[i,0] = -a
            elif a >= 0 and b <= a:
                k[i,0] = -1
                eta[0] += 1
                Psi[0] += 1 - xjVals[i, 1] + xjVals[i, 2]
                psiPlus[i,0] = a
                psiMinus[i,0] = a - b
            else:
                k[i,0] = -2
                Psi[0] += 2*xjVals[i, 1] + xjVals[i, 2]
                psiPlus[i,0] = b - a
                psiMinus[i,0] = b

            if b <= 0 and b <= a:
                k[i,1] = 0
                psiPlus[i,1] = -b
                psiMinus[i,1] = a - b
                eta[1] += 2
                Psi[1] += 2 - 2*xjVals[i, 1] - xjVals[i, 2]
            elif a <= 0 and a <= b:
                k[i,1] = -1
                psiPlus[i,1] = b - a
                psiMinus[i,1] = -a
                eta[1] += 1
                Psi[1] += 1 + xjVals[i, 1] - xjVals[i, 2]
            else:
                assert a >= 0 and b >= 0
                k[i,1] = -2
                psiPlus[i,1] = a
                psiMinus[i,1] = b
                Psi[1] += xjVals[i, 1] + 2*xjVals[i, 2]

        cut = [False, False]
        if sum(cut) == 2:
            print('two')
        for Theta in 0, 1:
            if Psi[Theta] < 2 - 1e-8 and eta[Theta] % 3 == 2:
                cut[Theta] = True
            elif Psi[Theta] < 2 - 1e-8:
                argsPlus = np.argsort(psiPlus[:d, Theta])
                argsMinus = np.argsort(psiMinus[:d, Theta])
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
            for i in range(d):
                k[i, Theta] = (k[i, Theta] + 2) % 3 - 2
        if not any(cut):
            return False
        if cut[0]:
            kk = k[:d, 0]
            kappa = 2*(d-1) + np.sum(kk)
            lhs = gu.LinExpr()
            for i in range(d):
                if kk[i] == -2:
                    lhs += -2 * xj[i, 1] - xj[i, 2]
                elif kk[i] == -1:
                    lhs += xj[i, 1] - xj[i, 2]
                else:
                    lhs += xj[i, 1] + 2*xj[i, 2]
            with self.timer:
                self.model.addConstr(lhs, gu.GRB.LESS_EQUAL, kappa)
            self._stats['lpTime'] += self.timer.duration
            # print('insert Theta0 {} <= {}'.format(kk, kappa))
        if cut[1]:
            kk = k[:d, 1]
            kappa = 2*(d-1) + np.sum(kk)
            lhs = gu.LinExpr()
            for i in range(d):
                if kk[i] == -2:
                    lhs += - xj[i, 1] - 2* xj[i, 2]
                elif kk[i] == -1:
                    lhs += - xj[i, 1] + xj[i, 2]
                else:
                    lhs += 2*xj[i, 1] + xj[i, 2]
            with self.timer:
                self.model.addConstr(lhs, gu.GRB.LESS_EQUAL, kappa)
            self._stats['lpTime'] += self.timer.duration
            # print('insert Theta1 {} <= {}'.format(kk, kappa))
        self._stats['cuts'] += sum(cut)
        return True


    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        self.model.setObjective(gu.LinExpr(self.llrs, self.xvars))
        for constr in self.model.getConstrs():
            if constr.ConstrName[0] != 'S':
                self.model.remove(constr)
        while True:
            self._stats['totalLPs'] += 1
            with self.timer:
                self.model.optimize()
            self._stats['lpTime'] += self.timer.duration
            if self.model.Status != gu.GRB.OPTIMAL:
                raise RuntimeError('unknown Gurobi status {}'.format(self.model.Status))
            cutAdded = False
            # fill self.xVals
            for i in range(self.code.blocklength):
                for j in (1, 2):
                    self.xVals[i, j] = self.x[i, j].X
            for j in range(self.matrix.shape[0]):
                cutAdded |= self.cutSearch(j)
            if cutAdded:
                self.model.update()
            else:
                self.mlCertificate = self.foundCodeword = True
                for i in range(self.code.blocklength):
                    self.solution[i] = 0
                    for k in range(1, 3):
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

