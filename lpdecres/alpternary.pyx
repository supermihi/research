# -*- coding: utf-8 -*-
# cython: boundscheck=True
# cython: nonecheck=True
# cython: cdivision=False
# cython: wraparound=True
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
import gurobimh as gu


cdef class AdaptiveTernaryLPDecoder(Decoder):

    cdef list Nj, hj, xj, xvars
    cdef object model, timer
    cdef double[:,:] xVals, xjVals
    cdef np.ndarray psiPlus, psiMinus
    cdef np.int_t[:,:] matrix
    cdef np.int_t[:] k
    cdef int[:] theta
    cdef object[:, :] x

    def __init__(self, code):
        assert code.q == 3
        Decoder.__init__(self, code, name='AdaptiveTernaryLPDecoder')
        model = gu.Model('')
        self.model = model
        self.matrix = code.parityCheckMatrix
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        self.x = np.empty((code.blocklength, 3), dtype=np.object)
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
            xj = []
            for i in range(d):
                if hj[i] == 1:
                    xj += [self.x[Nj[i], 1], self.x[Nj[i], 2]]
                else:
                    xj += [self.x[Nj[i], 2], self.x[Nj[i], 1]]
            self.xj.append(xj)
        self.xVals = np.empty((code.blocklength, 3))
        self.psiPlus = np.empty(code.blocklength)
        self.psiMinus = np.empty(code.blocklength)
        self.xjVals = np.empty((code.blocklength, 3))
        self.k = np.empty(code.blocklength, dtype=np.int)
        self.theta = np.empty(2*code.blocklength, dtype=np.intc)

    def setStats(self, stats):
        for stat in 'cuts', 'totalLPs', 'lpTime', 'simplexIters':
            if stat not in stats:
                stats[stat] = 0
        Decoder.setStats(self, stats)

    cdef bint cutSearch(self, int j):
        cdef:
            np.intp_t[:] Nj = self.Nj[j]
            np.int_t[:] hj = self.hj[j]
            int d = Nj.size
            double[:,:] xjVals = self.xjVals
            double[:,:] xVals = self.xVals
            int i, iPlus, jPlus, iMinus, jMinus
            np.int_t[:] k = self.k
            np.ndarray[dtype=double, ndim=1] psiMinus = self.psiMinus
            np.ndarray[dtype=double, ndim=1] psiPlus = self.psiPlus
            np.intp_t[:] argsPlus, argsMinus
            double a, b, Psi
            int eta, kSum, kappa
            bint cut=False, anyCut=False
            int[:] theta = self.theta
            double iMinusV, jMinusV, iPlusV, jPlusV
        for i in range(d):
            if hj[i] == 1:
                xjVals[i, 1] = xVals[Nj[i], 1]
                xjVals[i, 2] = xVals[Nj[i], 2]
            else:
                xjVals[i, 1] = xVals[Nj[i], 2]
                xjVals[i, 2] = xVals[Nj[i], 1]

        # Theta 1 case
        Psi = 0
        eta = 0
        cut = False
        iPlusV = jPlusV = iMinusV = jMinusV = np.inf
        iPlus = iMinus = jPlus = jMinus = 0
        for i in range(d):
            a = 1 - 3*xjVals[i, 2]
            b = 2 - 3*xjVals[i, 1] - 3*xjVals[i, 2]

            if a <= 0 and b <= 0:
                k[i] = 0
                eta += 2
                Psi += 2 - xjVals[i, 1] - 2*xjVals[i, 2]
                psiPlus[i] = -b
                psiMinus[i] = -a
            elif a >= 0 and b <= a:
                k[i] = -1
                eta += 1
                Psi += 1 - xjVals[i, 1] + xjVals[i, 2]
                psiPlus[i] = a
                psiMinus[i] = a - b
            else:
                k[i] = -2
                Psi += 2*xjVals[i, 1] + xjVals[i, 2]
                psiPlus[i] = b - a
                psiMinus[i] = b
            if psiPlus[i] < jPlusV:
                if psiPlus[i] < iPlusV:
                    jPlus = iPlus
                    jPlusV = iPlusV
                    iPlus = i
                    iPlusV = psiPlus[i]
                else:
                    jPlus = i
                    jPlusV = psiPlus[i]
            if psiMinus[i] < jMinusV:
                if psiMinus[i] < iMinusV:
                    jMinus = iMinus
                    jMinusV = iMinusV
                    iMinus = i
                    iMinusV = psiMinus[i]
                else:
                    jMinus = i
                    jPlusV = psiMinus[i]
        if Psi < 2 - 1e-12 and eta % 3 == 2:
            cut = True
        elif Psi < 2 - 1e-12:
            # argsPlus = np.argsort(psiPlus[:d])
            # argsMinus = np.argsort(psiMinus[:d])
            # iPlus = argsPlus[0]
            # jPlus = argsPlus[1]
            # iMinus = argsMinus[0]
            # jMinus = argsMinus[1]
            if eta % 3 == 1:
                if psiPlus[iPlus] < psiMinus[iMinus] + psiMinus[jMinus]:
                    k[iPlus] += 1
                    Psi += psiPlus[iPlus]
                else:
                    k[iMinus] -= 1
                    k[jMinus] -= 1
                    Psi += psiMinus[iMinus] + psiMinus[jMinus]
            else:
                if psiMinus[iMinus] < psiPlus[iPlus] + psiPlus[jPlus]:
                    k[iMinus] -= 1
                    Psi += psiMinus[iMinus]
                else:
                    k[iPlus] += 1
                    k[jPlus] += 1
                    Psi += psiPlus[iPlus] + psiPlus[jPlus]
            if Psi < 2 - 1e-12:
                cut = True
        if cut:
            kSum = 0
            for i in range(d):
                k[i] = (k[i] + 5) % 3 - 2
                kSum += k[i]
            kappa = 2*(d-1) + kSum
            for i in range(d):
                if k[i] == -2:
                    theta[2*i] = -2
                    theta[2*i+1] = -1
                    #lhs += -2 * xj[i, 1] - xj[i, 2]
                elif k[i] == -1:
                    theta[2*i] = 1
                    theta[2*i+1] = -1
                    #lhs += xj[i, 1] - xj[i, 2]
                else:
                    theta[2*i] = 1
                    theta[2*i+1] = 2
                    #lhs += xj[i, 1] + 2*xj[i, 2]
            with self.timer:
                self.model.addConstr(gu.LinExpr(theta[:2*d], self.xj[j]), gu.GRB.LESS_EQUAL, kappa)
            self._stats['lpTime'] += self.timer.duration
            anyCut = True
            self._stats['cuts'] += 1

        # Theta 2 case
        Psi = 0
        eta = 0
        cut=False
        iPlusV = jPlusV = iMinusV = jMinusV = np.inf
        iPlus = iMinus = jPlus = jMinus = 0
        for i in range(d):
            a = 1 - 3*xjVals[i, 2]
            b = 2 - 3*xjVals[i, 1] - 3*xjVals[i, 2]

            if b <= 0 and b <= a:
                k[i] = 0
                psiPlus[i] = -b
                psiMinus[i] = a - b
                eta += 2
                Psi += 2 - 2*xjVals[i, 1] - xjVals[i, 2]
            elif a <= 0 and a <= b:
                k[i] = -1
                psiPlus[i] = b - a
                psiMinus[i] = -a
                eta += 1
                Psi += 1 + xjVals[i, 1] - xjVals[i, 2]
            else:
                k[i] = -2
                psiPlus[i] = a
                psiMinus[i] = b
                Psi += xjVals[i, 1] + 2*xjVals[i, 2]
            if psiPlus[i] < jPlusV:
                if psiPlus[i] < iPlusV:
                    jPlus = iPlus
                    jPlusV = iPlusV
                    iPlus = i
                    iPlusV = psiPlus[i]
                else:
                    jPlus = i
                    jPlusV = psiPlus[i]
            if psiMinus[i] < jMinusV:
                if psiMinus[i] < iMinusV:
                    jMinus = iMinus
                    jMinusV = iMinusV
                    iMinus = i
                    iMinusV = psiMinus[i]
                else:
                    jMinus = i
                    jPlusV = psiMinus[i]
        if Psi < 2 - 1e-12 and eta % 3 == 2:
            cut = True
        elif Psi < 2 - 1e-12:
            # argsPlus = np.argsort(psiPlus[:d])
            # argsMinus = np.argsort(psiMinus[:d])
            # iPlus = argsPlus[0]
            # jPlus = argsPlus[1]
            # iMinus = argsMinus[0]
            # jMinus = argsMinus[1]
            if eta % 3 == 1:
                if psiPlus[iPlus] < psiMinus[iMinus] + psiMinus[jMinus]:
                    k[iPlus] += 1
                    Psi += psiPlus[iPlus]
                else:
                    k[iMinus] -= 1
                    k[jMinus] -= 1
                    Psi += psiMinus[iMinus] + psiMinus[jMinus]
            else:
                if psiMinus[iMinus] < psiPlus[iPlus] + psiPlus[jPlus]:
                    k[iMinus] -= 1
                    Psi += psiMinus[iMinus]
                else:
                    k[iPlus] += 1
                    k[jPlus] += 1
                    Psi += psiPlus[iPlus] + psiPlus[jPlus]
            if Psi < 2 - 1e-12:
                cut = True
        if cut:
            kSum = 0
            for i in range(d):
                k[i] = (k[i] + 5) % 3 - 2
                kSum += k[i]
            kappa = 2*(d-1) + kSum
            for i in range(d):
                if k[i] == -2:
                    theta[2*i] = -1
                    theta[2*i+1] = -2
                    #lhs += - xj[i, 1] - 2* xj[i, 2]
                elif k[i] == -1:
                    theta[2*i] = -1
                    theta[2*i+1] = 1
                    #lhs += - xj[i, 1] + xj[i, 2]
                else:
                    theta[2*i] = 2
                    theta[2*i+1] = 1
                    #lhs += 2*xj[i, 1] + xj[i, 2]
            with self.timer:
                self.model.addConstr(gu.LinExpr(theta[:2*d], self.xj[j]), gu.GRB.LESS_EQUAL, kappa)
            self._stats['lpTime'] += self.timer.duration
            self._stats['cuts'] += 1
            anyCut = True
        return anyCut


    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        cdef int i, j
        cdef bint cutAdded
        cdef double[:, :] xVals = self.xVals
        with self.timer:
            self.model.setObjective(gu.LinExpr(self.llrs, self.xvars))
            for constr in self.model.getConstrs():
                if constr.ConstrName[0] != 'S':
                    self.model.remove(constr)
        self._stats['lpTime'] += self.timer.duration
        while True:
            self._stats['totalLPs'] += 1
            with self.timer:
                self.model.optimize()
            self._stats['lpTime'] += self.timer.duration
            self._stats['simplexIters'] += self.model.IterCount
            if self.model.Status != gu.GRB.OPTIMAL:
                raise RuntimeError('unknown Gurobi status {}'.format(self.model.Status))
            cutAdded = False
            # fill self.xVals
            for i in range(self.code.blocklength):
                xVals[i, 1] = self.x[i, 1].X
                xVals[i, 2] = self.x[i, 2].X
            for j in range(self.matrix.shape[0]):
                cutAdded |= self.cutSearch(j)
            if not cutAdded:
                self.mlCertificate = self.foundCodeword = True
                self.objectiveValue = self.model.ObjVal
                for i in range(self.code.blocklength):
                    self.solution[i] = 0
                    for j in (1, 2):
                        if xVals[i, j] > 1e-3:
                            if self.solution[i] != 0:
                                self.mlCertificate = self.foundCodeword = False
                                self.solution[:] = .5  # error
                                return
                            else:
                                self.solution[i] = j
                break

    def params(self):
        return OrderedDict(name=self.name)

