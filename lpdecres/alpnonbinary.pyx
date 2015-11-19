# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False
# cython: wraparound=False
# cython: language_level=3
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import gurobimh as gu
import numpy as np
cimport numpy as np
from libc.math cimport fmin
from numpy.math cimport INFINITY
from lpdec.decoders.gurobihelpers import GurobiDecoder
from lpdec.utils import Timer
from lpdec import gfqla
from lpdecres.bbclass import BuildingBlockClass
from lpdecres.bbclass cimport BuildingBlockClass


class NonbinaryALPDecoder(GurobiDecoder):
    """Adaptive LP decoder for non-binary codes.
    """

    def __init__(self, code, name=None, gurobiParams=None, gurobiVersion=None):
        if name is None:
            name = 'NonbinaryALPDecoder'
        GurobiDecoder.__init__(self, code, name, gurobiParams, gurobiVersion, integer=False)

        self.matrix = code.parityCheckMatrix
        self.htilde = self.matrix.copy()
        q = self.q = code.q
        self.bbClasses = BuildingBlockClass.validFacetDefining(q)
        print('found {} valid classes:'.format(len(self.bbClasses)))
        for c in self.bbClasses:
            print(c)

        # add simplex constraints
        for i in range(self.code.blocklength):
            self.model.addConstr(gu.quicksum(self.x[i, j] for j in range(1, q)) <= 1,
                                 name='S{}'.format(i))
        self.model.update()
        self.Nj = []  # list of check-node neighbors (nonzero indices of j-th row)
        self.hj = []  # list of H[j, Nj] (non-zero entries of j-th row)

        self.timer = Timer()

        # inverse elements in \F_q
        self.inv = np.array([0] + [gfqla.inv(i, q) for i in range(1, q)])

        self.permutations = np.zeros((q, q), dtype=np.int)
        for j in range(1, q):
            for i in range(q):
                self.permutations[j, i] = (i*j) % q
        for j, row in enumerate(code.parityCheckMatrix):
            Nj = np.flatnonzero(row)
            self.Nj.append(Nj)
            hj = row[Nj]
            self.hj.append(hj)
        self.xVals = np.zeros((code.blocklength, q))
        self.initCutSearchTempArrays()

    def initCutSearchTempArrays(self):
        self.rotatedXvals = np.zeros((self.code.blocklength, self.q))
        self.v = np.empty((self.code.blocklength, self.q))
        self.thetaHat = np.empty(self.code.blocklength, dtype=np.int)

        # dynamic programming tables
        self.T = np.empty((self.code.blocklength, self.q))
        self.S = np.empty((self.code.blocklength, self.q), dtype=np.int)
        self.coeffs = np.empty(self.code.blocklength * (self.q-1))
        self.vars = np.empty(self.code.blocklength * (self.q-1), dtype=np.object)
        self.Njtmp = np.empty(self.code.blocklength, dtype=np.intp)
        self.hjtmp = np.empty(self.code.blocklength, dtype=np.int)


    def setStats(self, stats):
        for stat in 'cuts', 'totalLPs', 'simplexIters', 'optTime':
            if stat not in stats:
                stats[stat] = 0
        GurobiDecoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        cdef:
            int i, j, row, phi, q = self.q
            bint cutAdded
            double[:, ::1] xVals = self.xVals
            BuildingBlockClass bbClass
        for constr in self.model.getConstrs()[self.code.blocklength:]:
            # remove all constraints except "simplex" (generalized box) inequalities
            self.model.remove(constr)
        self.mlCertificate = self.foundCodeword = True
        while True:
            self._stats['totalLPs'] += 1
            with self.timer:
                self.model.optimize()
            self._stats['optTime'] += self.timer.duration
            self._stats['simplexIters'] += self.model.IterCount
            if self.model.Status != gu.GRB.OPTIMAL:
                raise RuntimeError('unknown Gurobi status {}'.format(self.model.Status))
            cutAdded = False
            # fill self.xVals
            for i in range(self.code.blocklength):
                for j in range(1, q):
                    xVals[i, j] = self.x[i, j].X
            for row in range(self.matrix.shape[0]):
                for bbClass in self.bbClasses:
                    for phi in range(1, q):
                        cutAdded |= self.cutSearch(self.hj[row], self.Nj[row], self.hj[row].size, bbClass, phi)
            if not cutAdded:
                Htilde = self.diagonalize()
                for row in range(Htilde.shape[0]):
                    d = self.makeHjNj(Htilde[row,:])
                    for bbClass in self.bbClasses:
                        for phi in range(1, q):
                            cutAdded |= self.cutSearch(self.hjtmp, self.Njtmp, d, bbClass, phi)
                if not cutAdded:
                    self.mlCertificate = self.foundCodeword = self.readSolution()
                    break

    def makeHjNj(self, np.int_t[::1] row):
        cdef int j, i = 0
        for j in range(row.size):
            if row[j] != 0:
                self.hjtmp[i] = row[j]
                self.Njtmp[i] = j
                i += 1
        return i

    def cutSearch(self, np.int_t[::1] hj, np.intp_t[::1] Nj, int d, BuildingBlockClass bbClass, int phi):
        """Non-binary cut search algorithm.

        Parameters
        ----------
        hj : np.int_t[:]
            Row of parity-check matrix to test.
        Nj : np.intp_t[:]
            List of non-zero indices of PCM row.
        d : int
            check node degree (i.e. number of entries in hj and Nj to consider)
        bbClass : BuildingBlockClass
            Building block class to separate.
        phi : int from {1,...,q-1}
            Automorphism (permutation).
        """
        cdef:
            int q = self.q
            np.int_t[:,::1] t = bbClass.vals, S = self.S, permutations = self.permutations
            int sigma = bbClass.sigma

            double[:, ::1] rotatedXvals = self.rotatedXvals
            int i, j, k, phiHjInv, zeta, next_
            double[:, ::1] v = self.v, T = self.T
            np.int_t[::1] thetaHat = self.thetaHat
            int sumKhat = 0, goalSum = (d-1)*sigma % q
            double PsiVal = 0
            double minVi, columnMin

        # fill rotatedXvals matrix that contains x vals "rotated" according to hj and phi
        for i in range(d):
            phiHjInv = self.inv[phi*hj[i] % q]
            for j in range(1, q):
                rotatedXvals[i, j] = self.xVals[Nj[i], permutations[phiHjInv, j]]

        # compute the v^j(x_i) and unconstrained solution on-the-fly
        for i in range(d):
            minVi = INFINITY
            for j in range(q):
                v[i, j] = t[0, sigma] - t[0, j]  #- np.dot(t[j], rotatedXvals[i])
                for k in range(1, q):
                    v[i, j] -= t[j, k] * rotatedXvals[i, k]
                if v[i, j] < minVi:
                    minVi = v[i, j]
                    thetaHat[i] = j
            sumKhat += thetaHat[i]
            PsiVal += v[i, thetaHat[i]]

        # check shortcutting conditions
        if PsiVal >= t[0, sigma] - 1e-5:
            return False  # unconstrained solution already too large
        if sumKhat % q == (d-1)*sigma % q:
            self.insertCut(bbClass, hj, Nj, d, thetaHat, phi)
            return True

        # start DP approach
        for zeta in range(q):
            T[0, zeta] = v[0, zeta]
            S[0, zeta] = zeta
        for i in range(1, d):
            columnMin = INFINITY
            for zeta in range(q):
                # if i == d - 1 and zeta != goalSum:
                #     continue
                T[i, zeta] = INFINITY
                S[i, zeta] = -1
                for alpha in range(q):
                    val = v[i, alpha] + T[i-1, (zeta - alpha) % q]
                    if val < T[i, zeta]:
                        T[i, zeta] = val
                        S[i, zeta] = alpha
                columnMin = fmin(columnMin, T[i, zeta])
            # if columnMin >= t[0, sigma]:
            #     print('MINMIN')
            #     return False  # see remark 8, point 3
        if T[d-1, goalSum] < t[0, sigma] - 1e-5:
            # found cut
            thetaHat[d-1] = S[d-1, goalSum]
            next_ = (goalSum - thetaHat[d-1]) % q
            for i in range(d-2, -1, -1):
                thetaHat[i] = S[i, next_]
                next_ = (next_ - thetaHat[i]) % q
            self.insertCut(bbClass, hj, Nj, d, thetaHat, phi)
            return True
        return False

    def insertCut(self, BuildingBlockClass bbClass, np.int_t[::1] hj, np.intp_t[::1] Nj, int d, np.int_t[::1] theta, int phi):
        cdef:
            int  q = self.q
            double[::1] coeffs = self.coeffs
            np.ndarray vars = self.vars
            double kappa = (d-1)*bbClass.vals[0, bbClass.sigma]
            int i, j, tki, hjPhi
            np.int_t[:, ::1] permutations = self.permutations, vals = bbClass.vals
        for i in range(d):
            hjPhi = hj[i]*phi % q
            for j in range(q-1):
                coeffs[(q-1)*i + j] = vals[theta[i], permutations[hjPhi, j + 1]]
                vars[(q-1)*i + j] = self.x[Nj[i], j + 1]
            kappa -= vals[0, theta[i]]
        self.model.addConstr(gu.LinExpr(coeffs[:(q-1)*d], vars[:(q-1)*d]), gu.GRB.LESS_EQUAL, kappa)

    def diagonalize(self):
        """Perform gaussian elimination on the code's parity-check matrix to search for RPC cuts.
        """
        import scipy.stats
        from lpdec import gfqla
        # compute hidden 0 xVals
        self.xVals[:, 0] = 1 - np.sum(self.xVals[:, 1:], 1)
        entropies = -scipy.stats.entropy(self.xVals.T)  # large entropies first
        #entropies = np.sum(np.abs(self.xVals - 1./self.q), 1)
        sortIndices = np.argsort(entropies)
        gfqla.gaussianElimination(self.htilde, sortIndices, diagonalize=True, q=self.q)
        return self.htilde


    def params(self):
        ret = GurobiDecoder.params(self)
        return ret