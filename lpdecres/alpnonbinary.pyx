# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3
# distutils: libraries = ["gurobi65"]
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np
from collections import OrderedDict

cimport numpy as np
from libc.math cimport fmin, log2, fabs
from numpy.math cimport INFINITY

import gurobimh as g
cimport gurobimh as g

from lpdec.decoders import gurobihelpers
from lpdec.decoders.base cimport Decoder
from lpdec.utils import Timer
from lpdec import gfqla
from lpdec.gfqla cimport gaussianElimination

from lpdecres.bbclass import BuildingBlockClass
from lpdecres.bbclass cimport BuildingBlockClass


cdef class NonbinaryALPDecoder(Decoder):
    """Adaptive LP decoder for non-binary codes.
    """

    cdef:
        g.Model model
        bint useEntropy
        bint RPC
        int q, blocklength
        list bbClasses,
        object grbParams, timer

        object[::1] vars
        np.int_t[::1] inv, thetaHat, hjtmp, dj
        int[::1] varInds
        np.intp_t[::1] Njtmp, successfulCols
        double[::1] coeffs
        double[::1] tmpVals

        object[:,::1] x
        np.intp_t[:, ::1] Nj
        np.int_t[:,::1] matrix, htilde, permutations, S, hj
        double[:, ::1] xVals, rotatedXvals, v, T



    def __init__(self, code, name=None, gurobiParams=None, gurobiVersion=None, **kwargs):
        if name is None:
            name = 'NonbinaryALPDecoder'
        Decoder.__init__(self, code=code, name=name)
        if gurobiParams is None:
            gurobiParams = dict()
        self.model = gurobihelpers.createModel(name, gurobiVersion, **gurobiParams)
        self.grbParams = gurobiParams.copy()

        self.useEntropy = kwargs.get('useEntropy', False)
        self.RPC = kwargs.get('RPC', True)

        self.x = np.empty((code.blocklength, code.q), dtype=np.object)
        for i in range(code.blocklength):
            for k in range(1, code.q):
                self.x[i, k] = self.model.addVar(0, 1, obj=0.0, vtype=g.GRB.CONTINUOUS, name='x{},{}'.format(i, k))
        self.model.update()


        self.matrix = code.parityCheckMatrix
        self.htilde = self.matrix.copy()
        q = self.q = code.q
        self.blocklength = code.blocklength
        self.bbClasses = BuildingBlockClass.validFacetDefining(q)
        print('found {} valid classes:'.format(len(self.bbClasses)))
        for c in self.bbClasses:
            print(c)

        # add simplex constraints
        for i in range(self.code.blocklength):
            self.model.addConstr(g.quicksum(self.x[i, j] for j in range(1, q)) <= 1)
        self.model.update()
        self.Njtmp = np.empty(self.code.blocklength, dtype=np.intp)
        self.hjtmp = np.empty(self.code.blocklength, dtype=np.int)
        self.Nj = np.empty((code.parityCheckMatrix.shape[0], code.blocklength), dtype=np.intp)
        self.hj = np.empty((code.parityCheckMatrix.shape[0], code.blocklength), dtype=np.int)
        self.dj = np.zeros(code.parityCheckMatrix.shape[0], dtype=np.int)

        self.timer = Timer()

        # inverse elements in \F_q
        self.inv = np.array([0] + [gfqla.inv(i, q) for i in range(1, q)])

        self.permutations = np.zeros((q, q), dtype=np.int)
        for j in range(1, q):
            for i in range(q):
                self.permutations[j, i] = (i*j) % q
        for j, row in enumerate(code.parityCheckMatrix):
            self.dj[j] = self.makeHjNj(row)
            self.Nj[j, :] = self.Njtmp[:]
            self.hj[j, :] = self.hjtmp[:]
        self.xVals = np.zeros((code.blocklength, q))
        self.tmpVals = np.empty(code.blocklength * (q-1))
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
        self.varInds = np.empty(self.code.blocklength * (self.q - 1), dtype=np.intc)
        self.successfulCols = np.empty(self.code.blocklength, dtype=np.intp)


    def setStats(self, stats):
        for stat in 'cuts', 'totalLPs', 'simplexIters', 'optTime':
            if stat not in stats:
                stats[stat] = 0
        Decoder.setStats(self, stats)

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        self.model.fastSetObjective(0, llrs.size, llrs)
        Decoder.setLLRs(self, llrs, sent)

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef:
            int i, j, d, row, phi, q = self.q
            double tmpSum
            bint cutAdded
            BuildingBlockClass bbClass
        cdef g.Constr constr

        for constr in self.model.getConstrs()[self.blocklength:]:
            # remove all constraints except "simplex" (generalized box) inequalities
            self.model.remove(constr)
        self.model.update()

        self.mlCertificate = self.foundCodeword = True
        while True:
            self._stats['totalLPs'] += 1
            with self.timer:
                self.model.optimize()
            self._stats['optTime'] += self.timer.duration
            self._stats['simplexIters'] += self.model.IterCount
            if self.model.Status != g.GRB.OPTIMAL:
                raise RuntimeError('unknown Gurobi status {}'.format(self.model.Status))
            cutAdded = False
            # fill self.xVals
            self.model.fastGetX(0, (self.q-1)*self.blocklength, self.tmpVals)
            for i in range(self.blocklength):
                tmpSum = 0
                for j in range(1, q):
                    self.xVals[i, j] = self.tmpVals[i*(q-1) + j - 1]
                    tmpSum += self.xVals[i, j]
                self.xVals[i, 0] = 1 - tmpSum
            for row in range(self.matrix.shape[0]):
                for bbClass in self.bbClasses:
                    for phi in range(1, q):
                        cutAdded |= self.cutSearch(self.hj[row, :], self.Nj[row, :], self.dj[row], bbClass, phi)
            if not cutAdded:
                if self.RPC:
                    self.diagonalize()
                    for row in range(self.htilde.shape[0]):
                        d = self.makeHjNj(self.htilde[row,:])
                        if d != 0:
                            for bbClass in self.bbClasses:
                                for phi in range(1, q):
                                    cutAdded |= self.cutSearch(self.hjtmp, self.Njtmp, d, bbClass, phi)
                if not cutAdded:
                    self.mlCertificate = self.foundCodeword = self.readSolution()
                    break

    cdef int makeHjNj(self, np.int_t[::1] row):
        cdef int j, i = 0
        for j in range(row.size):
            if row[j] != 0:
                self.hjtmp[i] = row[j]
                self.Njtmp[i] = j
                i += 1
        return i

    cdef object cutSearch(self, np.int_t[::1] hj, np.intp_t[::1] Nj, int d, BuildingBlockClass bbClass, int phi):
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
            int i, j, k, alpha, phiHjInv, zeta, next_
            double[:, ::1] v = self.v, T = self.T
            int sumKhat = 0, goalSum = (d-1)*sigma % q
            double PsiVal = 0
            double minVi, columnMin, val

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
                    self.thetaHat[i] = j
            sumKhat += self.thetaHat[i]
            PsiVal += v[i, self.thetaHat[i]]

        # check shortcutting conditions
        if PsiVal >= t[0, sigma] - 1e-5:
            return False  # unconstrained solution already too large
        if sumKhat % q == (d-1)*sigma % q:
            self.insertCut(bbClass, hj, Nj, d, self.thetaHat, phi)
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
            self.thetaHat[d-1] = S[d-1, goalSum]
            next_ = (goalSum - self.thetaHat[d-1]) % q
            for i in range(d-2, -1, -1):
                self.thetaHat[i] = S[i, next_]
                next_ = (next_ - self.thetaHat[i]) % q
            self.insertCut(bbClass, hj, Nj, d, self.thetaHat, phi)
            return True
        return False

    cdef bint readSolution(self):
        cdef bint codeword = True
        cdef int i, k
        self.objectiveValue = self.model.ObjVal
        for i in range(self.blocklength):
            self.solution[i] = 0
            for k in range(1, self.q):
                if self.xVals[i, k] > 1e-5:
                    if self.solution[i] != 0:
                        self.solution[:] = .5  # error
                        return False
                    else:
                        self.solution[i] = k
        return True

    cdef void insertCut(self, BuildingBlockClass bbClass, np.int_t[::1] hj, np.intp_t[::1] Nj, int d, np.int_t[::1] theta, int phi):
        cdef:
            int  q = self.q
            double kappa = (d-1)*bbClass.vals[0, bbClass.sigma]
            int i, j, tki, hjPhi
            np.int_t[:, ::1] permutations = self.permutations, vals = bbClass.vals
        for i in range(d):
            hjPhi = hj[i]*phi % q
            for j in range(q-1):
                self.coeffs[(q-1)*i + j] = vals[theta[i], permutations[hjPhi, j + 1]]
                self.varInds[(q-1)*i + j] = (q-1)*Nj[i] + j
            kappa -= vals[0, theta[i]]
        self.model.fastAddConstr2(self.coeffs[:(q-1)*d], self.varInds[:(q-1)*d], g.GRB.LESS_EQUAL, kappa)

    cdef void diagonalize(self):
        """Perform gaussian elimination on the code's parity-check matrix to search for RPC cuts.
        """
        cdef int i, j
        cdef np.intp_t[:] sortIndices
        for i in range(self.blocklength):
            self.coeffs[i] = 0
            for j in range(self.q):
                if self.useEntropy:
                    self.coeffs[i] += self.xVals[i, j]*log2(self.xVals[i, j]) if self.xVals[i,j] > 1e-6 else 0
                else:
                    self.coeffs[i] += (self.xVals[i, j] - 1./self.q)**2
        sortIndices = np.argsort(self.coeffs[:self.blocklength])
        gaussianElimination(self.htilde, sortIndices, True, self.successfulCols, self.q)



    def params(self):
        ret =  OrderedDict(name=self.name)
        if len(self.grbParams):
            ret['gurobiParams'] = self.grbParams
        ret['gurobiVersion'] = '.'.join(str(v) for v in g.gurobi.version())
        if not self.RPC:
            ret['RPC'] = False
        if self.useEntropy:
            ret['useEntropy'] = True
        return ret
