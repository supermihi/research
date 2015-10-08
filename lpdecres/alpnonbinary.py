# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import gurobimh as gu
import numpy as np

from lpdec.decoders.gurobihelpers import GurobiDecoder
from lpdec.utils import Timer
from lpdec import gfqla
from lpdecres.nonbinary import BuildingBlockClass


class NonbinaryALPDecoder(GurobiDecoder):
    """Adaptive LP decoder for non-binary codes.
    """


    def __init__(self, code, name=None, gurobiParams=None, gurobiVersion=None):
        if name is None:
            name = 'NonbinaryALPDecoder'
        GurobiDecoder.__init__(self, code, name, gurobiParams, gurobiVersion, integer=False)


        self.matrix = code.parityCheckMatrix
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
        self.xVals = np.empty((code.blocklength, q))

    def setStats(self, stats):
        for stat in 'cuts', 'totalLPs', 'simplexIters', 'optTime':
            if stat not in stats:
                stats[stat] = 0
        GurobiDecoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
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
                for j in range(1, self.q):
                    self.xVals[i, j] = self.x[i, j].X
            for row in range(self.matrix.shape[0]):
                for bbClass in self.bbClasses:
                    for phi in range(1, self.q):
                        cutAdded |= self.cutSearch(row, bbClass, phi)
            if not cutAdded:
                self.mlCertificate = self.foundCodeword = self.readSolution()
                break

    def cutSearch(self, row: int, bbClass: BuildingBlockClass, phi) -> bool:
        """Non-binary cut search algorithm.

        Parameters
        ----------
        row : int
            Row index of parity-check matrix to test.
        bbClass : BuildingBlockClass
            Building block class to separate.
        phi : int from {1,...,q-1}
            Automorphism (permutation).
        """
        q = self.q
        Nj = self.Nj[row]
        hj = self.hj[row]
        d = Nj.size
        t = bbClass.vals
        sigma = bbClass.sigma
        # fill rotatedXvals matrix that contains x vals "rotated" according to hj and phi
        rotatedXvals = np.empty((d, q))
        rotatedXvals[:, 0] = 0
        for i in range(d):
            for j in range(1, q):
                rotatedXvals[i, j] = self.xVals[Nj[i], self.permutations[self.inv[phi*hj[i] % q], j]]
        # compute the v^j(x_i) and w_i^j
        v = np.empty((d, q))
        w = np.empty((d, q))
        for i in range(d):
            for j in range(q):
                v[i, j] = t[0, sigma] - t[0, j] - np.dot(t[j], rotatedXvals[i])
        for i in range(d):
            for j in range(q):
                w[i, j] = v[i, 0] - v[i, j]
        thetaHat = np.empty(d)
        VkTheta = np.zeros(q, dtype=np.int)  # entries \abs{V_k^\theta}
        PsiVal = 0
        for i in range(d):
            khat = np.argmax(w[i])
            thetaHat[i] = khat
            VkTheta[khat] += 1
            PsiVal += v[i, khat]

        if PsiVal >= t[0, sigma]:
            return False
        eta = sum(VkTheta[k]*(sigma-k) for k in range(q)) % q
        if eta == sigma:
            self.insertCut(bbClass, thetaHat, Nj, hj, phi)
            return True
        psiPlus = np.empty((d, q))
        for i in range(d):
            for j in range(1, q):
                psiPlus[i, j] = w[i, thetaHat[i]] - w[i, (thetaHat[i] + j) % q]
        irplus = -np.ones((d, q), dtype=np.int)
        for j in range(1, q):
            irplus[:, j] = np.argsort(psiPlus[:, j])
        self.minPsiPlusHelper(psiPlus, irplus)
        print(psiPlus)
        print(irplus)



    def insertCut(self, bbClass, theta, Nj, hj, phi):
        print('insert')
        d = Nj.size
        q = self.q
        coeffs = np.zeros((self.code.blocklength,self.q-1))
        for i in range(d):
            tki = bbClass.vals[theta[i, 1:]]
            coeffs[Nj[i], :] = tki[self.permutations[hj[i]*phi % q, 1:]]





    def params(self):
        ret = GurobiDecoder.params(self)
        return ret