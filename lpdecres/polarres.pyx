# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
# distutils: libraries = ["gurobi60"]
#
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport fabs
import gurobimh as g
cimport gurobimh as g


from lpdec.mod2la cimport gaussianElimination
from lpdec.decoders.base cimport Decoder
from lpdec.utils import Timer
from lpdec.codes.polar import PolarCode
from lpdec.polytopes import feldmanInequalities

logger = logging.getLogger('polarres')

cdef class AdaptivePolarLPDecoder(Decoder):
    """
    not documented yet
    """

    cdef public bint removeAboveAverageSlack, keepCuts, initWithSparseLP, sparseRPC, allZero
    cdef int nrFixedConstraints, solveCalls
    cdef public double minCutoff
    cdef public int removeInactive, maxRPCrounds
    cdef np.int_t[:,::1] hmat, htilde
    cdef public g.Model model
    cdef double[::1] diffFromHalf, longSolution, newSolution, setV
    cdef int[::1] Nj, fixes
    cdef public object timer
    cdef double chg1, chg2, chg3
    cdef int n, N

    def __init__(self, code, **params):
        assert isinstance(code, PolarCode)
        name = params.get('name', 'AdaptivePolarLPDecoder')
        Decoder.__init__(self, code=code, name=name)
        self.maxRPCrounds = params.get('maxRPCrounds', -1)
        self.minCutoff = params.get('minCutoff', 1e-5)
        self.removeInactive = params.get('removeInactive', 0)
        self.removeAboveAverageSlack = params.get('removeAboveAverageSlack', False)
        self.keepCuts = params.get('keepCuts', False)
        self.initWithSparseLP = params.get('initWithSparseLP', True)
        self.sparseRPC = params.get('sparseRPC', False)
        self.allZero = params.get('allZero', False)
        self.n = code.n
        self.N = code.blocklength
        self.model = g.Model('PolarALP')
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', 1)
        vars = []
        for i in range(self.N):
            vars.append(self.model.addVar(0, 1, 0, g.GRB_CONTINUOUS, name='x{}'.format(i)))
        fg = code.factorGraph()
        fg.sparsify()
        sparseH = fg.parityCheckMatrix().copy()
        if self.initWithSparseLP or self.sparseRPC:
            for i in range(self.N, len(fg.varNodes)):
                vars.append(self.model.addVar(0, 1, 0, g.GRB_CONTINUOUS, name='v{}'.format(i)))

        self.model.update()
        if self.initWithSparseLP:
            A, b = feldmanInequalities(sparseH, fundamentalCone=self.allZero)
            for i in range(len(b)):
                self.model.addConstr(g.LinExpr(A[i], vars), g.GRB_LESS_EQUAL, b[i])
            self.model.update()
        self.nrFixedConstraints = self.model.NumConstrs
        # initialize various structures
        self.hmat = sparseH if self.sparseRPC else code.parityCheckMatrix
        print(sparseH)
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.newSolution = np.empty(self.hmat.shape[1])
        self.longSolution = np.empty(self.hmat.shape[1])
        self.diffFromHalf = np.empty(self.hmat.shape[1])
        self.setV = np.empty(self.hmat.shape[1], dtype=np.double)
        self.Nj = np.empty(self.hmat.shape[1], dtype=np.intc)
        self.fixes = -np.ones(self.hmat.shape[1], dtype=np.intc)
        self.timer = Timer()


    cdef int cutSearchAlgorithm(self, bint originalHmat) except -3:
        """Runs the cut search algorithm and inserts found cuts. If ``originalHmat`` is True,
        the code-defining parity-check matrix is used for searching, otherwise :attr:`htilde`
        which is the result of Gaussian elimination on the most fractional positions of the last
        LP solution.
        :returns: The number of cuts inserted
        """
        cdef:
            double[::1] setV = self.setV
            int[::1] Nj = self.Nj
            double[::1] solution = self.longSolution
            np.int_t[:,::1] matrix
            int inserted = 0, row, j, ind, setVsize, minDistIndex, Njsize
            double minDistFromHalf, dist, vSum
        matrix = self.hmat if originalHmat else self.htilde
        for row in range(matrix.shape[0]):
            #  for each row, we build the set Nj = { j: matrix[row,j] == 1}
            #  and V = {j ∈ Nj: solution[j] > .5}. The variable setV will be of size Njsize and
            #  have 1 and -1 entries, depending on whether the corresponding index belongs to V or
            #  not.
            Njsize = 0
            setVsize = 0
            minDistFromHalf = 1
            minDistIndex = -1
            for j in range(matrix.shape[1]):
                if matrix[row, j] == 1:
                    Nj[Njsize] = j
                    if solution[j] > .5:
                        setV[Njsize] = 1
                        setVsize += 1
                    else:
                        setV[Njsize] = -1
                    if self.diffFromHalf[j] < minDistFromHalf:
                        minDistFromHalf = self.diffFromHalf[j]
                        minDistIndex = Njsize
                    elif minDistIndex == -1:
                        minDistIndex = Njsize
                    Njsize += 1
            if Njsize == 0:
                # skip all-zero rows (might occur due to Gaussian elimination)
                continue
            if setVsize % 2 == 0:
                #  V size must be odd, so add entry with minimum distance from .5
                setV[minDistIndex] *= -1
                setVsize += <int>setV[minDistIndex]
            vSum = 0 # left hand side of the induced Feldman inequality
            for ind in range(Njsize):
                if setV[ind] == 1:
                    vSum += 1 - solution[Nj[ind]]
                elif setV[ind] == -1:
                    vSum += solution[Nj[ind]]
            if vSum < 1 - self.minCutoff:
                # inequality violated -> insert
                inserted += 1
                self.model.fastAddConstr2(setV[:Njsize], Nj[:Njsize], b'<', setVsize - 1)
            if originalHmat and vSum < 1-1e-5:
                #  in this case, we are in the "original matrix" phase and would have a cut for
                #  insertion which is declined because of minCutoff. This implies that we don't
                #  have a codeword although this method may return 0
                self.foundCodeword = self.mlCertificate = False
        if inserted > 0:
            self._stats['cuts'] += inserted
            self.model.update()
        return inserted

    def setStats(self, object stats):
        statNames = ["cuts", "totalLPs", "ubReached", 'lpTime']
        for item in statNames:
            if item not in stats:
                stats[item] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int i, int val):
        if val == 1:
            self.model.setElementDblAttr(b'LB', i, 1)
        else:
            self.model.setElementDblAttr(b'UB', i, 0)
        self.fixes[i] = val

    cpdef release(self, int i):
        self.model.setElementDblAttr(b'LB', i, 0)
        self.model.setElementDblAttr(b'UB', i, 1)
        self.fixes[i] = -1

    def fixed(self, int i):
        """Returns True if and only if the given index is fixed."""
        return self.fixes[i] != -1

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        self.model.fastSetObjective(0, llrs.size, llrs)
        Decoder.setLLRs(self, llrs, sent)
        self.removeNonfixedConstraints()


    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef double newObjectiveValue
        cdef int i, iteration = 0, rpcrounds = 0
        self.chg1 = self.chg2 = self.chg3 = 1e6
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -INFINITY
        if self.sent is not None and ub == INFINITY:
            # calculate known upper bound on the objective from sent codeword
            ub = np.dot(self.sent, self.llrs) + 2e-6
        while True:
            iteration += 1
            with self.timer:
                self.model.optimize()
            self._stats['lpTime'] += self.timer.duration
            self._stats["totalLPs"] += 1
            if self.model.Status in (g.GRB_INFEASIBLE, g.GRB_INF_OR_UNBD):
                self.objectiveValue = INFINITY
                self.foundCodeword = self.mlCertificate = False
                break
            elif self.model.Status != g.GRB_OPTIMAL:
                raise RuntimeError('Unknown Gurobi status {}'.format(self.model.Status))
            newObjectiveValue = self.model.ObjVal
            if newObjectiveValue <= self.objectiveValue:
                # print(newObjectiveValue, self.objectiveValue, iteration)
                if iteration > 100000:
                    # prevent infinite loops in some rare cases where numerical issues cause
                    # non-increasing objective value after cut generation
                    print('cga: no improvement in iteration {}'.format(iteration))
                    break
            self.objectiveValue = newObjectiveValue
            if self.objectiveValue >= ub - 1e-6:
                # lower bound from the LP is above known upper bound -> no need to proceed
                self.objectiveValue = INFINITY
                self._stats['ubReached'] += 1
                self.foundCodeword = self.mlCertificate = False
                break
            self.model.fastGetX(0, self.hmat.shape[1], self.newSolution)
            if iteration >= 3:
                self.chg3 = self.chg2
            if iteration >= 2:
                self.chg2 = self.chg1
            self.chg1 = 0
            for i in range(self.newSolution.size):
                self.chg1 += fabs(self.newSolution[i] - self.longSolution[i])
            if self.chg1 + self.chg2 + self.chg3 < 1e-6:
                print('no chg {}'.format(iteration))
            self.longSolution[:] = self.newSolution
            integral = True
            for i in range(self.longSolution.size):
                if self.longSolution[i] < 1e-6:
                    self.longSolution[i] = 0
                elif self.longSolution[i] > 1-1e-6:
                    self.longSolution[i] = 1
                else:
                    integral = False
                self.diffFromHalf[i] = fabs(self.longSolution[i]-.499999)
            if self.removeInactive != 0 \
                    and self.model.NumConstrs - self.nrFixedConstraints >= self.removeInactive:
                self.removeInactiveConstraints()
            self.foundCodeword = self.mlCertificate = True
            if not self.initWithSparseLP:
                numCuts = self.cutSearchAlgorithm(True)
                if numCuts > 0:
                    # found cuts from original H matrix
                    continue
            if integral:
                break
            if rpcrounds >= self.maxRPCrounds and self.maxRPCrounds != -1:
                self.foundCodeword = self.mlCertificate = False
                break
            # search for RPC cuts
            xindices = np.argsort(self.diffFromHalf)
            gaussianElimination(self.htilde, xindices, True)
            numCuts = self.cutSearchAlgorithm(False)
            if numCuts == 0:
                self.mlCertificate = self.foundCodeword = False
                break
            rpcrounds += 1
        self.solution[:] = self.newSolution[:self.N]

    cdef void removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        cdef int i, removed = 0
        cdef double avgSlack, slack
        #  compute average slack of constraints all constraints, if only those above the average
        # slack should be removed
        if self.removeAboveAverageSlack:
            if self.model.NumConstrs == self.nrFixedConstraints:
                avgSlack = 1e-5
            else:
                slacks = self.model.get('Slack', self.model.getConstrs()[self.nrFixedConstraints:])
                avgSlack = np.mean(slacks)
        else:
            avgSlack = 1e-5  # some tolerance to avoid removing active constraints
        for constr in self.model.getConstrs()[self.nrFixedConstraints:]:
            if constr.Slack > avgSlack:
                removed += 1
                self.model.remove(constr)
        if removed:
            self.model.update()


    cdef void removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.
        """
        cdef g.Constr constr
        for constr in self.model.getConstrs()[self.nrFixedConstraints:]:
            self.model.remove(constr)
        self.model.update()
                   
    def params(self):
        params = OrderedDict(name=self.name)
        if self.maxRPCrounds != -1:
            params['maxRPCrounds'] = self.maxRPCrounds
        if self.minCutoff != 1e-5:
            params['minCutoff'] = self.minCutoff
        if self.removeInactive != 0:
            params['removeInactive'] = self.removeInactive
        if self.removeAboveAverageSlack:
            params['removeAboveAverageSlack'] = True
        if self.keepCuts:
            params['keepCuts'] = True
        if not self.initWithSparseLP:
            params['initWithSparseLP'] = False
        params['name'] = self.name
        return params
