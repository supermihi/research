#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import print_function
from lpdecoding.codes import blocktrellis
from lpdecoding.decoders import cplexhelpers
from lpdecoding.core import Decoder

import cplex, numpy


class CplexBlockTrellisDecoder(Decoder):
    
    def __init__(self, code, rowGroups, ip = False, name = None):
        Decoder.__init__(self)
        self.cat = 'B' if ip else 'C'
        self.code = code
        self.name = name
        hmat = code.parityCheckMatrix
        self.trellises = []
        for rows in rowGroups:
            self.trellises.append(blocktrellis.BlockTrellis(hmat[numpy.array(rows)]))
        self.cplex = cplex.Cplex()                    
        self.x = [ "x" + str(num) for num in range(code.blocklength) ] # one x-var for each column (codeword bit)
        self.cplex.variables.add(
            types=self.cat*code.blocklength,
            names= self.x)

        for i, trellis in enumerate(self.trellises):
            prefix = "t{0}".format(i)
            self._makeArcVars(trellis, prefix)
            self._addFlowConservation(trellis, prefix)
            self._makeEqualityConstraints(trellis, prefix)
            
        if ip:
            self.cplex.set_problem_type(self.cplex.problem_type.MILP)
        else:
            self.cplex.set_problem_type(self.cplex.problem_type.LP)
        self.cplex.set_results_stream(None)       

    def solve(self):
        self.cplex.objective.set_linear(zip(self.x, self.llrVector))
        self.cplex.solve()
        cplexhelpers.checkKeyboardInterrupt(self.cplex)
        self.objectiveValue = self.cplex.solution.get_objective_value()
        self.solution = numpy.array(self.cplex.solution.get_values(self.x))
    def _makeArcVars(self, trellis, prefix):
            """Create a binary IP variable for each arc of the given trellis. The prefix is used to individually
            name the variable."""
            for segment in trellis:
                for node in segment.values():
                    for arc in node.outArcs():
                        arc.lp_var = "{0}_p{1}_s{2}_b{3}".format(prefix, arc.pos, arc.state, arc.bit)
                        self.cplex.variables.add(lb = [0], ub = [1],
                                                 obj = [0], types = self.cat, names = [arc.lp_var])
        
    def _addFlowConservation(self, trellis, prefix):
        """Create the flow conservation constraints (including source and sink constraints) for the given trellis. The prefix should
        be unique by this trellis, since it is used for naming the constraints."""
        self.cplex.linear_constraints.add(names = ["{0}_src".format(prefix), "{0}_sink".format(prefix)],
                                          senses = ["E", "E"],
                                          rhs = [1.0, 1.0],
                                          lin_expr = [ cplex.SparsePair(ind = [arc.lp_var for arc in trellis[0][0].outArcs()],
                                                                        val = [1]*len(trellis[0][0].outArcs())),
                                                       cplex.SparsePair(ind = [arc.lp_var for arc in trellis[-1][0].inArcs()],
                                                                        val = [1]*len(trellis[-1][0].inArcs()))]
                                         )
        for segment in trellis[1:-1]:
            for node in segment.values():
                self.cplex.linear_constraints.add(
                    names = ["{0}_fc_p{1}_s{2}".format(prefix, node.pos, node.state)],
                    rhs = [0],
                    senses = ["E"],
                    lin_expr = [ cplex.SparsePair(ind = [arc.lp_var for arc in node.inArcs()] + [arc.lp_var for arc in node.outArcs()],
                                                  val = [1]*len(node.inArcs()) + [-1]*len(node.outArcs())) ],
                    )
    def _makeEqualityConstraints(self, trellis, prefix):
        for trellisPos,codePos in enumerate(trellis.indexes):
            oneArcs = [arc.lp_var for arc in trellis[trellisPos].allArcsOfBit(1)]
            self.cplex.linear_constraints.add(names = ['eq_{0}_{1}'.format(prefix, codePos)],
                                              senses = ['E'],
                                              rhs = [0],
                                              lin_expr = [cplex.SparsePair( ind = oneArcs + [self.x[codePos]], val = [1]*len(oneArcs) + [-1]) ] )
    
    def __str__(self):
        return "BlockTrellis[{0}]{1}".format(len(self.trellises), self.name if self.name else "")