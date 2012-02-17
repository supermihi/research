# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from lpdecoding.core import Decoder
from lpdecoding.decoders import cplexhelpers
import cplex
import numpy
import itertools

class ModulationSchema(object):
    
    def __init__(self, bps = None, mapping = None, patternList = None, patternFile = None):
        object.__init__(self)
        if bps is not None:
            self.bps = bps
            self.mapping = mapping
        elif patternFile is not None:
            patternList = []
            with open(patternFile, 'rt') as pfile:
                for line in pfile:
                    pattern = tuple(map(int, line.split(';')[0][1:-1].split()))
                    print(pattern)
                    patternList.append(pattern)
        assert patternList is not None
        self.bps = len(patternList[0])
        self.mapping = {}
        for num, pattern in enumerate(patternList):
            for pos, bit in enumerate(pattern):
                if bit == 1:
                    if pos not in self.mapping:
                        self.mapping[pos] = set()
                    self.mapping[pos].add(num)
        print(self.mapping)
                    

def parseLDRs(filename):
    ldrList = []
    currentLDR = []
    with open(filename, 'rt') as ldrFile:
        for line in ldrFile:
            if line.isspace():
                if len(currentLDR) > 0:
                    ldrList.append(numpy.array(currentLDR, dtype=numpy.double))
                    currentLDR = []
            else:
                currentLDR.extend(map(float, line.split(",")[1:-1])) # strip away the "0" for symbol -1 and the " \n" at the end
    print('returned {0} ldrs'.format(len(ldrList)))
    return ldrList
                
    
    
class ModulationMLDecoder(Decoder):
    
    """A maximum likelihood decoder that takes the modulated channel symbols as input rather
    than the bitwise LLR values obtained from the demodulator."""
    
    def __init__(self, code, schema):
        Decoder.__init__(self)
        
        self.code = code
        self.matrix = self.code.parityCheckMatrix
        self.schema = schema
        
        self.blocklength_m = self.code.blocklength // self.schema.bps
        self.channelSymbols = 2**self.schema.bps
        self.cplex = cplex.Cplex()
        self.cplex.parameters.parallel.set(-1) # opportunistic parallel mode
        
        self.xsymb = [ "x_{0}_{1}".format(i, symb) for i, symb in itertools.product(range(self.blocklength_m), range(1, self.channelSymbols)) ]
        self.x = [ "x" + str(num) for num in range(self.matrix.shape[1]) ]
        self.z = [ "z" + str(num) for num in range(self.matrix.shape[0]) ]
        for varType in (self.xsymb, self.x):
            self.cplex.variables.add(
                    types=[self.cplex.variables.type.binary]*len(varType),
                    names= varType)
            
        self.cplex.variables.add(
            types=[self.cplex.variables.type.integer]*len(self.z),
            names= self.z)
        
        # constraints (1.1): only one symbol decision variable =1 for each transmitted symbol
        for i in range(self.blocklength_m):
            self.cplex.linear_constraints.add(
                names = [ "one_symbol_{0}".format(i)],
                rhs = [1],
                senses = "L",
                lin_expr = [ cplex.SparsePair(ind = ["x_{0}_{1}".format(i, symb) for symb in range(1, self.channelSymbols)], val = [1]*(self.channelSymbols-1))]
                )
        # constraints (1.2): connect codeword bits to channel symbols
        for j in range(self.code.blocklength):
            i = j // self.schema.bps
            pos = j % self.schema.bps # this could be replaced by an interleaver
            self.cplex.linear_constraints.add(
                names = [ "connect_x{0}".format(j)],
                rhs = [0],
                senses = "E",
                lin_expr = [cplex.SparsePair(ind = ["x_{0}_{1}".format(i, symb) for symb in self.schema.mapping[pos]] + ["x{0}".format(j)],
                                            val = [1]*len(self.schema.mapping[pos]) + [-1])]
                )
        # Hx = 2z constraints
        self.cplex.linear_constraints.add(
            names=[ "parity_check_" + str(num) for num in range(self.matrix.shape[0]) ])
        
        for cnt, row in enumerate(self.matrix):
            nonzero_indices = [ (self.x[i], float(row[i])) for i in range(row.size) if row[i] != 0 ]
            nonzero_indices.append( (self.z[cnt], -2 ) )
            self.cplex.linear_constraints.set_linear_components(
                "parity_check_{0}".format(cnt),
                zip(*nonzero_indices))
    
    def solve_m(self):
        assert len(self.llrVector) == len(self.xsymb)
        self.cplex.objective.set_sense(self.cplex.objective.sense.maximize)
        self.cplex.objective.set_linear( zip(self.xsymb, self.llrVector) )
        self.cplex.set_results_stream(None)
        self.cplex.set_warning_stream(None)
        self.cplex.set_error_stream(None)
        self.cplex.solve()
        cplexhelpers.checkKeyboardInterrupt(self.cplex)
        self.objectiveValue = self.cplex.solution.get_objective_value()
        self.solution = numpy.rint(self.cplex.solution.get_values(self.x)).astype(numpy.int)