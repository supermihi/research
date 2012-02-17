#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import modulation
import itertools
import numpy
from lpdecoding import LinearCode, matrix

code = LinearCode(filename = 'LDPC_N96_K48_GF64_BI.txt')
#patternList = list(itertools.product([0,1],repeat=6))
patternFile = '64qam_mapping.txt'
schema = modulation.ModulationSchema(patternFile = patternFile)
decoder = modulation.ModulationMLDecoder(code, schema)

#llr =  numpy.random.normal(0, 0.01, (code.blocklength // schema.bps)*(2**schema.bps - 1))
ldrs = modulation.parseLDRs('ldrs_N96_K48_GF64_64QAM_16db.txt')
for ldr in ldrs:
    decoder.llrVector = ldr
    decoder.solve_m()
    print(" ".join(map(str, decoder.solution)))
    