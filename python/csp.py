#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright 2012 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdecoding.codes import turbolike, trellis, interleaver
import numpy

SIZE = 40
code = turbolike.StandardTurboCode(encoder = trellis.LTE_Encoder(), interleaver = interleaver.lte_interleaver(SIZE), name = 'test')

mu = numpy.randn(SIZE)

done = False
while not done:
    # calculate mu
    code.enc_1.trellis.clearArcCosts()
    code.enc_2.trellis.clearArcCosts()
    cost1, path1 = algorithms.path.shortestPath(code.enc_1.trellis, mu, ...)
    path1, path2 = # shortes path in trellis1, trellis2 wrt. mu