#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2011 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from lpdecoding import *
from lpdecoding.codes.turbolike import ThreeDTurboCode
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
from lpdecoding.algorithms.pseudoweight import *
import logging, os ,pickle
logging.basicConfig(level = logging.INFO)


code = ThreeDTurboCode(128, 19, 32, 7, 32)
decoder1 = CplexTurboLikeDecoder(code, ip = False)
decoder2 = CplexTurboLikeDecoder(code, coneProjection = True, ip = False)

runs = 1000
hist1 = []
hist2 = []

tmp = os.times()
time_a = tmp[0] + tmp[2]
bestPW1 = chertkovAlgorithm1(decoder1, code, runs = 1000, histogram = hist1)[0]
with open('chert1_hist.pkl', 'wb') as file:
    pickle.dump(hist1, file)
print('best chert1: {0}'.format(bestPW1))
print('count best chert1: {0}'.format(hist1.count(bestPW1)))
tmp = os.times()
time1 = tmp[0] + tmp[2] - time_a
print('time chert1: {0}'.format(time1))
print('relative efficiency chert1: {0}'.format(hist1.count(bestPW1)/time1))

tmp = os.times()
time_a = tmp[0] + tmp[2]
bestPW2 = chertkovAlgorithm2(decoder2, code, runs = 1000, histogram = hist2)[0]
with open('chert2_hist.pkl', 'wb') as file:
    pickle.dump(hist2, file)
print('best chert2: {0}'.format(bestPW2))
print('count best chert2: {0}'.format(hist2.count(bestPW2)))
tmp = os.times()
time2 = tmp[0] + tmp[2] - time_a
print('time chert2: {0}'.format(time2))
print('relative efficiency chert2: {0}'.format(hist2.count(bestPW2)/time2))
