# -*- coding: utf-8 -*-
# Copyright 2011-2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
from lpdecoding.core cimport Decoder, Code
from lpdecoding.utils cimport StopWatch
import logging, datetime, itertools
import lpdecoding.algorithms.pseudoweight as pwt
import numpy as np
cimport numpy as np
ctypedef np.double_t DTYPE_t
logging.basicConfig()
logger = logging.getLogger('supportsearch')
logger.setLevel(logging.DEBUG)

def findLowSupportPCWs(identifier, Decoder decoder, Code code, int runs=10):
    """Applies the second pseudocodeword search algorithm by Chertkov and Stepanov.

    The algorithm is described in the paper "Polytope of Correct (Linear Programming) Decoding and
    Low-Weight Pseudo-Codewords", Proc. 2011 IEEE Int. Symp. Inf. Theory, pp. 1654--1658.
    """

    start_time = datetime.datetime.utcnow()
    cdef:
        StopWatch timer = StopWatch()
        int n = code.blocklength
        np.ndarray[DTYPE_t, ndim=1] center = np.ones(n)
        double bestPW = np.inf, pw
        np.ndarray[DTYPE_t, ndim=1] bestPCW, bestSupportPCW, beta_k, beta_k_plus_one, rounded
        int steps = 0
        int iteration, i
        int bestSupport = code.blocklength
        double new_pw
    timer.start()
    for iteration in range(runs):
        pw = np.inf
        logger.debug('run {0}'.format(iteration))
        beta_k = center + np.random.standard_normal(n)
        beta_k = beta_k * (code.blocklength / beta_k.sum())
        for i in itertools.count(start=1):
            if i % 100 == 0:
                logger.warning('step {0}; pw={1}'.format(i, pw))
            steps += 1
            beta_k_plus_one = decoder.decode(beta_k - center)
            new_pw = (beta_k_plus_one.sum()**2) / np.square(beta_k_plus_one).sum()
            rounded = np.around(beta_k_plus_one, 8)
            if np.count_nonzero(rounded) < bestSupport and len(np.unique(rounded))>2:
                bestSupport = np.count_nonzero(rounded)
                bestSupportPCW = decoder.rescaledPseudoCodeword()
            if new_pw > pw - 1e-8:
                break
            pw = new_pw
            beta_k = beta_k_plus_one
        if new_pw < bestPW:
            bestPW = new_pw
            bestPCW = decoder.rescaledPseudoCodeword()
            logger.debug('[iteration {0}] best: {1} after {2} steps'.format(iteration, pw, i))
        else:
            logger.debug('[iteration {0}] {1} not better'.format(iteration, pw))
    logger.info('average convergence steps: {0}'.format(steps / runs))
    options = dict(runs=runs)
    stats = dict(totalLPs=steps)
    result = pwt.PseudoweightComputation(identifier, "low support search",
                                   code, decoder, bestSupport, bestSupportPCW,
                                   options, timer.stop(), stats,
                                   start_time, datetime.datetime.utcnow())
    result.support = bestSupport
    result.lowestSupportPCW = bestSupportPCW
    return result
