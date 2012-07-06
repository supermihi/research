#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
import numpy
import logging, os
from lpdecoding.codes import interleaver, trellis, turbolike
from cspdecoder import NDFDecoder
if __name__ == "__main__":
    from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
    from lpdecoding import simulate
    numpy.random.seed(1337)
    inter = interleaver.Interleaver(repr = [1,0,4,2,3] )
    encoder = trellis.TD_InnerEncoder() # 4 state encoder
    
    inter = interleaver.lte_interleaver(80)
    encoder = trellis.LTE_Encoder()
    code = turbolike.StandardTurboCode(encoder, inter)
    
    decoder = NDFDecoder(code)
    ref_decoder =CplexTurboLikeDecoder(code, ip = False)
    
    gen = simulate.AWGNSignalGenerator(code, snr = 1)
    for i in range(5):
        llr = next(gen)
        #llr = numpy.array([-0.2, -0.8,  1.2,  1.1,  1.2,  0.4,  0. ,  0.2, -0. , -0.9, -0.2, -1.3, -0.5,  0.8])
        logging.debug("llr vector: {0}".format(repr(llr)))
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        ref_decoder.decode(llr)
        tmp = os.times()
        print('ref decoding time: {}'.format(tmp[0] + tmp[2] - time_a))
        print('real: {0}'.format(ref_decoder.objectiveValue))
        tmp = os.times()
        time_a = tmp[0] + tmp[2]
        decoder.decode(llr)
        tmp = os.times()
        print('CSP total time: {0}'.format(tmp[0] + tmp[2] - time_a))
        print('solution: {0}'.format(decoder.objectiveValue))
        #print('X: {0}'.format(decoder.X))
        logging.debug('real solution: {0}'.format(ref_decoder.objectiveValue))
