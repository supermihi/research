#!/usr/bin/python
from __future__ import print_function, division
from lpdecoding.algorithms import pseudoweight
from lpdecoding import *
from lpdecoding.codes import interleaver
from lpdecoding.decoders.trellisdecoders import CplexTurboLikeDecoder
import itertools, random
from lpdecoding.codes.turbolike import ThreeDTurboCode

size = 320
snr = 1.8
if __name__ == '__main__':
    outerQPPs = interleaver.allQPPInterleavers(size, unique = True, QIonly = True)
    innerQPPs = interleaver.allQPPInterleavers(size//2, unique = True, QIonly = True) 
    allPairs = list(itertools.product(outerQPPs.keys(), innerQPPs.keys()))
    random.seed(179378946)
    
    randoms = random.sample(allPairs, 5)
    for (o1,o2), (i1, i2) in randoms:
        print('code: {}'.format( ((o1,o2), (i1, i2))))
        code = ThreeDTurboCode(size, o1, o2, i1, i2)
        decoder = CplexTurboLikeDecoder(code, ip = False)
        hist = list()
        pseudoweight.chertkovAlgorithm1(decoder, code, 300, snr, threshold = 0, histogram = hist)
        min_hist = [hist[0]]
        for pw in hist:
            min_hist.append(min(min_hist[-1], pw))
        print(min_hist)
