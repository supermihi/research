# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import pprint

if __name__ == '__main__':
    from lpdec.imports import *
    code = TernaryGolayCode()
    code = NonbinaryLinearBlockCode(
        parityCheckMatrix='~/UNI/ternarycodes/HmatrixOptRMCodes_n27_k10_d9_best.txt')
    #code = TernaryGolayCode()
    code = NonbinaryLinearBlockCode(parityCheckMatrix='Nonbinary_PCM_GF5_155_93.txt')
    code = NonbinaryLinearBlockCode(parityCheckMatrix='Nonbinary_PCM_GF7_155_93.txt')
    from lpdec.codes import random
    #code = random.makeRandomCode(96, 48, .5, q=5, seed=12345)
    decoders = []
    # decFL = StaticLPDecoder(code, ml=False)
    print('decFL')
    # decoders.append(decFL)
    decCas = StaticLPDecoder(code, cascade=True, ml=False)
    decoders.append(decCas)
    from lpdecres.alpternary import AdaptiveTernaryLPDecoder
    if code.q == 3:
        decTE = AdaptiveTernaryLPDecoder(code)
        print('decTE')
        decoders.append(decTE)
    from lpdecres.alpnonbinary import NonbinaryALPDecoder
    # decNew = NonbinaryALPDecoder(code, RPC=True, name='NonbinaryALP+RPC')
    # decoders.append(decNew)

    # decNewEnt = NonbinaryALPDecoder(code, RPC=True, useEntropy=True, name='NonbinaryALP+RPCe')
    # decoders.append(decNewEnt)
    decT1 = NonbinaryALPDecoder(code, RPC=True, useEntropy=True, onlyT1=True, name='NonbinaryALP+RPCeT1')
    decoders.append(decT1)
    decNewPlain = NonbinaryALPDecoder(code, RPC=False, name='NonbinaryALP')
    decoders.append(decNewPlain)
    decML = GurobiIPDecoder(code, gurobiParams='2')
    print('decML')
    #decoders.append(decML)
    simulation.ALLOW_DIRTY_VERSION = True
    simulation.ALLOW_VERSION_MISMATCH = True
    # simulation.DEBUG_SAMPLE = 1
    db.init('sqlite:///:memory:')
    channel = AWGNC(15, code.rate, seed=8374, q=code.q)
    simulator = Simulator(code, channel, decoders, 'ternary')
    simulator.maxSamples = 1000
    simulator.maxErrors = 1000
    simulator.wordSeed = 1337
    simulator.outputInterval = 1
    simulator.dbStoreTimeInterval = 10
    simulator.revealSent = True
    simulator.concurrent = False
    import pstats, cProfile
    cProfile.runctx("simulator.run()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats(10)
    # simulator.run()
    # for decoder in decoders:
    #     pprint.pprint(decoder.stats())