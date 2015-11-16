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
    decoders = []
    # decFL = StaticLPDecoder(code, ml=False)
    # print('decFL')
    # decoders.append(decFL)
    from lpdecres.alpternary import AdaptiveTernaryLPDecoder
    decTE = AdaptiveTernaryLPDecoder(code)
    print('decTE')
    decoders.append(decTE)
    from lpdecres.alpnonbinary import NonbinaryALPDecoder
    decNew = NonbinaryALPDecoder(code)
    print('decNew')
    decoders.append(decNew)
    decML = GurobiIPDecoder(code)
    print('decML')
    decoders.append(decML)
    simulation.ALLOW_DIRTY_VERSION = True
    simulation.ALLOW_VERSION_MISMATCH = True
    # simulation.DEBUG_SAMPLE = 1
    db.init('sqlite:///:memory:')
    channel = AWGNC(2, code.rate, seed=8374, q=3)
    simulator = Simulator(code, channel, decoders, 'ternary')
    simulator.maxSamples = 300
    simulator.maxErrors = 1000
    simulator.wordSeed = 1337
    simulator.outputInterval = 1
    simulator.dbStoreTimeInterval = 10
    simulator.revealSent = True
    simulator.concurrent = False
    import pstats, cProfile
    # cProfile.runctx("simulator.run()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()
    simulator.run()
    for decoder in decoders:
        pprint.pprint(decoder.stats())