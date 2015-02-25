# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation


if __name__ == '__main__':
    from lpdec.imports import *
    code = TernaryGolayCode()
    code = NonbinaryLinearBlockCode(
        parityCheckMatrix='~/papers/ternarycodes/HmatrixOptRMCodes_n27_k10_d9_best.txt')
    print(code.parityCheckMatrix)
    decFL = FlanaganLPDecoder(code, ml=False)
    from lpdecres.alpternary import AdaptiveTernaryLPDecoder
    decTE = AdaptiveTernaryLPDecoder(code)
    simulation.ALLOW_DIRTY_VERSION = True
    simulation.ALLOW_VERSION_MISMATCH = True
    #simulation.DEBUG_SAMPLE = 8
    db.init('sqlite:///:memory:')
    channel = AWGNC(0, code.rate, seed=8374, q=3)
    simulator = Simulator(code, channel, [decTE], 'ternary')
    simulator.maxSamples = 1000
    simulator.maxErrors = 100
    simulator.wordSeed = 1337
    simulator.outputInterval = 1
    simulator.dbStoreTimeInterval = 10
    simulator.revealSent = True
    simulator.concurrent = False
    simulator.run()
    print(decTE.stats())
    print(decFL.stats())