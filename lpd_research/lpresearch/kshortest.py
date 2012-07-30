# -*- coding: utf-8 -*-
# Copyright 2012 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdecoding.core import Decoder
from lpdecoding.decoders.trellisdecoders import KShortestPathDecoder
from cspdecoder import NDFDecoder

class KSP2Decoder(Decoder):
    
    def __init__(self, code, maxK = None):
        self.cspdec = NDFDecoder(code)
        self.kspdec = KShortestPathDecoder(code, maxK = maxK)
        self.code = code
    
    def solve(self):
        self.cspdec.decode(self.llrVector)
        for encoder in self.code.encoders:
            for arc in encoder.trellis.allArcs():
                arc.orig_cost = arc.cost
        X = self.cspdec.X
        #print(X)
        for encoder in self.code.encoders:
            trellis = encoder.trellis
            for segment in trellis:
                for node in segment:
                    for arc in node.outArcs():
                        arc.cost *= X[-1]
                        if arc.infobit and segment.g_input != 0:
                            arc.cost += X[segment.g_input_index]*segment.g_input
                        if arc.parity and segment.g_parity != 0:
                            arc.cost += X[segment.g_parity_index]*segment.g_parity
        self.kspdec.objectiveValue = float("inf")
        self.kspdec.runKSP()
        
        self.objectiveValue = self.kspdec.objectiveValue
        self.solution = self.kspdec.solution
    
    def __str__(self):
        return "KSP2Decoder(maxK={0})".format(self.kspdec.maxK) if self.kspdec.maxK else "KSP2Decoder"
                    
        