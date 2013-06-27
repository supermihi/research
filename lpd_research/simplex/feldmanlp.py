# -*- coding: utf-8 -*-
# Copyright 2013 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdecoding.core import Decoder
from lpdecoding.decoders import lp_toolkit
import simplex
import numpy as np

class CustomFeldmanLPDecoder(Decoder):
    
    def __init__(self, code, name=None):
        Decoder.__init__(self, code)
        if name is None:
            name = "LP Decoder(simplex)"
        self.name = name
        self.A, self.b = lp_toolkit.forbiddenSetInequalities(code.parityCheckMatrix)
        
    def solve(self, hint=None, lb=1):
        z, x = simplex.primal01SimplexRevised(self.A, self.b, self.llrVector)
        self.objectiveValue = z
        self.solution = x
        
    def params(self):
        return OrderedDict(name=self.name)
        
if __name__ == "__main__":
    from lpdecoding import *
    from lpdecoding.decoders.feldmanlpdecoders import *
    #code = HammingCode(3)
    code = LinearCode("/home/helmling/Forschung/codez/Tanner_155_64.alist")
    decoder_ref = CplexLPDecoder(code)
    decoder_new = CustomFeldmanLPDecoder(code)
    #print(matrix.strBinary(decoder_new.A))
    #print(matrix.strBinary(decoder_new.b))
    np.savetxt("florian_A_groß.txt", decoder_new.A, '%d')
    np.savetxt("florian_b_groß.txt", decoder_new.b,  '%d') 
    chan =  AWGNC(snr=0.0, coderate=code.rate, seed=2198437)
    signalg = SignalGenerator(code, chan, False)
    logging.basicConfig(level=logging.DEBUG)
    for i in range(10):
        next(signalg)
    for i in range(1):
        print(i)
        llr = next(signalg)
        np.savetxt("florian_llr_groß.txt", llr)
        decoder_ref.decode(llr)
        decoder_new.decode(llr)
        if not np.allclose(decoder_ref.solution, decoder_new.solution):
            print(":(")
            print(decoder_ref.objectiveValue)
            print(decoder_new.objectiveValue)
            print(decoder_ref.solution)
            print(decoder_new.solution)
            raw_input()
        raw_input()

