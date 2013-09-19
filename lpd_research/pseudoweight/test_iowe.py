#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2013 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division

import iowe
import sys
from collections import OrderedDict
import os.path
import numpy as np



def removeDuplicates(path, K, codewords=False):
    theiowe = iowe.IOWE.fromFile(path, K, codewords=codewords)
    table = OrderedDict()
    for (a,b,c,d), e in zip(theiowe.itable, theiowe.ftable):
        if b == 0 and d == 0 and (b, a, d, c) in table:
            assert table[(b,a,d,c)] <= e
            print('del ',b,a,d,c)
            del table[b,a,d,c]
        table[a,b,c,d] = e
    itable = np.array(table.keys(), dtype=np.int)
    ftable = np.array(table.values(), dtype=np.double)
    return iowe.IOWE(itable, ftable, K)
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'complete':
            iowe.initlogplus()
            iowe.completeEstimation(*(sys.argv[2:-1] + [int(sys.argv[-1])]))
        else:
            outname = os.path.basename(sys.argv[1]).replace('.txt', '_nodups0.txt')
            theiowe = removeDuplicates(sys.argv[1], int(sys.argv[2]), codewords=False)
            theiowe.write(outname)

    else:
        iowe.initlogplus()
        print("initialized")
        for K in [208, 256, 320, 512]:
            print('**********************\nK={}'.format(K))
            #path = "/home/helmling/.Dropbox-encrypted/Dropbox/3DTC_Pseudo/PseudoIOWEs/pseudoIOWE_Outer_K100.txt"
            path = "/home/helmling/.Dropbox-encrypted/Dropbox/3DTC_Pseudo/PseudoIOWEs/pseudoIOWE_Outer_K{}_40_40_nodups.txt.bz2".format(K)
            theiowe = iowe.IOWE.fromFile(path, K)
            print("done reading table. size: {}".format(theiowe.length))
            pcc = iowe.concatenatedIOWE(theiowe, MAXW=50)
            pcc.write('pseudoIOWE_PCC_K{}_new.txt'.format(K))
            print('done computing PCC')
            inner = iowe.IOWE.fromFile('/home/helmling/.Dropbox-encrypted/Dropbox/3DTC_Pseudo/PseudoIOWEs/pseudoIOWE_Inner_K{}_55_55_nodups.txt.bz2'.format(K), K)
            #inner = iowe.IOWE.fromFile('/home/helmling/.Dropbox-encrypted/Dropbox/3DTC_Pseudo/PseudoIOWEs/pseudoIOWE_Inner_K{}.txt'.format(K), K)
            iowe.completeWE(pcc, inner, 'pseudoWE_3DTC_K{}_michael_nodups.txt'.format(K), MAXW=55)
            print('done computing WE')
            we = iowe.readWE('pseudoWE_3DTC_K{}_michael_nodups.txt'.format(K))
            result = iowe.estimate(we, .5)
            print('estimated awgn pseudoweight={}'.format(result))
            with open('estimation_K{}_michael_nodups.txt'.format(K), 'wt') as f:
                f.write(str(result))