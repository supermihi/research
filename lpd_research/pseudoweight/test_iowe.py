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
import numpy as np



def removeDuplicates(theiowe, codewords=False, eiriksVersion=False):
    table = OrderedDict()
    for (a,b,c,d), e in zip(theiowe.itable, theiowe.ftable):
        if (not codewords) or (a == 0 and c == 0):
            table[a,b,c,d] = e
        if b == 0 and d == 0 and (a != 0 or c != 0) and (b,a,d,c) in table:
            assert table[(b,a,d,c)] <= e
            if eiriksVersion and table[(b,a,d,c)] == e:
                del table[a,b,c,d]
            else:
                del table[b,a,d,c]
    itable = np.array(table.keys(), dtype=np.int)
    ftable = np.array(table.values(), dtype=np.double)
    return iowe.IOWE(itable, ftable, K)

def chain(outer, inner, codewords=False, eiriksVersion=False):
    outerRed = removeDuplicates(outer, codewords, eirik)
    outerRed.write('pseudoIOWE_Outer_K{}_nodups20sep{}.txt'.format(outer.K, '_eirik' if eiriksVersion else ""))
    print('outer duplicates removed')
    pcc = iowe.concatenatedIOWE(outerRed)
    print('pcc computed')
    pcc.write('pseudoIOWE_PCC_K{}_nodups20sep{}.txt'.format(outer.K, '_eirik' if eiriksVersion else ""))
    innerRed = removeDuplicates(inner, codewords, eirik)
    innerRed.write('pseudoIOWE_Inner_K{}_nodups20sep{}.txt'.format(outer.K, '_eirik' if eiriksVersion else ""))
    print('inner duplicates removed')
    we = iowe.completeWE(pcc, innerRed)
    print('complete WE computed')
    we.write('pseudoWE_3DTC_K{}_20sep{}{}.txt'.format(outer.K, "_dmin" if codewords else "", '_eirik' if eiriksVersion else ''))
    print('complete WE written')
    sortd = we.toSortedPseudoweight()
    result = iowe.estimateAWGNPseudoweight(sortd, .5)
    print('estimated pseudoweight: {}'.format(result))

if __name__ == "__main__":
    K = int(sys.argv[1])
    outer = iowe.IOWE.fromFile(sys.argv[2], K)
    print('done reading outer')
    inner = iowe.IOWE.fromFile(sys.argv[3], K)
    print('done reading inner')
    codewords = bool(int(sys.argv[4]))
    eirik = bool(int(sys.argv[5]))
    if codewords:
        print('dmin case')
    if eirik:
        print('eiriks version')
    chain(outer, inner, codewords, eirik)