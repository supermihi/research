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
    return iowe.IOWE(itable, ftable, theiowe.K)


def nodupsPath(path, eiriksVersion):
    name, suffix = path.split('.', 1)
    insert = '_nodups' + '_eirik' if eiriksVersion else ''
    return name + insert + '.' + suffix

def chain(K, outerPath, innerPath, codewords=False, eiriksVersion=False, stopping=False):
    maxw1 = 0 if stopping else 50
    maxw2 = 192 if stopping else 50
    outer = iowe.IOWE.fromFile(outerPath, K, stopping=stopping)
    print('done reading outer')
    if stopping:
        outerRed = outer
    else:
        outerRed = removeDuplicates(outer, codewords, eirik)
        outerRed.write(nodupsPath(outerPath, eiriksVersion))
        print('outer duplicates removed')
    inner = iowe.IOWE.fromFile(innerPath, K, stopping=stopping)
    print('done reading inner')
    if stopping:
        innerRed = inner
    else:
        innerRed = removeDuplicates(inner, codewords, eirik)
        innerRed.write(nodupsPath(innerPath, eiriksVersion))
        print('inner duplicates removed')
    pcc = iowe.concatenatedIOWE(outerRed, MAXW1=maxw1, MAXW2=maxw2)
    print('pcc computed')
    firstPart = outerPath.split('_', 1)[0]
    pcc.write(firstPart + '_PCC_K{}_nodups_{}.txt.bz2'.format(K, '_eirik' if
        eiriksVersion else ""))
    we = iowe.completeWE(pcc, innerRed, MAXW1=maxw1, MAXW2=maxw2)
    print('complete WE computed')
    name = ('stopipng' if stopping else 'pseudo') + '_3DTC_K{}_'.format(K)
    we.write(name + '{}{}.txt.bz2'.format('_dmin' if codewords else "",
                                          '_eirik' if eiriksVersion else ''))
    print('complete WE written')
    sortd = we.toSortedPseudoweight()
    result = iowe.estimateAWGNPseudoweight(sortd, .5)
    print('estimated weight: {}'.format(result))

if __name__ == "__main__":
    K = int(sys.argv[1])
    codewords = bool(int(sys.argv[4]))
    eirik = bool(int(sys.argv[5]))
    stopping = False
    if len(sys.argv) > 6:
        stopping = bool(int(sys.argv[6]))
        print('stopping={}'.format(stopping))
    if codewords:
        print('dmin case')
    if eirik:
        print('eiriks version')
    chain(K, sys.argv[2], sys.argv[3], codewords, eirik, stopping)