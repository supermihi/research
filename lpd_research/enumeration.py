# -*- coding: utf-8 -*-
# Copyright 2011 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""A module for enumeration algorithms"""
from __future__ import print_function
import sys, subprocess, tempfile, os, collections
import numpy
from fractions import Fraction
def chertkovPolytope(matrix, file = sys.stdout):
    '''Outputs the fundamental cone, projected onto Σx_i=1, in H-representation. Input is the parity-check matrix.'''
    numberOfInequalities = matrix.sum() + matrix.shape[1] + 1
    file.write('cone\nH-representation\n')
    file.write('linearity 1 {0}\n'.format(numberOfInequalities))
    file.write('begin\n')
    file.write('{0} {1} rational\n'.format(numberOfInequalities, matrix.shape[1] + 1))
    for row in matrix.A:
        N_i = numpy.nonzero(row)[0]
        for i in N_i:
            ineq = row.copy()
            ineq[i] = -1
            file.write('0 ' + ' '.join(map(str, ineq)) + '\n')
    for var in range(matrix.shape[1]):
        z = numpy.zeros(matrix.shape[1] + 1, dtype = numpy.int)
        z[var+1] = 1
        file.write(' '.join(map(str, z)) + '\n')
    eq = numpy.ones(matrix.shape[1] + 1, dtype = numpy.int)
    eq[0] = -1
    file.write(' '.join(map(str, eq))+'\n')
    file.write('end\n')
    
def parseOutput(file, m):
    out = numpy.zeros(m, dtype = numpy.double)
    def stof(string):
        return float(Fraction(string))
    def l2(line):
        return numpy.square(map(stof, line.split()[1:])).sum()
    for line in file:
        if len(line) > 0 and line[:3] == b' 1 ':
            out[0] = l2(line)
            rows = 1
            break
    for row in range(1, m):
        out[row] = l2(next(file))
        if row % 10000 == 0:
            print(row)
    return out
        
def evaluateOutput(file, counter):
    best = float('Inf')
    iteration = 0
    for line in file:
        l = line.decode('utf-8').strip()
        if len(l) > 0 and l[0] == '1':
            iteration += 1
            vertex = l.split()[1:]
            for i, string in enumerate(vertex):
                if '/' in string:
                    num, den = map(float,string.split('/'))
                    vertex[i] = num/den
                else:
                    vertex[i] = float(string)
            pw = 1./numpy.square(vertex).sum()
            counter[round(pw,8)] += 1
            if pw < best:
                best = pw
                bestV = vertex
                print('[iteration {it}] best: {0}'.format(best, it = iteration))
            if iteration % 10000 == 0:
                print('[iteration {0}]'.format(iteration))
    print('total number of pseudocodewords: {0}'.format(iteration))
    return best, bestV

def pseudoEnumeration(matrix):
    file = tempfile.NamedTemporaryFile(suffix = '.ine', mode = 'wt', delete = False)
    chertkovPolytope(matrix, file)
    file.close()
    print(file.name)
    pipe = subprocess.Popen(('lrs', file.name), bufsize=0, stdout=subprocess.PIPE).stdout
    counter = collections.Counter()
    evaluateOutput(pipe, counter)
    os.remove(file.name)
    return counter
    
    