# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division
import heapq
from lpdec.codes import BinaryLinearBlockCode
import numpy as np
from lpdec.codes.factorgraph import *


def bitChannelDegrading(W, mu, m, i):
    Q = W.degradingMerge(mu)
    i_binary = np.binary_repr(i, m)
    for j in range(m):
        if i_binary[j] == '0':
            W = Q.arikanTransform1()
        else:
            assert i_binary[j] == '1'
            W = Q.arikanTransform2()
        Q = W.degradingMerge(mu)
    return Q


class PolarBlock:

    def __init__(self, level, index, rows):
        """Represents a collection of bit-channels in level *level*, that means, it has :math:`2^{
        \mathrm{level}}` in- and outputs."""
        self.level = level
        self.index = index
        self.rows = rows

    @property
    def size(self):
        return 2 ** self.level


def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class PolarFactorGraph(FactorGraph):
    """Creates the factor graph of a polar code with block length :math:`n=2^m`."""
    def __init__(self, n):
        N = 2 ** n
        polarVars = np.empty((n+1, N), dtype=object)
        for layer in range(n+1):
            for i in range(N):
                polarVars[layer, i] = VariableNode('s{},{}'.format(layer, i))
                polarVars[layer, i].layer = layer
                polarVars[layer, i].index = i
        xVars = []
        polarChecks = np.empty((n, N), dtype=object)
        for layer in range(n):
            for i in range(N):
                polarChecks[layer, i] = CheckNode('c{},{}'.format(layer, i))
                polarChecks[layer, i].layer = layer
                polarChecks[layer, i].index = i
                polarChecks[layer, i].connect(polarVars[layer, i])
        # create the first N "bit-channels" that equal the underlying BMS channel
        channelLevels = [[PolarBlock(0, i, [i]) for i in range(N)]]
        for level in range(1, n+1):
            newBlocks = []
            for i, (upper, lower) in enumerate(pairwise(channelLevels[-1])):
                block = PolarBlock(level, i, upper.rows + lower.rows)
                for upper, lower in zip(upper.rows, lower.rows):
                    # create Z-structure
                    polarVars[level, upper].connect(polarChecks[level-1, upper])
                    polarVars[level, lower].connect(polarChecks[level-1, lower])
                    polarVars[level, lower].connect(polarChecks[level-1, upper])
                newBlocks.append(block)
            channelLevels.append(newBlocks)
        bitReversed = lambda num: int(np.binary_repr(num, n)[::-1], 2)
        self.u = [polarVars[n, bitReversed(i)] for i in range(N)]
        self.x = polarVars[0].tolist()
        for i in range(N):
            self.u[i].identifier = 'u{}'.format(i)
            self.x[i].identifier = 'x{}'.format(i)
        FactorGraph.__init__(self, polarVars.flatten().tolist(), polarChecks.flatten().tolist(),
                             x=self.x)
        self.polarVars = polarVars
        self.polarChecks = polarChecks

F = np.array([[1, 0], [1, 1]])


def polarG(n):
    G = np.ones((1,1))
    for i in range(n):
        G = np.kron(G, F)
    out = np.empty(G.shape, dtype=np.int)
    assert G.shape == (2**n, 2**n)
    for i in range(2**n):
        out[i] = G[int(np.binary_repr(i, n)[::-1], 2)]
    return out


def test(n):
    for i in range(2**n):
        print(bitChannelDegrading(W, 128, n, i).errorProbability())


def makePolarMatrix(n, frozenIndices):
    G = polarG(n)
    H = G.T[frozenIndices]
    return BinaryLinearBlockCode(parityCheckMatrix=H, name='PolarCode')

def constructPolarCode(W, n, mu, threshold=None, rate=None):
    N = 2**n
    P = np.zeros(N)
    for i in range(N):
        print('computing channel {}/{}'.format(i, N))
        channel = bitChannelDegrading(W, mu, n, i)
        P[i] = channel.errorProbability()
    if threshold:
        ind = [i for i in range(N) if P[i] > threshold]
    else:
        sortedP = np.argsort(P)
        targetLength = (1-rate)*N
        ind = sortedP[-targetLength:]
    return ind


if __name__ == '__main__':
    import polar_helpers
    W = polar_helpers.BMSChannel.BEC(.5)
    W8Probs = [0.30317055971460743, 0.069407030285054974, 0.042288321636850028, 0.00093408836310256615, 0.023764302403986744, 0.00028928759594364951, 0.00014639928205584219, 1.0717944050002485e-08]
    # import cProfile, pstats
    # cProfile.runctx('test()', globals(), locals(), 'Profile.prof')
    # s = pstats.Stats('Profile.prof')
    # s.strip_dirs().sort_stats('time').print_stats()
    import sys
    #test(int(sys.argv[1]))
    # code = constructPolarCode(W, 5, 128, rate=1/2)
    # from lpdec.decoders.branchcut import BranchAndCutDecoder
    # decoder = BranchAndCutDecoder(code, selectionMethod='mixed-/{}/{}/{}/{}'.format(120, 1, 1, .3),
    #     childOrder='10', branchMethod='mostFractional', iterParams=dict(reencodeOrder=2, iterations=10, reencodeRange=.1),
    #     lpParams=dict(removeInactive=50, maxRPCrounds=100, minCutoff=.3, variableFixing=True, keepCuts=True))
    # print(code.blocklength)
    # print(code.infolength)
    # print(code.rate)
    from lpdec import matrices
    #matrices.formatMatrix(code.parityCheckMatrix, filename='polar10_good.txt')
    n = 3
    fg = PolarFactorGraph(n)
    frozen = constructPolarCode(W, n, 128, rate=1/2)
    frozenVars = [ fg.u[i] for i in frozen ]
    import networkx as nx
    G = nx.Graph()
    for node in fg.varNodes:
        G.add_node(node, label=str(node))
    for node in fg.checkNodes:
        G.add_node(node, label=str(node))
    for check in fg.checkNodes:
        for var in check.neighbors:
            G.add_edge(check, var)
    import matplotlib.pyplot as plt
    pos = {}
    for node in G.nodes():
        pos[node] = (-node.layer-.5*isinstance(node, CheckNode), -node.index)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=400, nodelist=fg.varNodes)
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=400, nodelist=fg.checkNodes)
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=400, nodelist=frozenVars)
    nx.draw_networkx_labels(G,pos, font_color='w', font_size=10,font_family='sans-serif')
    nx.draw_networkx_edges(G, pos)

    plt.show()
    #print(decoder.minimumDistance())








