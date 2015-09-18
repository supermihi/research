# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import gurobipy as g
import itertools
d = 7
p = 5
b = 1


def proveQ5():

    model = g.Model()
    model.setParam('OutputFlag', 0)
    xVars = {}
    aVars = {}
    for i in range(1, p):
        for j in range(d):
            aVars[i, j] = model.addVar(vtype=g.GRB.CONTINUOUS, lb=-g.GRB.INFINITY, name='a{},{}'.format(i, j))
    model.update()
    for ijs in itertools.product(list(range(p)), repeat=d-1):
        last = (1 - sum(ijs)) % p
        theSum = g.quicksum(aVars[ijs[j], j] for j in range(d-1) if ijs[j] != 0)
        if last != 0:
            theSum += aVars[last, d-1]
        model.addConstr(theSum, g.GRB.GREATER_EQUAL, b)

    numProblems = 0
    for i, xs in enumerate(xCombinations()):
        #print(xs)
        model.setObjective(g.quicksum(aVars[xs[j], j] for j in range(d)))
        model.optimize()
        if model.Status != g.GRB.OPTIMAL:
            print(model.Status)
            raise RuntimeError()
        elif model.ObjVal <= b + 1e-6:
            numProblems += 1
            print('i={}'.format(i))
            #print(xs)
            # for i in range(1, p):
            #     for j in range(d):
            #         print('a_{},{} = {}'.format(i, j, aVars[i, j].X))
            print('obj={}'.format(model.ObjVal))
            #raise RuntimeError()
    print('num Probs = ', numProblems)

def xCombinations():
    for xs in itertools.product(list(range(1, p)), repeat=d-1):
        #yield xs
        xd = (1 - sum(xs)) % p
        if xd != 0:
            yield xs + (xd,)


if __name__ == '__main__':
    proveQ5()
