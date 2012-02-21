#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2011 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from matplotlib import pyplot as plt
import sys

for i, filename in enumerate(sys.argv[1:]):
    plt.figure(i)
    with open(filename, 'rt') as file:
        for line in file:
            title = line
            line = next(file)
            hist = eval(line)
            plt.plot(hist, label = title)
    plt.legend()
    plt.title(filename)

plt.show()