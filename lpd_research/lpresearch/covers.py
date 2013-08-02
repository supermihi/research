# -*- coding: utf-8 -*-
# Copyright 2012 helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import numpy as np

def submatrix(pcm, x):
    positions = np.flatnonzero(x)
    pcm = pcm[:, positions]
    positions = np.flatnonzero(pcm.sum(1))
    print(positions)
    return pcm[positions, :] 