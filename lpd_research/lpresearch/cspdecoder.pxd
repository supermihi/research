from lpdecoding.core cimport Decoder
cimport numpy as np
from lpdecoding.utils cimport StopWatch
cdef class CSPDecoder(Decoder):
    cdef public np.int_t k
    cdef public object constraints
    cdef public double lstsq_time, sp_time, cho_time, r_time
    cdef public int majorCycles, minorCycles, maxMajorCycles
    #  temporary variables
    cdef np.ndarray \
        direction, \
        space1, space2, space3, \
        R, \
        P, \
        S, Sfree, \
        w, \
        RHS, \
        P_J, \
        X, \
        Y
    cdef int lenS
    cdef void NearestPointAlgorithm(self)
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result)
    cdef void resetData(self) 
    cdef void updateData(self, double delta_r)
    cdef void updateData2(self, double delta_r)
    cdef int innerLoop(self)
    cdef StopWatch timer