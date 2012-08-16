from lpdecoding.core cimport Decoder
cimport numpy as np
from lpdecoding.utils cimport StopWatch
from lpdecoding.codes.ctrellis cimport Trellis

cdef class CSPDecoder(Decoder):
    cdef public np.int_t k
    cdef public object constraints
    cdef public double lstsq_time, sp_time, cho_time, r_time
    cdef public int majorCycles, minorCycles, maxMajorCycles
    cdef object definingEncoder
    #  temporary variables
    cdef np.ndarray \
        direction, \
        space1, space2, space3, \
        R, \
        P, \
        S, Sfree, \
        w, \
        RHS, \
        X, \
        paths
    cdef int lenS
    cdef double current_ref
    cdef void NearestPointAlgorithm(self)
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result, np.int_t[:] path)
    cdef void resetData(self, np.double_t[:] initPoint) 
    cdef void updateData(self, double delta_r)
    cdef void calculateSolution(self)
    #cdef void updateData2(self, double delta_r)
    cdef int innerLoop(self)
    cdef StopWatch timer