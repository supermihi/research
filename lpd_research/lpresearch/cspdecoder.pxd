from lpdecoding.core cimport Decoder
cimport numpy as np
from lpdecoding.utils cimport StopWatch
from lpdecoding.codes.trellis cimport Trellis

cdef class CSPDecoder(Decoder):
    cdef public object constraints
    cdef public double lstsq_time, sp_time, cho_time, r_time, setcost_time
    cdef public int majorCycles, minorCycles, maxMajorCycles, blocklength
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
        codewords, paths
    cdef np.int_t k, lenS
    cdef bint measureTimes, keepLP
    cdef int heuristic
    cdef int numEncoders
    cdef double current_ref
    cdef int NearestPointAlgorithm(self)
    cdef int solveScalarization(self, np.double_t[:] direction,
                                 np.double_t[:] result, np.double_t[:] codeword,
                                 np.int_t[:,:] paths=?)
    cdef void resetData(self, np.double_t[:] initPoint) 
    cdef void updateData(self, double delta_r)
    cdef void calculateSolution(self)
    cdef int innerLoop(self)
    cdef StopWatch timer
    # heuristic solution and objective value
    cdef public double hObjectiveValue
    cdef public np.ndarray hSolution