from lpdecoding.core cimport Decoder
cimport numpy as np
from lpdecoding.utils cimport StopWatch
cdef class CSPDecoder(Decoder):
    cdef public np.int_t k
    cdef public object constraints
    cdef public double lstsq_time, sp_time, cho_time, r_time
    cdef public int majorCycles, minorCycles, maxMajorCycles
    cdef public np.ndarray X
    cdef np.ndarray direction, space1, space2, space3, R, P, S, Sfree, w
    cdef int lenS
    cdef void NPA(self, np.ndarray[ndim=1, dtype=np.double_t] Y, np.ndarray[ndim=1, dtype=np.double_t] X)
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result)
    cdef void updateR(self)
    cdef StopWatch timer