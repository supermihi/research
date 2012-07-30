from lpdecoding.core cimport Decoder
cimport numpy as np
cdef class NDFDecoder(Decoder):
    cdef public np.int_t k
    cdef public object constraints
    cdef public double lstsq_time, sp_time, cho_time, npa_time, r_time
    cdef public int majorCycles, minorCycles, maxMajorCycles
    cdef public np.ndarray X
    cdef int NPA(self,
            np.ndarray[ndim=1, dtype=np.double_t] Y,
            np.ndarray[ndim=2, dtype=np.double_t] P,
            np.ndarray[ndim=1, dtype=np.int_t] S,
            np.ndarray[ndim=1, dtype=np.uint8_t, cast=True] Sfree,
            np.ndarray[ndim=1, dtype=np.double_t] w,
            np.ndarray[ndim=2, dtype=np.double_t] R,
            int lenS,
            np.ndarray[ndim=1, dtype=np.double_t] X)
    cdef void solveScalarization(self, np.double_t[:] direction, np.double_t[:] result)