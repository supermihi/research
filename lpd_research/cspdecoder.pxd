from lpdecoding.core cimport Decoder
cimport numpy
cdef class NDFDecoder(Decoder):
    cdef public numpy.int_t k
    cdef public object code, constraints
    cdef public double lstsq_time, sp_time, omg_time
    cdef public int majorCycles, minorCycles
    cdef object NPA(self,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] Y,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] P,
            numpy.ndarray[ndim=1, dtype=numpy.int_t] S,
            numpy.ndarray[ndim=1, dtype=numpy.double_t] w,
            numpy.ndarray[ndim=2, dtype=numpy.double_t] R)
    cdef numpy.double_t[:] solveScalarization(self, numpy.double_t[:] direction)