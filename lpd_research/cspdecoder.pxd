from lpdecoding.core cimport Decoder
cdef class NDFDecoder(Decoder):
    cdef public int k
    cdef public object code, constraints
    cdef public double lstsq_time, sp_time, omg_time
    cdef public int majorCycles, minorCycles