cimport numpy as np

cdef class BuildingBlockClass:

    cdef readonly int q
    cdef readonly list shifts
    cdef readonly np.int_t[:, ::1] vals
    cdef readonly int sigma