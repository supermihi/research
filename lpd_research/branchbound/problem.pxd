cimport numpy as np
from lpdecoding.core cimport Decoder
from lpdecoding.codes.turbolike cimport TurboLikeCode

cdef class Problem:
    cdef public np.ndarray solution
    cdef public double objectiveValue
    
    cdef solve(self)
    
cdef class CSPTurboLPProblem(Problem):
    cdef public TurboLikeCode code
    cdef public Decoder decoder
    
    cpdef setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c)
    cdef solve(self)