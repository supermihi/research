cimport numpy as np
from lpdecoding.core cimport Code, Decoder

cdef class Problem:
    cdef public np.ndarray solution
    cdef public double objectiveValue
    
    cpdef solve(self)
    
cdef class CSPTurboLPProblem(Problem):
    cdef public Code code
    cdef public Decoder decoder
    
    cpdef setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c)
    cpdef solve(self)