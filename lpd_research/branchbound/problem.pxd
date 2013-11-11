cimport numpy as np
from lpdecoding.core cimport Decoder
from lpdecoding.codes.turbolike cimport TurboLikeCode
from lpresearch.cspdecoder cimport CSPDecoder

cdef class Problem:
    cdef public np.ndarray solution, hSolution
    cdef public double objectiveValue, hObjectiveValue
    
    cpdef setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c)
    
    cpdef int solve(self, double lb=?, double ub=?)
    
    cpdef fixVariable(self, int var, int val)
    cpdef unfixVariable(self, int var)
    
cdef class CSPTurboLPProblem(Problem):
    cdef public TurboLikeCode code
    cdef public CSPDecoder decoder
    
    cpdef setObjectiveFunction(self, np.ndarray[ndim=1, dtype=np.double_t] c)
    cpdef int solve(self, double lb=?, double ub=?)
    
    cpdef fixVariable(self, int var, int val)
    cpdef unfixVariable(self, int var)