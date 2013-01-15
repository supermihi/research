cimport numpy as np
from branchbound.nodeselection cimport BranchMethod

cdef class BranchAndBound:
    cdef:
        public object problem
        public Node root
        public double eps
        public BranchMethod selectionMethod
        public object branchRule
        public np.ndarray optimalSolution
        public double optimalObjectiveValue
        public int branchCount, unfixCount, fixCount, moveCount
        public double time
        public double lpTime
        public double lpVsAll
    
    cdef void updBound(self, Node node)


cdef class Node:
    cdef:
        public Node parent, child0, child1
        public np.ndarray solution
        public double objectiveValue
        public int branchVariable, branchValue
        public int depth
        public double lowerb, upperb