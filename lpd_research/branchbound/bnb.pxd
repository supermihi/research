cimport numpy as np

cdef class BranchAndBound:
    cdef:
        public object problem
        public Node root
        public double eps
        public object selectionMethod, branchRule
        public np.ndarray optimalSolution
        public double optimalObjectiveValue
        public int moveCount, branchCount, fixCount, unfixCount
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