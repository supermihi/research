cimport numpy as np
from branchbound.nodeselection cimport SelectionMethod

cdef class BranchAndBound:
    cdef:
        public object problem
        public Node root
        public double eps
        public SelectionMethod selectionMethod
        public object branchRule
        public np.ndarray optimalSolution
        public double optimalObjectiveValue
        public int branchCount, unfixCount, fixCount, moveCount
        public double time
        public double lpTime
        public double lpVsAll
        public double getTime
        public double selectionTime
        public double refreshTime
        public double boundTime
        public double createTime
        public double addTime
        public double moveTime
        public double boundVsAll
        public double refreshVsAll
        public double getVsAll
        public double addVsAll
        public double createVsAll
        public double moveVsAll
        public double selectionVsAll
    
    


cdef class Node:
    cdef:
        public Node parent, child0, child1
        public np.ndarray solution
        public double objectiveValue
        public int branchVariable, branchValue, varToBranch
        public int depth
        public double lowerb, upperb