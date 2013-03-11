from branchbound.bnb cimport Node
from branchbound.myList cimport myDeque
from branchbound.problem cimport Problem
#from branchbound.queue import Queue
import heapq
from collections import deque

cdef class BranchMethod:
	cdef:
		public Node root
		public Problem problem
		public bint FirstSolutionExists
		public double lpTime
		public int moveCount
		public int fixCount
		public int unfixCount
		public double moveTime
	
	cdef void refreshActiveNodes(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	cdef void move(self, Node fromNode, Node toNode)
	cdef Node getActiveNode(self, Node activeOld)
	
	
cdef class BFSMethod(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists

	cdef void addNodes(self, Node node0, Node node1)
	cdef Node getActiveNode(self, Node activeOld)
	cdef void createNodes(self, int branchVariable, Node parent)
	
cdef class MyBFSMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
	
	cdef void addNodes(self, Node node0, Node node1)
	cdef Node getActiveNode(self, Node activeOld)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class BFSRandom(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)	
	
cdef class MyBFSRandom(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class BFSRound(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyBFSRound(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSMethod(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	
cdef class MyDFSMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	 

cdef class DFSRandom(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
	
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		 
cdef class MyDFSRandom(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
	
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSRound(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyDFSRound(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BBSMethod(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DSTMethod(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyDSTMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSandBBSMethod(BranchMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	cdef void refreshActiveNodes(self, Node activeOld)