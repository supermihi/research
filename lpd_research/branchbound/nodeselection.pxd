from branchbound.bnb cimport Node
from branchbound.myList cimport myDeque
from branchbound.queue cimport Queue
#cimport heapq
#from collections cimport deque

cdef class BranchMethod:
	cdef:
		public Node root
		public object problem
		public bint FirstSolutionExists
		public double lpTime
		public int moveCount
		public int fixCount
		public int unfixCount
	
	cdef void refreshActiveNodes(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	cdef void move(self, Node fromNode, Node toNode)
	cdef Node getActiveNode(self, Node activeOld)
	
cdef class BFSMethod(BranchMethod):
	cdef:
		public Queue activeNodes
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
	
	
cdef class MyBFSRandom(BranchMethod):
	cdef:
		public myDeque activeNodes
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
	
	
cdef class MyDFSMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
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
	
	
cdef class MyDFSRound(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BBSMethod(BranchMethod):
	#cdef:
	#	public heap activeNodes
		
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
		public myDeque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	cdef void refreshActiveNodes(self, Node activeOld)