from branchbound.bnb cimport Node
from branchbound.myList cimport myDeque
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
		public myDeque activeNodes
		
	#cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	cdef Node getActiveNode(self, Node activeOld)
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BFSRandom(BranchMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BFSRound(BranchMethod):
	cdef:
		public myDeque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class DFSMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	 
	cdef void addNodes(self, Node node0, Node node1)
	 
	cdef void createNodes(self, int branchVariable, Node parent)
	 
	 
cdef class DFSRandom(BranchMethod):
	cdef:
		public myDeque activeNodes
	
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class DFSRound(BranchMethod):
	cdef:
		public myDeque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BBSMethod(BranchMethod):
	#cdef:
	#	public heap activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class DSTMethod(BranchMethod):
	cdef:
		public myDeque activeNodes
		
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