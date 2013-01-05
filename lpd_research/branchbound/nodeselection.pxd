from branchbound.bnb cimport Node
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
		
	cdef void move(self, Node fromNode, Node toNode)	

	
cdef class BFSMethod(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BFSRandom(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BFSRound(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class DFSMethod(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	 
	cdef void addNodes(self, Node node0, Node node1)
	 
	cdef void createNodes(self, int branchVariable, Node parent)
	 
	 
cdef class DFSRandom(BranchMethod):
	#cdef:
	#	public deque activeNodes
	
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class DFSRound(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
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
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSandBBSMethod(BranchMethod):
	#cdef:
	#	public deque activeNodes
		
	cdef Node getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef void createNodes(self, int branchVariable, Node parent)
	
	cdef void refreshActiveNodes(self, Node activeOld)