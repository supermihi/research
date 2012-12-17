from branchbound.bnb import Node

cdef class BranchMethod:
	cdef:
		public Node root
		public object problem
		public bool FirstSolutionExists
		public double lpTime
	
	cdef (int, int, int) refreshActiveNodes(self, Node activeOld)
		
	cdef (int, int) move(self, Node fromNode, Node toNode)	

	
cdef class BFSMethod(BranchMethod):
	cdef:
		public deque activeNodes
		
	cdef (Node, int, int) getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	cdef (int, int) createNodes(self, int branchVariable, Node parent)
	
	
cdef class BFSRandom(BranchMethod):
	cdef:
		public deque activeNodes
		
	cdef (Node, int, int) getActiveNode(self, Node activeOld)
	
	cdef void addNodes(self, Node node0, Node node1)
	
	def (int, int) createNodes(self, int branchVariable, Node parent)