from branchbound.bnb cimport Node
from branchbound.myList cimport myDeque
from branchbound.problem cimport Problem
#from branchbound.queue import Queue
import heapq
from collections import deque

cdef class SelectionMethod:
	cdef:
		public Node root
		public Problem problem
		public object branchRule
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
	cdef void updBound(self, Node node)
	
	
cdef class BFSMethod(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists

	cdef void addNodes(self, Node node0, Node node1)
	cdef Node getActiveNode(self, Node activeOld)
	cdef void createNodes(self, int branchVariable, Node parent)
	
cdef class MyBFSMethod(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
	
	cdef void addNodes(self, Node node0, Node node1)
	cdef Node getActiveNode(self, Node activeOld)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class BFSRandom(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)	
	
cdef class MyBFSRandom(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class BFSRound(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyBFSRound(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSMethod(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	
cdef class MyDFSMethod(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	 

cdef class DFSRandom(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
	
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		 
cdef class MyDFSRandom(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
	
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSRound(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyDFSRound(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	
	
cdef class BBSMethod(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DSTMethod(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
		
cdef class MyDSTMethod(SelectionMethod):
	cdef:
		public myDeque activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	

cdef class DFSandBBSMethod(SelectionMethod):
	cdef:
		public object activeNodes
		public bint firstSolutionExists
		
	cdef Node getActiveNode(self, Node activeOld)
	cdef void addNodes(self, Node node0, Node node1)
	cdef void createNodes(self, int branchVariable, Node parent)
	cdef void refreshActiveNodes(self, Node activeOld)