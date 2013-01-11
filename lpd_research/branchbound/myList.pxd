from branchbound.bnb cimport Node

cdef class dequeNode:
	cdef:
		public Node pre, post, element
		
cdef class myDeque:
	cdef:
		public dequeNode first, last
		public int length
		
	cdef Node popleft(self)
	
	cdef Node pop(self)
	
	cdef void append(self, Node nextNode)
	