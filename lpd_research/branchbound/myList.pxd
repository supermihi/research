from branchbound.bnb cimport Node

cdef class dequeNode:
	cdef:
		public Node element
		public dequeNode pre, post
		
cdef class myDeque:
	cdef:
		public dequeNode first, last
		public int length
		
	cdef Node popleft(self)
	
	cdef Node pop(self)
	
	cdef void append(self, Node nextNode)
	