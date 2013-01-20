cimport cqueue

cdef class Queue:
	cpdef append(self, int value)
	cdef extend(self, int* values, size_t count)
	cpdef int peek(self) except? -1
	cdef int pop(self) except? -1
	cdef int popleft(self) except? -1
	
	