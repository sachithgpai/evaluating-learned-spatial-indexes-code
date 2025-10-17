cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def node_query_overlap_count(double[:,::1] queries, double lx, double ly, double hx, double hy):
    cdef Py_ssize_t Q = queries.shape[0]
    cdef int result = 0
    cdef Py_ssize_t idx
    for idx in range(Q):
        if(max(queries[idx][0],lx) < min(queries[idx][2],hx) and  max(queries[idx][1],ly) < min(queries[idx][3],hy)):
            result+=1
    return result