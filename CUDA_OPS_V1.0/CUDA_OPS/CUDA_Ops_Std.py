import math
import numpy as np
from numba import cuda, jit, complex64, int16, int64, float32, void
import timeit


@cuda.jit(void(complex64[:,:],complex64[:,:],complex64[:,:]),fastmath = True)
def MatMul(A,B,C):
    '''Calculate the matrix product of two matrices AB which are assumed to be of the right sizes
    Parameters:
    -----------
    A (complex64[:,:]) - First array in the product
    B (complex64[:,:]) - Second array in the product
    C (complex64[:,:]) - Array for storing the product of the multiplication
    Assumptions:
    ------------
    Block width is the number of columns
    Grid width is the number of rows
    '''

    #we assign each block a row of the output and each thread an element
    
    i = cuda.blockIdx.x #position of the block
    j = cuda.threadIdx.x #position of the thread in a block

    M = B.shape[0] #Number of columns in A and rows in B

    temp = 0
    for k in range(M):
        temp += A[i,k]*B[k,j]
    
    C[i,j] = temp



