import math
import numpy as np
from numba import jit, complex128, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import Bessel_C as Bess
from CylHar import WaveField_G as WF

@jit(void(complex128[:,:],complex128[:,:],complex128[:,:]),fastmath = True,nopython = True)
def MatMul(A,B,C):
    '''Calculate the matrix product of two matrices C= AB which are assumed to be of the right sizes
    Parameters:
    -----------
    A (complex128[:,:]) - First array in the product
    B (complex128[:,:]) - Second array in the product
    C (complex128[:,:]) - Array for storing the product of the multiplication
    Assumptions:
    ------------
    Block width is the number of columns
    Grid width is the number of rows and they are equal
    '''
    M = B.shape[0]

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp = 0
            for k in range(M):
                temp = temp + A[i,k]*B[k,j]
            C[i,j] = temp

@jit(complex128[:,:](complex128[:,:],float32[:],float32),nopython = True, fastmath = True)
def compute_wavefield(E_mat,k_arr,R):
    '''Compute the wavefield coefficients from a set of Q error signals in the frequency domain
        Parameters:
        -----------
        E_mat (complex128[:,:]) - matrix of values of the error signal of the form [e_1(k),e_2(k),....,e_Q(k)] 
        k_arr (float=32[:]) -  array of values of the wavenumber to represent the fourier domain
        R (float32) - Radius at which the error microphones are placed
        Returns:
        --------
        Alpha (complex128[:,:]) - Wave domain coefficients with row = wave number and column = m index
    '''
    N,Q = E_mat.shape #Pulls the relvant parameters for the matrix sizes

    M = np.int64(k_arr[-1]*R*np.e//2)+1 #We use the heuristic given in one of Thushara's papers to determine our general upper limit of summation
    #We now define relevant arrays in memory
    W = np.empty(shape = (Q,2*M+1),dtype=np.complex128) 
    for q in range(Q):
        for m in range(2*M+1):
            angle = 2*np.pi*((m-M)/Q)*q
            W[q][m] = math.cos(angle)-1j*math.sin(angle)

    #We define an array called A which will be used as a stepping stone to the whole alpha matrix
    A = np.empty(shape = (N,2*M+1),dtype=np.complex128)
    Alpha = np.empty(shape = (N,2*M+1),dtype=np.complex128)
    #We compute A as EW
    MatMul(E_mat,W,A)

    #We now compute the required array of bessel function values
    J = Bess.bessj(M,k_arr*R)

    #We now take the element-wise product of A and 1/QJ to give the alpha matrix
    for k in range(N):
        M_temp = M*k_arr[k]//k_arr[-1]+1
        
        for m in range(2*M+1):
            m_normalised = m-M
            if m_normalised<-M_temp:
                Alpha[k][m]= 0.0+0.0*1j
    
            elif m_normalised>M_temp:
                Alpha[k][m]= 0.0+0.0*1j
                
            else:
                if m_normalised >= 0: #Checks to deal with the negative m case
                    Alpha[k][m] = A[k][m]/(Q*J[k][m_normalised])
                else:
                    Alpha[k][m] = ((-1)**m_normalised)*A[k][m]/(Q*J[k][-m_normalised])
                
    return Alpha

    




    