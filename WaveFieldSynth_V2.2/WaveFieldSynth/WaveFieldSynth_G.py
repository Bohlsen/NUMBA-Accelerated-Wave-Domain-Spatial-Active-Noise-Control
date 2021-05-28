import math
import numpy as np
from numba import cuda, complex128, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import Bessel_G as Bess
from CylHar import WaveField_G as WF
from WaveFieldSynth import SoundFields

#Imported and edited from the CUDA_Ops_Std file (data types needed to be changed for eager compilation)
@cuda.jit(void(complex128[:,:],complex128[:,:],complex128[:,:]),fastmath = True)
def MatMul(A,B,C):
    '''Calculate the matrix product of two matrices AB which are assumed to be of the right sizes
    Parameters:
    -----------
    A (complex128[:,:]) - First array in the product
    B (complex128[:,:]) - Second array in the product
    C (complex128[:,:]) - Array for storing the product of the multiplication
    Assumptions:
    ------------
    Block width is the number of columns in C
    Grid width is the number of rows in C
    '''

    #we assign each block a row of the output and each thread an element
    
    i = cuda.blockIdx.x #position of the block in the grid and the row of the A matrix 
    j = cuda.threadIdx.x #position of the thread in a block and the column of the A matrix


    V = B.shape[0] #Number of columns in A and rows in B

    temp = 0.0+0.0*1j
    for k in range(V):
        temp = temp + A[i,k]*B[k,j]#temp = temp + 1e10*A[i,k]*B[k,j] #scaling deals with values being too small for CUDA to handle
       
    C[i,j] = temp#C[i,j] = temp*1e-10
   

@cuda.jit(void(int64,int64,complex128[:,:]),fastmath = True)
def compute_W(Q,M,W_out):
    '''Computes the elements of the W matrix on GPU
        Parameters:
        -----------
        Q (int64) - Number of microphones
        M (int64) -  Maximum index in the truncated cylindrical harmonics expansion
        W_out (complex128[:,:]) - Qx(2M+1) array of complex128 passed for output
        Assumptions:
        ------------
        Q is assumed to never be greater that 1024 as Q will be the number of threads per block and so must be less than 1024
        M is the maximum m value in the cylindrical harmonics expansion and 2M+1 is assumed to be the number of blocks per grid
    '''
    #Pull the indexes
    q = cuda.threadIdx.x
    m = cuda.blockIdx.x

    if m<2*M+1:
        if q<Q:
            angle = 2*np.pi*((m-M)/Q)*q
            W_out[q][m] = math.cos(angle)-1j*math.sin(angle)

@cuda.jit(void(complex128[:,:],complex128[:,:],float64[:,:],float32[:],int64,int64,int64),fastmath = True)
def compute_alpha(Alpha,A,J,k_arr,Q,N,M):
    '''Compute the Alpha matric from the A and J matrices
    Parameters:
    -----------
    Alpha (complex128[:,:]) - Matrix of the alpha values in the form Alpha[n][m] = alpha_m(k_n) passed for output
    A (complex128[:,:]) - The A Matrix calculated as the EW product
    J (float64[:,:]) - The matrix of bessel function values
    Q (int64) - Number of microphones
    N (int64) - Number of data points in the frequency domain
    M (int64) - Maximum index in the truncated cylindrical harmonics expansion
    Assumptions:
    Grid Width is assumed to be at least N
    Block Width is assumed to be at least 2M+1
    '''
    k = cuda.blockIdx.x
    m = cuda.threadIdx.x

    M_temp = M*k_arr[k]//k_arr[-1]+1
    m_normalised = m-M #moves into the doman [-m,m]

    if m_normalised<-M_temp:
        Alpha[k][m]= 0.0+0.0*1j
    
    elif m_normalised>M_temp:
        Alpha[k][m]= 0.0+0.0*1j
    
    else:
        temp1 = 0.0+0.0*1j
        temp2 = 0.0+0.0*1j
        temp3 = 0.0+0.0*1j
        if m_normalised >= 0: #Checks to deal with the negative m case
            temp1 = A[k,m]
            temp2 = Q*J[k,m_normalised]#computes the wave domain coefficients
            temp3 = temp1/temp2
        else:
            temp1 = ((-1)**(m_normalised))*A[k,m]
            temp2 = Q*J[k,-m_normalised]
            temp3 = temp1/temp2 #computes the wave domain coefficients
        Alpha[k][m] = temp3

    
        

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
    W_dev = cuda.device_array(shape= (Q,2*M+1),dtype=np.complex128)

    #We now compute the W array
    compute_W[2*M+1,Q](Q,M,W_dev)
    
    #We define an array called A which will be used as a stepping stone to the whole alpha matrix\
    A_dev = cuda.device_array(shape = (N,2*M+1),dtype=np.complex128)
    
    #We compute A as EW
    E_dev = cuda.to_device(E_mat)
    MatMul[N,2*M+1](E_dev,W_dev,A_dev)

    #We now compute the required array of bessel function values
    J = Bess.bessj(M,k_arr*R)
    J_dev = cuda.to_device(J)

    #We now take the element-wise product of A and 1/QJ to give the alpha matrix    
    Alpha_dev = cuda.device_array(shape = (N,2*M+1),dtype=np.complex128)
    k_dev = cuda.to_device(k_arr)
    TPB = 2*M+1
    compute_alpha[N,TPB](Alpha_dev,A_dev,J_dev,k_dev,Q,N,M)
    Alpha  = Alpha_dev.copy_to_host() 

    return Alpha

    
    




    