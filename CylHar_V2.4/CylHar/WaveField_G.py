import math
import numpy as np
from numba import cuda, jit, complex64, complex128, int16, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import Bessel_G as Bess


#Imported and edited from the CUDA_Ops_Std file (data types needed to be changed for eager compilation)
@cuda.jit(void(complex128[:,:],float64[:,:],complex128[:,:]),fastmath = True)
def MatMul(A,B,C):
    '''Calculate the matrix product of two matrices AB which are assumed to be of the right sizes
    Parameters:
    -----------
    A (complex128[:,:]) - First array in the product
    B (float64[:,:]) - Second array in the product
    C (complex128[:,:]) - Array for storing the product of the multiplication
    Assumptions:
    ------------
    Block width is the number of columns
    Grid width is the number of rows and they are equal
    '''

    #we assign each block a row of the output and each thread an element
    
    i = cuda.blockIdx.x #position of the block
    j = cuda.threadIdx.x #position of the thread in a block


    M = B.shape[0] #Number of columns in A and rows in B

    temp = 0
    for k in range(M):
        temp = temp + A[i,k]*B[k,j]
    
    C[i,j] = temp

@cuda.jit(void(complex128[:],float32[:],complex128[:,:]),fastmath = True)
def Gamma_kernel(alpha_arr,theta_arr,Gamma):
    '''Calculates the coefficients of the bessel functions for the case of calculating the wave field
    and returns them organised as an N x M+1 matrix which we call Gamma
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        theta_arr (float32[:]) - array of angular positions
        Gamma (complex128[:,:]) -  matrix to store the output values
    '''

    i = cuda.blockIdx.x #position in grid
    m = cuda.threadIdx.x #position in Block
    M = cuda.blockDim.x-1 #maximum M value
    
    if m == 0:
        Gamma[i,0] = alpha_arr[M] #M is the middle index of a 2M+1 element array

    else:
        term_1 = alpha_arr[M+m]*np.complex64((math.cos(m*theta_arr[i])+1j*math.sin(m*theta_arr[i])))

        term_2 = alpha_arr[M-m]*np.complex64((math.cos(m*theta_arr[i])-1j*math.sin(m*theta_arr[i])))

        Gamma[i,m] = term_1+np.float32(((-1)**m))*term_2


def Wave_Field_k(alpha_arr,k):
    '''Compute the wavefield over space for a particular frequency value for a given set of wave domain coefficients on GPU.
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        k (float32) - frequency to evaluate the wavefield at
        Returns:
        --------
        e_matrix (complex128[:,:]) - 2-d array of the values of the wave field in the frequency domain passed for output
    '''
    #We define the range of radii and angles we are concerned with
    r = np.linspace(0.01,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    M = np.shape(alpha_arr)[0]//2
    N = r.shape[0]

    #Defines the required 2-d arrays of function values in memory
    e = np.zeros(shape = (N,N)).astype(np.complex128)
    Gamma = np.zeros(shape = (N,M+1),dtype = np.complex128)

    #We begin by calculating the required values of the Bessel functions and take the transpose to give us as an M+1 x N array
    J = np.transpose(Bess.bessj(M,k*r))
    #print(J)
    #We then run calculate all of the coefficients for each of the terms in the linear
    #combinations and store them as an N x M+1 array which is named Gamma

    #Sets up the required Grid sizes
    BPG = N
    TPB = M+1

    #Sets up arrays into GPU memory
    alpha_device = cuda.to_device(alpha_arr)
    thet_device = cuda.to_device(thet)
    Gamma_device = cuda.device_array(shape = (N,M+1),dtype = np.complex128)

    #Computes the Gamma matrix
    Gamma_kernel[BPG,TPB](alpha_device,thet_device,Gamma_device)

    #Stores the gamma matrix in device memory
    Gamma_device.copy_to_host(Gamma)

    #Takes the matrix product of Gamma and J to compute e
    MatMul[N,N](Gamma,J,e)

    return e
 
def noise_level(alpha_arr,k):
    '''Calculate a noise_level metric for a given set of alpha coefficients by summing over a set of sample points.
        Note that we sample at 100 angles and at 10 different radii.
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        k (float32) - frequency to evaluate the wavefield at
        Returns:
        --------
        output (float64) - A value representing the noise level metric
    '''

    E = Wave_Field_k(alpha_arr,k)

    output = 0.0
    for n in range(10):
        for m in range(100):
            output += np.abs(E[100*n,10*m])**2
    return output

def Plot_Wave_Field_k(alpha_arr,k):
    '''Plots the wavefield over space for a particular frequency value for a given set of wave domain coefficients on GPU.
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        k (float32) - frequency to evaluate the wavefield at
        Returns:
        --------
        e_matrix (complex128[:,:]) - 2-d array of the values of the wave field in the frequency domain in the form e[i][j] = e(r_j,theta_i)
    '''
     #Computes the Wave_Field
    Z = Wave_Field_k(alpha_arr,k)

    #Builds the grid over space in polar coordinates
    r = np.linspace(0.01,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    R, Thet = np.meshgrid(r,thet)
    X = R*np.cos(Thet)
    Y = R*np.sin(Thet)
    
    #Plots the wave field as a function of space as a visualisation
    plt.rcParams['figure.figsize'] = (16,16)
    fig = plt.figure() 
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X,Y,np.real(Z), cmap = cm.get_cmap('coolwarm'), linewidth=0,antialiased = True)
    #ax.set_zlim([-10,10])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



