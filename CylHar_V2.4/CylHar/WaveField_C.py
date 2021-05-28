import math
import numpy as np
from numba import cuda, jit, complex64, complex128, int16, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import Bessel_C as Bess

@jit(void(complex128[:,:],float64[:,:],complex128[:,:]),fastmath = True,nopython = True)
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
    M = B.shape[0]

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp = 0
            for k in range(M):
                temp = temp + A[i,k]*B[k,j]
            C[i,j] = temp

@jit(void(complex128[:],float32[:],complex128[:,:]),fastmath = True,nopython = True)
def Compute_Gamma(alpha_arr,theta_arr,Gamma):
    '''Calculates the coefficients of the bessel functions for the case of calculating the wave field
    and returns them organised as an N x M+1 matrix which we call Gamma
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        theta_arr (float32[:]) - array of angular positions
        Gamma (complex128[:,:]) -  matrix to store the output values
    '''
    M = np.int64(len(alpha_arr)//2) #computes the maximum M value

    for i in range(Gamma.shape[0]):
        for m in range(Gamma.shape[1]):
            if m == 0:
                Gamma[i,0] = alpha_arr[M] #M is the middle index of a 2M+1 element array

            else:
                term_1 = alpha_arr[M+m]*np.complex128((math.cos(m*theta_arr[i])+1j*math.sin(m*theta_arr[i])))

                term_2 = alpha_arr[M-m]*np.complex128((math.cos(m*theta_arr[i])-1j*math.sin(m*theta_arr[i])))

                Gamma[i,m] = term_1+np.float64(((-1)**m))*term_2

@jit(complex128[:,:](complex128[:],float32),fastmath = True,nopython = True)    
def Wave_Field_k(alpha_arr,k):
    '''Compute the wavefield over space for a particular frequency value for a given set of wave domain coefficients on CPU.
        Parameters:
        -----------
        alpha_arr (complex128[:]) - array of wave domain coefficients
        k (float32) - frequency to evaluate the wavefield at
        Returns:
        --------
        e_matrix (complex128[:,:]) - 2-d array of the values of the wave field in the frequency domain passed for output
    '''
    #We define the range of radii and angles we are concerned with
    r = np.linspace(0.1,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    M = np.shape(alpha_arr)[0]//2
    N = r.shape[0]

    #Defines the required 2-d arrays of function values in memory
    e = np.zeros(shape = (N,N)).astype(np.complex128)
    Gamma = np.zeros(shape = (N,M+1),dtype = np.complex128)

    #We begin by calculating the required values of the Bessel functions and take the transpose to give us as an M+1 x N array
    J = np.transpose(Bess.bessj(M,k*r))
    #print(Bess.bessj(M,k*r))

    #We then run calculate all of the coefficients for each of the terms in the linear
    #combinations and store them as an N x M+1 array which is named Gamma

    #Computes the Gamma matrix
    Compute_Gamma(alpha_arr,thet,Gamma)

    #We then take the matrix product of Gamma and R to return the e matrix
    MatMul(Gamma,J,e)

    return e

@jit(float64(complex128[:],float32),fastmath = True,nopython = True) 
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
    '''Plots the wavefield over space for a particular frequency value for a given set of wave domain coefficients on CPU.
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
    r = np.linspace(0.1,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    R, Thet = np.meshgrid(r,thet)
    X = R*np.cos(Thet)
    Y = R*np.sin(Thet)
    
    #Plots the wave field as a function of space as a visualisation
    plt.rcParams['figure.figsize'] = (16,16)
    fig = plt.figure() 
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X,Y,np.real(Z), cmap = cm.get_cmap('coolwarm'), linewidth=0,antialiased = True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

