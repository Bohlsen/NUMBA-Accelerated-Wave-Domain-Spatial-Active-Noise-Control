import numpy as np
import math
from numba import cuda, jit, complex64, int16, int64, float32, void
import timeit
import matplotlib.pyplot as plt 

from CylHar import Bessel_C as C
from CylHar import Bessel_G as G

def bessj(m,x_arr,CUDA = False):
    '''Calculate the bessel function of the first kind of order m on either CPU of GPU
        Parameters:
        -----------
        m (int64) - The order of the bessel function
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at
        CUDA (bool) - A bolean stating whether or not to run on graphics
        Returns:
        --------
        y_arr (float32[:,:]) - The values of all of the positive bessel functions at each input point up to m
                               with the arangement y_arr[x_n,m] = J_m(x_n)
    '''
    #runs on GPU
    if CUDA:
        y_arr = G.bessj(m,x_arr)

    #runs on CPU
    else:
        y_arr = C.bessj(m,x_arr)

    return y_arr

def hankel1_0(x_arr,CUDA = False):
    '''Calculate the zeroth order Hankel function of the first kind on either CPU of GPU
        Parameters:
        -----------
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at
        CUDA (bool) - A bolean stating whether or not to run on graphics
        Returns:
        --------
        y_arr (complex128[:]) - The values of all of the positive bessel functions at each input point up to m
                               with the arangement y_arr[x_n,m] = J_m(x_n)
    '''
    #runs on GPU
    if CUDA:
        y_arr = G.hankel1_0(x_arr)

    #runs on CPU
    else:
        y_arr = C.hankel1_0(x_arr)

    return y_arr

def plot_bessj(m,x_arr,CUDA = False,color = 'r'):
    '''Plot the bessel function for a particular m at every point in an input point set
    Parameters:
        -----------
        m (int64) - The order of the bessel function
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at
        CUDA (bool) - A bolean stating whether or not to run on graphics
        color (str) - A string indicating the matplotlib color to be called
    '''    

    if CUDA:
        y_arr = G.bessj(m,x_arr)
        j_arr_plot = np.transpose(y_arr)[m] #transpose pulls all of the values in the y_arr matrix associated with m at each x_n
        plt.plot(x_arr,j_arr_plot,color)

    #runs on CPU
    else:
        y_arr = C.bessj(m,x_arr)
        j_arr_plot = np.transpose(y_arr)[m]
        plt.plot(x_arr,j_arr_plot,color)

if __name__ == '__main__':
    print('Bessel_C.py running')
    X = np.arange(0.01,100,0.1).astype(np.float32)
    plot_bessj(0,X,CUDA = True,color = 'g')
    plot_bessj(1,X,CUDA = True,color = 'b')
    plot_bessj(50,X,CUDA = True,color = 'g')
    plot_bessj(0,X,CUDA = False,color = 'b--')
    plot_bessj(1,X,CUDA = False,color ='y--')
    plot_bessj(50,X,CUDA = False,color = 'm--')
    plt.show()