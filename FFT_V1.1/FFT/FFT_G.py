import numpy as np
import math
from numba import cuda, jit, complex64, int16, int64, float32, void
from scipy.io import wavfile
import timeit
import matplotlib.pyplot as plt 

@jit(void(complex64[:],complex64[:],int64),fastmath = True,nopython = True)
def reverse_binary_digits(In_arr,Out_arr,bits):
    '''Reverses the bit order of a power of 2 sized array.
        Parameters:
        ----------
            In_arr (ndarray) - array of complex numbers which positions are to be rearranged by bit reversal.
            Out_arr (ndarray) - array to store the output.
            bits (int64) -  the number of bits in the relvant dataset
        Assumptions:
        -----------
            Size of the input array must be a power of 2'''
    N = In_arr.shape[0]
    for pos in range(N):#runs over each element in the array
        n = pos
        j = 0 #tracks the current running total of the binary reversed version
        power = 0 #tracks the power
        while n!= 0: #implements the repeated division by 2 algorithm
            j += (n%2)*(2)**(bits-1-power)
            n = n//2
            power += 1
        Out_arr[pos] = In_arr[j]

@cuda.jit(void(complex64[:],complex64[:],int16,int16),fastmath = True)
def Radix_Cuda_stage(x_array,X_array,max_layer_depth,layer_depth):
    '''Performs a stage of the DIT radix-2 FFT algorithm
        Parameters:
        ----------
            x_array (ndarray) - Array of complex64 numbers whose DFT is being taken
            X_array (ndarray) - Output array which will store the DFT
            max_layer_depth (int16) - The number of radix-2 stages
            layer_depth (int16) - The number of elements in each DIT subblock for this stage in the calculation'''

    
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # grid width, i.e. number of blocks per grid
    gw = cuda.gridDim.x
    
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Compute flattened index inside the array of the output element this thread will handle
    k = tx + ty * bw

    N_temp = np.float32(2**layer_depth) #size of the stages in each layer

    #cancels if we are outside the overall size of the array
    if k >= x_array.shape[0]:
        return
    
    temp = 0    
    theta = np.float32((2*np.pi/N_temp)*k)
         
    #Writes each element as the required sum of 2 earlier elements
    if k%N_temp < N_temp//2:
        temp = x_array[k]+x_array[k+2**(layer_depth-1)]*(math.cos(theta)-1j*math.sin(theta))
    else:
        temp = x_array[k-2**(layer_depth-1)]+x_array[k]*(math.cos(theta)-1j*math.sin(theta))
        
    X_array[k] = temp

def FFT(x_array,X_array,numBlocks,threadsperblock):
    '''Take the Discrete Fourier Transform of an array of complex numbers.
        Parameters:
        ----------
            x_array (ndarray) - Array of complex64 numbers whose DFT is being taken
            X_array (ndarray) - Output array which will store the DFT
            numBlocks (int16) - The number of CUDA blocks to be initialised
            threadsperblock (int16) - The number of threads assigned to each CUDA block'''
    #stores the size of the input array and computes which power of two we are dealing with
    N = X_array.shape[0]
    max_layer_depth = np.int64(np.log2(N))

    #Since DIT has the effect of reversing the order of the indexing bits of the input data
    #the first step is to rearrange the input by reversing their bit orders. 
    #note that there are max_layer_depth binary bits to indicate the position of each of the elements
    x_array_rearrange = np.zeros_like(x_array)
    reverse_binary_digits(x_array,x_array_rearrange,max_layer_depth)
    
    layer_depth = np.int16(1)

    #Moves the arrays into GPU memory as the input signal array and the fourier array
    Fou_arr = cuda.to_device(X_array)
    Sig_arr = cuda.to_device(x_array_rearrange)
    x_temp = np.zeros_like(x_array) #defines a temporary zero array which will be used for storage
    

    while layer_depth <= max_layer_depth:
        
        #runs a stage of the radix-2 algorithm on graphics
        Radix_Cuda_stage[numBlocks,threadsperblock](Sig_arr,Fou_arr,max_layer_depth,layer_depth)

        #moves elements arround in memory so that the next stage of decimation in time can be called. 
        Fou_arr.copy_to_host(x_temp)
        Sig_arr = cuda.to_device(x_temp)
        Fou_arr = cuda.device_array_like(Sig_arr)
        layer_depth +=1

    X_array[:] = x_temp[:]

@cuda.jit(void(complex64[:],complex64[:],int16,int16),fastmath = True)
def Inverse_Radix_Cuda_stage(x_array,X_array,max_layer_depth,layer_depth):
    '''Performs a stage of the DIF radix-2 IFFT algorithm
        Parameters:
        ----------
            x_array (ndarray) - Array of complex64 numbers whose IDFT is being taken
            X_array (ndarray) - Output array which will store the IDFT
            max_layer_depth (int16) - The number of radix-2 stages
            layer_depth (int16) - The number of elements in each DIF subblock for this stage in the calculation'''

    
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # grid width, i.e. number of blocks per grid
    gw = cuda.gridDim.x
    
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Compute flattened index inside the array of the output element this thread will handle
    k = tx + ty * bw

    N_temp = np.float32(2**layer_depth) #size of the stages in each layer

    #cancels if we are outside the overall size of the array
    if k >= x_array.shape[0]:
        return
    
    temp = 0    
    theta = np.float32((2*np.pi/N_temp)*k)
         
    #Writes each element as the required sum of 2 earlier elements
    if k%N_temp < N_temp//2:
        temp = x_array[k]+x_array[k+2**(layer_depth-1)]*(math.cos(theta)+1j*math.sin(theta))
    else:
        temp = x_array[k-2**(layer_depth-1)]+x_array[k]*(math.cos(theta)+1j*math.sin(theta))
        
    X_array[k] = temp




def IFFT(x_array,X_array,numBlocks,threadsperblock):
    '''Take the Inverse Discrete Fourier Transform of an array of complex numbers.
        Parameters:
        ----------
            x_array (ndarray) - Array of complex64 numbers whose IDFT is being taken
            X_array (ndarray) - Output array which will store the IDFT
            numBlocks (int16) - The number of CUDA blocks to be initialised
            threadsperblock (int16) - The number of threads assigned to each CUDA block'''
    #stores the size of the input array and computes which power of two we are dealing with
    N = X_array.shape[0]
    max_layer_depth = np.int64(np.log2(N))

    #Since DIT has the effect of reversing the order of the indexing bits of the input data
    #the first step is to rearrange the input by reversing their bit orders. 
    #note that there are max_layer_depth binary bits to indicate the position of each of the elements
    x_array_rearrange = np.zeros_like(x_array)
    reverse_binary_digits(x_array,x_array_rearrange,max_layer_depth)
    
    layer_depth = np.int16(1)

    #Moves the arrays into GPU memory as the input signal array and the fourier array
    Fou_arr = cuda.to_device(X_array)
    Sig_arr = cuda.to_device(x_array_rearrange)
    x_temp = np.zeros_like(x_array) #defines a temporary zero array which will be used for storage
    

    while layer_depth <= max_layer_depth:
        
        #runs a stage of the radix-2 algorithm on graphics
        Inverse_Radix_Cuda_stage[numBlocks,threadsperblock](Sig_arr,Fou_arr,max_layer_depth,layer_depth)

        #moves elements arround in memory so that the next stage of decimation in time can be called. 
        Fou_arr.copy_to_host(x_temp)
        Sig_arr = cuda.to_device(x_temp)
        Fou_arr = cuda.device_array_like(Sig_arr)
        layer_depth +=1

    X_array[:] = x_temp[:]/N
