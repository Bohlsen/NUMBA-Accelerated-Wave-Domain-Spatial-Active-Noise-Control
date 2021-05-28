import numpy as np
import matplotlib.pyplot as plt
from numba import jit, complex64, int16, int64, float32, void
from scipy.io import wavfile
import timeit

@jit(void(complex64[:],complex64[:],int64),fastmath = True,nopython = True)
def reverse_binary_digits(In_arr,Out_arr,bits):
    '''Reverses the bit order of a power of 2 sized array.
        Parameters:
        ----------
            In_arr (ndarray) - array of complex numbers which positions are to be rearranged by digit reversal.
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

@jit(void(complex64[:],complex64[:]),fastmath = True,nopython = True)
def FFT(x_array,X_array):
    '''Take the Discrete Fourier Transform of an array of complex numbers via the Radix-2 Algorithm.
        Parameters:
        ----------
            x_array (ndarray) - Array of complex64 numbers whose DFT is being taken
            X_array (ndarray) - Output array which will store the DFT
        Assumptions:
        -----------
            The size of the input array must be a power of 2 for Radix-2 to operate correctly'''
    #stores the size of the input array and computes which power of two we are dealing with
    N = X_array.shape[0]
    max_layer_depth = np.int16(np.log2(N))

    #Since DIT has the effect of reversing the order of the indexing bits of the input data
    #the first step is to rearrange the input by reversing their bit orders. 
    #note that there are max_layer_depth binary bits to indicate the position of each of the elements
    x_array_rearrange = np.zeros_like(x_array)
    reverse_binary_digits(x_array,x_array_rearrange,max_layer_depth)

    #We begin by performing the two point DFT's on N//2 bins of two elements which is layer 1
    layer_depth = 1
    X_array_prev = x_array_rearrange[:]
    X_array_temp = np.zeros_like(X_array)

    #runs over each of the layers of a FFT web and at each stage collapses down the web
    while layer_depth <= max_layer_depth:
        
        N_temp = 2**layer_depth #size of the bins in each layer
        Bins_count = 2**(max_layer_depth-layer_depth) #number of bins in each layer
        W_temp = np.exp(-1j*2*np.pi/N_temp)


        for m in range(Bins_count):#runs over each bin of the output layer
            for k in range(N_temp//2):#runs over the elements of each bin
                W_temp_k = W_temp**k
                index_l = m*N_temp+k
                index_u = index_l+N_temp//2

                X_array_temp[index_l] = X_array_prev[index_l]+W_temp_k*X_array_prev[index_u]
                X_array_temp[index_u] = X_array_prev[index_l]-W_temp_k*X_array_prev[index_u]

        #we can throw away the input data and hence use the same memory space for our new layer arrays
        X_array_prev[:] = X_array_temp
        layer_depth += 1 #counts which layer of the web we are in

    X_array[:] = X_array_prev

@jit(void(complex64[:],complex64[:]),fastmath = True,nopython = True)
def FIFT(X_array,x_array):
    '''Take the Inverse Discrete Fourier Transform of an array of complex numbers via the Radix-2 Algorithm.
        Parameters:
        ----------
            X_array (ndarray) - Array of complex64 numbers whose IDFT is being taken
            x_array (ndarray) - Output array which will store the IDFT
        Assumptions:
        -----------
            The size of the input array must be a power of 2 for Radix-2 to operate correctly.
            Note that the decimation in time algorithm is used'''
    #stores the size of the input array and computes which power of two we are dealing with
    N = x_array.shape[0]
    max_layer_depth = np.int16(np.log2(N))
    
    #Since DIT has the effect of reversing the order of the indexing bits of the input data
    #the first step is to rearrange the input by reversing their bit orders. 
    #note that there are max_layer_depth binary bits to indicate the position of each of the elements
    X_array_rearrange = np.zeros_like(X_array)
    reverse_binary_digits(X_array,X_array_rearrange,max_layer_depth)


    #We begin by performing the two point DFT's on N//2 bins of two elements which is layer 1
    layer_depth = 1
    x_array_prev = X_array_rearrange[:]
    x_array_temp = np.zeros_like(x_array)

    #runs over each of the layers of a FFT web and at each stage collapses down the web
    while layer_depth <= max_layer_depth:
        
        N_temp = 2**layer_depth #size of the bins in each layer
        Bins_count = 2**(max_layer_depth-layer_depth) #number of bins in each layer
        W_temp = np.exp(1j*2*np.pi/N_temp)


        for m in range(Bins_count):#runs over each bin of the output layer
            for k in range(N_temp//2):#runs over the elements of each bin
                W_temp_k = W_temp**k
                index_l = m*N_temp+k
                index_u = m*N_temp+k+N_temp//2

                x_array_temp[index_l] = x_array_prev[index_l]+W_temp_k*x_array_prev[index_u]
                x_array_temp[index_u] = x_array_prev[index_l]-W_temp_k*x_array_prev[index_u]

        #we can throw away the input data and hence use the same memory space for our new layer arrays
        x_array_prev[:] = x_array_temp

        layer_depth += 1 #counts which layer of the web we are in

    #dividing by N normalises the output to the true IDFT
    x_array[:] = x_array_prev/N


if __name__ == '__main__':
    print('FFT_Radix2_CPU running')
    #tests the DFT for a particular test audio file
    samples,test_data = wavfile.read('Test.wav')
    
    #zero pads the audio file so that the number of points is a power
    x = np.zeros(2**17,dtype = np.complex64)
    for k in range(test_data.shape[0]):
        x[k] = np.complex64(test_data[k])
    
    X = np.zeros_like(x)
    start = timeit.default_timer()
    FFT(x,X)
    end = timeit.default_timer()
    print(end-start)

    k_arr = np.linspace(0,2**17,2**17)
    f_arr = k_arr*samples/(2**17)
    fig,ax = plt.subplots()
    ax.plot(f_arr,np.abs(X)**2)
    ax.set_xlim(0,2500)
    plt.show()
    
    X2 = np.zeros_like(x)
    start = timeit.default_timer()
    FFT(x,X2)
    end = timeit.default_timer()
    print(end-start)

    X3 = np.zeros_like(x)
    start = timeit.default_timer()
    FFT(x,X3)
    end = timeit.default_timer()
    print(end-start)
    
    x = np.zeros_like(X)
    start = timeit.default_timer()
    FIFT(X,x)
    end = timeit.default_timer()
    print(end-start)

    fig2,ax2 = plt.subplots()
    ax2.plot(np.real(x))
    plt.show()
    

    


    
    