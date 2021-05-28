from FFT import FFT_C as C
from FFT import FFT_G as G 
from numba import jit, float32, int16, complex64, cuda
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


@jit(nopython = True, fastmath = True)
def zero_pad(in_array):
    """
    Return an ndarray of complex numbers whch has values the same as array but with enough zeroes added onto the end that
    the length of the array is a power of two. 
     Parameters:
     ----------
     in_array (ndarray) - a numpy array of some numerical dataype

     Returns:
     -------
     out_array (ndarray) - a numpy array which has the same values as array but is padded with zeroes to the next
                            power of two length
    """
    N = in_array.shape[0]
    power = np.int16(np.ceil(np.log2(N)))
    
    out_array = np.zeros(2**(power),dtype=np.complex64)

    out_array[:N] = in_array.astype(np.complex64)

    return out_array

def FFT_From_WAV(path_to_file, CUDA = False):
    """
    Compute the Fast Fourier Transform of the audio signal in a wav file
     Parameters:
     ----------
     path_to_file (str) - Path to the wav file which we are taking the FFT of.
     CUDA (bool) - a bool which states whether to perform the calculation of th GPU (True) or CPU (False)

     Returns:
     out_array (ndarray) - an array of complex64 values which is the DFT of the original wav file
     sample_rate (int) - an integer pulled from the wavfile which describes the sample rate of the recording in Hz
     N (int) - a integer describing the number of points in the padded audiofile
    """
    sample_rate, data = wavfile.read(path_to_file)#reads the file

    

    data_padded = zero_pad(data)#adds zeroes to reach a power of 2
    N = data_padded.shape[0]

    if CUDA: #implements the FFT on the GPU
        fou = np.zeros_like(data_padded)
        TPB = 128
        numblocks = np.int16(fou.shape[0]//TPB)
        threadsperblock = np.int16(TPB)

        fou = np.zeros_like(data_padded)

        G.FFT(data_padded,fou,numblocks,threadsperblock)

        return fou, sample_rate, N

    else: #implements the FFT of the CPU
        fou = np.zeros_like(data_padded)
        C.FFT(data_padded,fou)
        return fou, sample_rate, N

def IFFT_From_WAV(path_to_file, CUDA = False):
    """
    Compute the Fast Fourier Transform of the audio signal in a wav file and then IFFT's it back for testing
     Parameters:
     ----------
     path_to_file (str) - Path to the wav file which we are taking the FFT of.
     CUDA (bool) - a bool which states whether to perform the calculation of th GPU (True) or CPU (False)

     Returns:
     out_array (ndarray) - an array of complex64 values which is the DFT of the original wav file
     sample_rate (int) - an integer pulled from the wavfile which describes the sample rate of the recording in Hz
     N (int) - a integer describing the number of points in the padded audiofile
    """
    sample_rate, data = wavfile.read(path_to_file)#reads the file
    
    

    data_padded = zero_pad(data)#adds zeroes to reach a power of 2
    N = data_padded.shape[0]

    if CUDA: #implements the FFT on the GPU
        fou = np.zeros_like(data_padded)
        TPB = 128
        numblocks = np.int16(fou.shape[0]//TPB)
        threadsperblock = np.int16(TPB)

        fou = np.zeros_like(data_padded)

        G.FFT(data_padded,fou,numblocks,threadsperblock)

        G.IFFT(fou,data_padded,numblocks,threadsperblock)

        return data_padded, sample_rate, N

    else: #implements the FFT of the CPU
        fou = np.zeros_like(data_padded)
        C.FFT(data_padded,fou)

        C.FIFT(fou,data_padded)

        return data_padded, sample_rate, N

def energy_spectrum(path_to_file, CUDA = False):
    """
    Graph the energy spectrum of the audio signal in a wav datafile as a line graph.
    Parameters:
     ----------
     path_to_file (str) - Path to the wav file which we are taking the FFT of.
     CUDA (bool) - a bool which states whether to perform the calculation of th GPU (True) or CPU (False)
    """
    CUDA_bool = CUDA

    fou, sample_rate, N = FFT_From_WAV(path_to_file,CUDA = CUDA_bool)
    

    E_spec = np.abs(fou)**2 #calculate the energy spectrum of the input datafile
    k_arr = np.arange(0,N//2,1) #creates the array of k values which index fou
    f_arr = (k_arr/N)*sample_rate #computes the associated time domain frequencies


    fig, ax = plt.subplots()
    ax.plot(f_arr,E_spec[:N//2])
    plt.show()

if __name__ == '__main__':    
    energy_spectrum('CantinaBand60.wav',CUDA = True)  








