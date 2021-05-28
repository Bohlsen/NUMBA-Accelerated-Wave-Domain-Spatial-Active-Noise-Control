import math
import numpy as np
from numba import cuda, complex128, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import WaveField_G as WF
from WaveFieldSynth.WaveFieldSynth_G import compute_wavefield
from WaveFieldSynth.SoundFields import Greens_Function_G_Kernel

def compute_transfer(Q,L,R_mic,R_speaker,K_arr):
    '''Compute the transfer matrix due to a circular array of L speakers.
        Parameters:
        -----------
        Q (int64) - Number of microphones
        L (int64) - Total number of loudspeakers
        R_mic (int64) - Radius of the microphone array
        R_speaker (int64) - Radius of the speaker array
        K_arr (float32[:]) - Array of wavenumbers to evaluate the signal at, is of shape (Nx1)

        Returns:
        --------
        T (complex128[:,:,:]) - 3-index complex array describing the transfer matrix at each frequency. Is in the form
                                T[l,k_n,m] = T_{m,l}(k_n)

    '''
    N = K_arr.shape[0]

    #Moves arrays into GPU memory
    K_device = cuda.to_device(K_arr)
    E_device = cuda.device_array(shape = (L,N,Q),dtype = np.complex128) #defines the array of measured sound field values

    Rows = Q
    Columns = N//L+1 #computes the required number of blocks to the expected grid shape 

    #Computes the sound field due to each loudspeaker at each microphone
    Greens_Function_G_Kernel[(Rows,Columns),L](E_device,Q,L,R_mic,R_speaker,K_device)

    E = E_device.copy_to_host() #Pulls the computes Green's function values back to the host

    #We compute the first case separately to avoid having to recalculate the M value
    Alpha_0 = compute_wavefield(E[0],K_arr,R_mic)

    #Assign the required arrays in memory and set the first entry in T to the expected form
    T = np.empty(shape = (L,N,Alpha_0.shape[1]),dtype=np.complex128)
    T[0] = Alpha_0

    for l in range(1,L):
        T[l] = compute_wavefield(E[l],K_arr,R_mic)

    return T

def main():
    #defines the same array of wavenumbers I always use
    sample_number = 2048 #number of samples in frequency
    f = np.linspace(0.1,1024,sample_number,dtype = np.float32) #array of frequency waves
    omega = 2*np.pi*f
    K_arr = omega/343 #computes omega/c as the wavenumber array

    for i in range(10):
        start = timeit.default_timer()
        T = compute_transfer(64,192,np.float32(1.0),np.float32(2.5),K_arr)
        end = timeit.default_timer()
        print('Runtime = ',end-start)

    WF.Plot_Wave_Field_k(T[8,2047],K_arr[2047])


if __name__ == '__main__':
    main()
    
    