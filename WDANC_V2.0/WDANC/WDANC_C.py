#General Libraries
from numba import jit, int64, float32, float64, complex64, complex128, void
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer

#Personal Libraries for WDANC
from FFT import FFT_Controller as F
from CylHar import WaveField_G as WF
from CylHar import Bessel_C as Bess
from Transfer import Transfer_C as Trans
from WaveFieldSynth import SoundFields
from WaveFieldSynth import WaveFieldSynth_C as WFS

@jit(float64(complex128[:]),fastmath = True, nopython = True)
def cost(Alpha_arr):
    output = 0.0
    for alpha in Alpha_arr:
        output += np.abs(alpha)**2
    return output

@jit(void(complex128[:],complex128[:,:],complex128[:]),nopython = True, fastmath = True)
def MatVecMul(OUTPUT,MAT,VEC):
    '''Compues the matrix-vector product of a complex valued matrix and vector of compatible shape
        Parameters:
        -----------
        OUTPUT (complex128[:]) - Vector passed for output
        MAT (complex128[:,:]) - Matrix which will is to be multiplied by a vector
        VEC (complex128[:]) - Vector which is to be multiplied by MAT
    '''
    N = OUTPUT.shape[0]
    M = VEC.shape[0]
    for i in range(N):
        OUTPUT[i] = np.complex128(0.0)
        for j in range(M):
            OUTPUT[i] += MAT[i,j]*VEC[j]

@jit(float64(complex128[:,:]),fastmath = True,nopython = True)
def Frobenius_Norm(MAT):
    '''Compute the Frobenius norm of a complex valued matrix 
        Paramers:
        ---------
        MAT (complex128[:,:]) - Complex valued matrix whose Frobenius norm is required
        Returns:
        --------
        Norm (float64) - The real number corresponding to the norm squared of the matrix
    '''
    N,M = MAT.shape
    Norm_squared = 0.0
    for n in range(N):
        for m in range(M):
            Norm_squared += np.abs(MAT[n,m])**2 #Sums the squares of all of the elements in the matrix
    return np.sqrt(Norm_squared)

def pull_delimited_region(open_file):
    '''Pulls the next delimited string from a line in the Test_Params.txt file
    '''
    temp = ''
    in_delimiters = False
    for char in open_file.readline():
        if char == '}': #If we have reached the end of a delimited region we cancel out of the scan
            break
        
        if in_delimiters: #When inside the delimited region we store the value
            temp += char

        if char == '{': #tracks that we have entered the delimited region
            in_delimiters = True
    return temp

def read_params():
    '''Read the test parameters from the Test_Params.txt file
        Parameters:
        -----------
        NONE
        Returns:
        --------
        Q (int64) - Number of microphones
        R_mic (int32) - Radius of the microphone array
        L (int64) - Number of loudspeakers
        R_speaker (float32) - Radius of the louspeaker array
        wavevector (float32[:]) - Vector which gives the direction the exterior planewave is travelling in 
        file_name (str) - name of the file whose FFT will be used as the spectrum
        f_max (int64) - Maximum frequency for ANC to be run over (Hz)
    '''
    Test_Parameters_File = open('Test_Params.txt','r')
    Test_Parameters_File.readline()
    Test_Parameters_File.readline()
    #Pulling the Q value
    Q_temp = pull_delimited_region(Test_Parameters_File)
    Q = np.int64(Q_temp)

    #Pulling the R_mic value
    R_mic_temp = pull_delimited_region(Test_Parameters_File)
    R_mic = np.float32(R_mic_temp)

    #Pulling the L value
    L_temp = pull_delimited_region(Test_Parameters_File)
    L = np.int64(L_temp)

    #Pulling the R_speaker value
    R_speaker_temp = pull_delimited_region(Test_Parameters_File)
    R_speaker = np.float32(R_speaker_temp)

    #Pulls the wavevector as a two element float32[:]
    wavevector_temp = pull_delimited_region(Test_Parameters_File)
    wave_vector = np.empty(shape = 2,dtype = np.float64)
    wave_vector[0] = np.float64(wavevector_temp[0])
    wave_vector[-1] = np.float64(wavevector_temp[-1])

    #Pulls the file_name as a string
    file_name = pull_delimited_region(Test_Parameters_File)

    #Pulling the f_max value
    f_max_temp = pull_delimited_region(Test_Parameters_File)
    f_max = np.int64(f_max_temp)

    Test_Parameters_File.close()

    return Q,R_mic,L,R_speaker,wave_vector,file_name,f_max

def WDANC_init():
    '''Performs some of the initial setup which is necessary for the WDANC process to be performed. Mostly setting up the geometry and spectral data. 
        Parameters:
        -----------
        Those included in the Test_Params.txt file
        Return:
        -------
        Beta[:,:] (complex128[:,:]) - Nx(2N+1) Matrix of wave domain coefficients of error microphone signals due to the incident plane wave in the form Beta[i][j] = beta(k_i,-M+j)
        T (complex128[:,:,:]) - 3-index complex array describing the transfer matrix at each frequency. Is in the form
                                T[l,k_n,m] = T_{m,l}(k_n)
        K_arr (float32[:]) - N long array of wavenumber values
        Q (int64) - Number of Microphones
        L (int64) - Number of Loudspeakers
        M (int64) - Order to which the cylindrical harmonics expansions will be cutoff
    '''
    #We pull the stored external simulation parameters 
    Q,R_mic,L,R_speaker,wave_vector,spectrum_file_name,f_max = read_params()


    #We first have to setup the array of wavenumber and the required spectrum
    Spectrum , sample_rate, N_temp= F.FFT_From_WAV(spectrum_file_name,CUDA=True) #We are free to use GPU here since this is really a formality

    omega_arr = np.arange(0,N_temp//2,1,dtype = np.float32) #creates the array of omega values which index the spectrum
    f_arr = (omega_arr/N_temp)*sample_rate #computes the associated time domain frequencies
    bool_arr = f_arr <= f_max
    f_temp = np.extract(bool_arr,f_arr) #Pulls the frequencies which are less than the maximum frequency we are interested in, we take this to be our frequency array
    f = f_temp[1:] #Removes the zero frequency value at the start
    omega = 2*np.pi*f #We now turn the frequency array into an array of wavenumbers
    K_arr = (omega/343).astype(np.float32) #computes omega/c as the wavenumber array
    N = K_arr.shape[0]
    Spectrum_Shortened = Spectrum[:N] #Chops the spectrum off to enough samples to fill the array, the actual content of this is irrelevant


    #We now go about computing the wavefield at the microphones
    E_planewave = SoundFields.plane_wave_fast(Q,R_mic,K_arr,Spectrum_Shortened,wave_vector)

    #We now compute the transfer matrix. Note to self, that this is not massively efficient since it will recalculate its
    #own version of the W array, and its own set of the bessel functions several times. Since this is transient though, I'm
    #not particularly concerned with this technical issue. 
    T = Trans.compute_transfer(Q,L,R_mic,R_speaker,K_arr)

    #We now compute the wave domain coefficients of the incident plane wave
    Beta = WFS.compute_wavefield(E_planewave,K_arr,R_mic)

    #Here we compute the upper limit of summation
    M = np.int64(K_arr[-1]*R_mic*np.e//2)+1 #We use the heuristic given in one of Thushara's papers to determine our general upper limit of summation


    return Beta, T, K_arr, Q, L, M

@jit(complex128[:,:,:](int64,float32,complex128[:,:],complex128[:,:,:],int64),fastmath = True,nopython = True)
def WDANC_Grad_Descent(iterations,mu,Beta,T,M):
    '''Performs a full WDANC simulation via the methods of gradient descent and outputs for the results.
        The specific method employed here is the normalised wave-domain algorithm updating driving signals (NWD-D).
        Paramters:
        ----------
        iterations (int64) - Number of iterations to run the gradient descent process for.
        mu (float32) - Step size fo the gradient descent calculations
        Beta[:,:] (complex128[:,:]) - Array of wave domain coefficients of the incident external sound field in the form Beta[i][j] = beta(k_i,-M+j)
        T[:,:,:] (complex128[:,:,:]) -  3-index complex array describing the transfer matrix at each frequency. Is in the form
                                T[l,k_n,m] = T_{m,l}(k_n)
        M (int64) - Order to which the cylindrical harmonics expansions will be cutoff
        Returns:
        --------
        AlPHA (complex128[:,:,:]) - (iterations)x(N)x(2M+1) Array of wavefield coefficients computed for each iteration
    '''
    N = Beta.shape[0]
    L = T.shape[0]

    #We begin by setting up relevant arrays in memory.
    d_arr = np.zeros(shape = (iterations,N,L),dtype = np.complex128) #We initialise the loudspeaker weights at zero
    gamma = np.zeros(shape = 2*M+1,dtype = np.complex128) #A vector named gamma is helpful to store
    update_arr = np.zeros(shape = L,dtype = np.complex128) #A vector which will be passed as an output during the gradient updating process
    ALPHA = np.zeros(shape = (iterations,N,2*M+1),dtype = np.complex128) #We initialise the entire wavefield at zero
    TK = np.zeros(shape = (N,2*M+1,L),dtype = np.complex128) #We initialise a matrix which will store T(k) in a more helpful form
    TH = np.zeros(shape = (N,L,2*M+1),dtype = np.complex128) #We initialise a matrix which will be the hermitian conjugate of T(k)
    T_Norms = np.zeros(shape = N,dtype = np.float64) #We create an array to store the norms of the transfer matrices

    #We now the Transfer matrix it in a more convenient form
    for l in range(L):
        for n in range(N):
            for m in range(2*M+1):
                TK[n,m,l] = T[l,n,m]
    
    #We now compute and store the complex conjugate of the Transfer matrix and write it in a more convenient form
    for l in range(L):
        for n in range(N):
            for m in range(2*M+1):
                TH[n,l,m] = np.conjugate(T[l,n,m]) 

    #we now compute the Frobenius (euclidean) norm of the Transfer matrix at each frequency
    for n in range(N):
        T_Norms[n] = Frobenius_Norm(TH[n]) 
        #NOTE TO SELF: some of these will come out to be inf, and when they do I think I can just
        #suggest that the loudspeaker weights will never change and thus we can treat the updates at those
        #frequencies as being zeroed. 

    #We now normalise the hermitian conjugate of the transfer matrices
    for n in range(N):
        TK[n] = TK[n]/T_Norms[n]
        TH[n] = TH[n]/T_Norms[n]
    
    #we now run the gradient descent process
    for i in range(iterations):
        #We treat each wavenumber individually
        for n in range(N):
            MatVecMul(gamma,TK[n],d_arr[i][n]) #we compute the wave domain coefficients due to the loudspeakers
            alpha = Beta[n]+gamma #We compute the full wave-domain coefficients on each pass
            ALPHA[i,n] = alpha #we store the wave domain coefficients for this iteration

            if i!=iterations-1:
                MatVecMul(update_arr,TH[n],alpha)
                d_arr[i+1,n] = d_arr[i,n] - mu*update_arr #We compute the updated driving coefficients


    return ALPHA

def WDANC_Controller(iterations,mu):
    '''Controls a WDANC simulation in pure Python so that the external files and be interfaced with
        Paramters:
        ----------
        iterations (int64) - Number of iterations to run the gradient descent process for.
        mu (float32) - Step size fo the gradient descent calculations (this is assumed be an element of [0,1])
        Returns:
        --------
        AlPHA (complex128[:,:,:]) - (iterations)x(N)x(2M+1) Array of wavefield coefficients computed for each iteration
        K_arr (float32[:]) - Array of wave numbers which helpful to output
        '''
    Beta, T, K_arr, Q, L, M = WDANC_init() #Performs the initialisation of the relevant geometry and the wavefields. 

    ALPHA = WDANC_Grad_Descent(iterations,mu,Beta,T,M)

    return ALPHA,K_arr

    

def NoiseReductionAgainstIteration():
    iterations = 250
    ALPHA,K_arr = WDANC_Controller(iterations,1.0)

    Noise_level_array_1024 = np.zeros(iterations,dtype=np.float64)
    Noise_level_ini_1024 = WF.noise_level(ALPHA[0,2047],K_arr[2047])

    Noise_level_array_768 = np.zeros(iterations,dtype=np.float64)
    Noise_level_ini_768 = WF.noise_level(ALPHA[0,1535],K_arr[1535])

    Noise_level_array_512 = np.zeros(iterations,dtype=np.float64)
    Noise_level_ini_512 = WF.noise_level(ALPHA[0,1023],K_arr[1023])

    for i in range(iterations):
        Noise_level_iteration_1024 = WF.noise_level(ALPHA[i,2047],K_arr[2047]) 
        Noise_level_array_1024[i] = 10*np.log10(Noise_level_iteration_1024/Noise_level_ini_1024)

        Noise_level_iteration_768 = WF.noise_level(ALPHA[i,1535],K_arr[1535]) 
        Noise_level_array_768[i] = 10*np.log10(Noise_level_iteration_768/Noise_level_ini_768)

        Noise_level_iteration_512 = WF.noise_level(ALPHA[i,1023],K_arr[1023]) 
        Noise_level_array_512[i] = 10*np.log10(Noise_level_iteration_512/Noise_level_ini_512)

    plt.rcParams['figure.figsize'] = (16,9)
    fig, ax = plt.subplots()
    ax.plot(Noise_level_array_1024,'b',label = '1024Hz')
    ax.plot(Noise_level_array_768,'r',label = '768Hz')
    ax.plot(Noise_level_array_512,'g',label = '512Hz')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Noise Reduction (db)')
    plt.legend()
    plt.show()

    

def NoiseReductionAgainstFrequency():
    ALPHA,K_arr = WDANC_Controller(100,1.0)

    Noise_level_array = np.zeros_like(K_arr).astype(np.float64)

    for i in range(K_arr.shape[0]):
        Noise_level_ini = WF.noise_level(ALPHA[0,i],K_arr[i])
        Noise_level_fin = WF.noise_level(ALPHA[-1,i],K_arr[i]) 
        Noise_level_array[i] = 10*np.log10(Noise_level_fin/Noise_level_ini)

    plt.rcParams['figure.figsize'] = (16,9)
    fig, ax = plt.subplots()
    ax.plot(K_arr*343/(2*np.pi),Noise_level_array,)
    ax.plot(K_arr*343/(2*np.pi),0.0*K_arr,'k')
    ax.set_xlabel('Frequency  (Hz)')
    ax.set_ylabel('Noise Reduction (db)')
    plt.show()

def main():
    NoiseReductionAgainstFrequency()

if __name__ == '__main__':
    main()