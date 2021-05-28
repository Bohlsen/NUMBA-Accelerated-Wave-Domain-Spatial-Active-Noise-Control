import numpy as np
import math
from numba import cuda, jit, complex64, complex128, int64, float32, float64, void
from CylHar import Bessel_C as C
import timeit

def gaussian(k):
    std_dev = 1.0
    mean = 10.0
    return np.exp(-(k-mean)**2/(std_dev)**2)

@jit(complex128[:,:](int64,float32,float32[:],complex64[:],float64[:]),nopython=True,fastmath=True)
def plane_wave_fast(Q,R,K_arr,spectrum,wavevector):
    '''Compute the values at the microphone array for the case of a plane wave multiplied by some spectrum function travelling in 1 direction
        Parameters:
        -----------
        Q (int64) - Number of microphones
        R (float32) - Radius of the microphone array
        K_arr (float32[:]) - Array of wavenumbers to evaluate the signal at
        spectrum (void) - Function which will give a coefficient to the plane wave of each wavenumber
                            Note, spectrum can also be an array with the same dimensions as K_arr containing a spectrum
        wavevector (float32[:]) - Vector which gives the direction the wave is travelling in 

        Returns:
        --------
        E_matrix    (complex128[:,:]) - NxQ Matrix of error microphone signals in the form E[i][j] = e(x_j,k_i)
    '''
    N = K_arr.shape[0] #pulls the number of frequencies
    E = np.zeros(shape = (N,Q),dtype = np.complex128)

    
    for q in range(Q):
        theta = 2*np.pi/Q*(q)
        x = R*np.cos(theta) #x-position
        y = R*np.sin(theta) #y-position
        for i in range(N):
            kdotx = K_arr[i]*(x*wavevector[0]+y*wavevector[1])

            E[i][q] = np.complex128(spectrum[i]*np.exp(-1j*kdotx))

    return E

def plane_wave_slow(Q,R,K_arr,spectrum,wavevector):
    '''Compute the values at the microphone array for the case of a plane wave multiplied by some spectrum function travelling in 1 direction
        Parameters:
        -----------
        Q (int64) - Number of microphones
        R (int64) - Radius of the microphone array
        K_arr (float32[:]) - Array of wavenumbers to evaluate the signal at
        spectrum (void) - Function which will give a coefficient to the plane wave of each wavenumber
                            Note, spectrum can also be an array with the same dimensions as K_arr containing a spectrum
        wavevector (float32[:]) - Vector which gives the direction the wave is travelling in 

        Returns:
        --------
        E_matrix    (complex128[:,:]) - NxQ Matrix of error microphone signals in the form E[i][j] = e(x_j,k_i)
    '''
    N = K_arr.shape[0] #pulls the number of frequencies
    E = np.zeros(shape = (N,Q),dtype = np.complex128)

    
    for q in range(Q):
        theta = 2*np.pi/Q*(q)
        x = R*np.cos(theta) #x-position
        y = R*np.sin(theta) #y-position
        for i in range(N):
            kdotx = K_arr[i]*(x*wavevector[0]+y*wavevector[1])

            if type(spectrum)==type(K_arr):
                E[i][q] = np.complex128(spectrum[i]*np.exp(-1j*kdotx))
            else:
                E[i][q] = np.complex128(spectrum(K_arr[i])*np.exp(-1j*kdotx))

    return E

@jit(complex128[:,:](int64,float32,float32,int64,int64,float32[:]),nopython = True,fastmath = True)
def Greens_Function_C(Q,R_mic,R_speaker,l,L,K_arr):
    '''Compute the greens function of the helmholtz equation due to a speaker at the position (R_speaker,2*l*pi/L) where L is the total number of speakers.
        Note that the green's function is (i/4)H_0^1(k||x-y_l||) where H denotes the hankel function
        Parameters:
        -----------
        Q (int64) - Number of microphones
        R_mic (int64) - Radius of the microphone array
        R_speaker (int64) - Radius of the speaker array
        l (int64) - Index of the loudspeaker being considered, should be between 0 and L-1
        L (int64) - Total number of loudspeakers
        K_arr (float32[:]) - Array of wavenumbers to evaluate the signal at, is of shape (Nx1)

        Returns:
        --------
        E_matrix    (complex128[:,:]) - NxQ Matrix of error microphone signals in the form E[i][j] = e(x_j,k_i)
    '''
    N = K_arr.shape[0]
    E = np.zeros(shape = (N,Q),dtype = np.complex128)

    theta_speaker = 2*np.pi*l/L
    x_speaker = R_speaker*math.cos(theta_speaker) #x-position of speaker
    y_speaker = R_speaker*math.sin(theta_speaker) #y-position of speaker

    for q in range(Q):
        theta_mic = 2*np.pi*q/Q
        x_mic = R_mic*math.cos(theta_mic) #x-position of q microphone
        y_mic = R_mic*math.sin(theta_mic) #y-position of q microphone

        distance = np.float32(math.sqrt((x_mic-x_speaker)**2+(y_mic-y_speaker)**2)) #compute the distance between the mic and the speaker
        
        E[:,q] = 1j*0.25*C.hankel1_0(K_arr*distance)

    return E


###################################################################################################################################################################################
#The functions below are imported from Bessel_G so that they can be called as device functions by the Greens_Function_G_Kernel
#############################################################################################################################################################################
@cuda.jit(float64(float32),fastmath=True,device = True)
def bessj0(x):
    '''Calculates the zeroth order bessel function of the first kind for a scalar by applying the approximations
    described in Numerical Recipes.
        Parameters:
        -----------
        x (float32) - Input value for the function
        Assumptions:
        ------------
        The input is assumed to be positive as for the case of wave domain ANC this is acceptable
        Returns:
        --------
        ans (float64) - The value of J_0 at the value x
    '''

    #For x<=8 we apply the rational fitting function given in (Hart,1968) and listed in Numerical Recipes
    if x <= 8:
        y = x*x
        Num = 57568490574.0+y*(-13362590354.0+y*(651619640.7+y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))))
        Den = 57568490411.0+y*(1029532985.0+y*(9494680.718+y*(59272.64853+y*(267.8532712+y*1.0))))
        ans = Num/Den
        return np.float64(ans)

    #For x>8 we apply the approximation forms described in Numerical Recipes
    else:
        z= 8/x
        y = z*z
        X_0 = x-0.785398164 #x-pi/4

        Term1 = 1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4+y*(-0.2073370639e-5+y*0.2093887211e-6)))
        Term2 = -0.1562499995e-1+y*(0.1430488765e-3+y*(-0.6911147651e-5+y*(0.7621095161e-6-y*0.9349945152e-7)))

        ans = math.sqrt(0.636619772/x)*(math.cos(X_0)*Term1-z*math.sin(X_0)*Term2)
        return np.float64(ans)

@cuda.jit(float64(float32),fastmath = True,device = True)
def bessy0(x):
    '''Calculates the zeroth order bessel function of the second kind for a scalar by applying the approximations
    described in Numerical Recipes.
        Parameters:
        -----------
        x (float32) - Input value for the function
        Assumptions:
        ------------
        The input is assumed to be positive as for the case of wave domain ANC this is acceptable
        Returns:
        --------
        ans (float64) - The value of Y_0 at the value x
    '''
    #For x<=8 we apply the rational fitting function given in (Hart,1968) and listed in Numerical Recipes
    if x<=8:
        y = x*x
        Num = -2957821389.0+y*(7062834065.0+y*(-512359803.6+y*(10879881.29+y*(-86327.92757+y*228.4622733))))
        Den = 40076544269.0+y*(745249964.8+y*(7189466.438+y*(47447.26470+y*(226.1030244+y*1.0))))
        ans = Num/Den+0.636619772*bessj0(x)*math.log(x)
        return np.float64(ans)

    #For x>8 we apply the approximation forms described in Numerical Recipes
    else:
        z= 8/x
        y = z*z
        X_0 = x-0.785398164 #x-pi/4

        Term1 = 1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4+y*(-0.2073370639e-5+y*0.2093887211e-6)))
        Term2 = -0.1562499995e-1+y*(0.1430488765e-3+y*(-0.6911147651e-5+y*(0.7621095161e-6-y*0.9349945152e-7)))

        ans = math.sqrt(0.636619772/x)*(math.sin(X_0)*Term1+z*math.cos(X_0)*Term2)
        return np.float64(ans)
    

@cuda.jit(complex128(float32),fastmath = True,device = True)   
def hankel1_0_device(x):
    '''Calculates the zeroth order hankel function of the first kind at a positive real scalar on GPU
        Parameters:
        -----------
        x (float32) - Points to evaluate the hankel function at
        Returns:
        --------
        H (complex128) - Value of the zeroth-order hankel function of the first kind passed for output
    '''
    H = bessj0(x)+1j*bessy0(x)
    return  H


@cuda.jit(void(complex128[:,:,:],int64,int64,float32,float32,float32[:]),fastmath = True)
def Greens_Function_G_Kernel(E_tensor,Q,L,R_mic,R_speaker,K_arr):
    '''Compute the greens function of the helmholtz equation due to a speaker at the position (R_speaker,2*l*pi/L) where L is the total number of speakers.
        Note that the green's function is (i/4)H_0^1(k||x-y_l||) where H denotes the hankel function
        Parameters:
        -----------
        E_tensor    (complex128[:,:,:]) - LxNxQ tensor of error microphone signals in the form E[l][q][n] = e_l(x_q,k_n)
        Q (int64) - Number of microphones
        R_mic (int64) - Radius of the microphone array
        R_speaker (int64) - Radius of the speaker array
        L (int64) - Total number of loudspeakers
        K_arr (float32[:]) - Array of wavenumbers to evaluate the signal at, is of shape (Nx1)

        Assumptions:
        ------------
        Each thread will handle a frequency at all loudspeaker positions and the blocks will be arranged in a grid QxW grid.
        We assume there will be 1024 loudspeakers at most
        
    '''
    shared_mic_positions = cuda.shared.array(2,dtype = float32) #Allocates an array in shared memory which stores the microphone positions
    Shared_Distance_array = cuda.shared.array(1024,dtype = float32) #Allocates an array in shared memory which will store the distances
                                                                    #between the q mic and all of the loudspeakers

    N = K_arr.shape[0] #pulls the number of wavenumbers

    L = cuda.blockDim.x #Pulls the size of the blocks which we assume to be number of loudspeakers
    l = cuda.threadIdx.x #Pulls the l value of the thread
    Q = cuda.gridDim.x #Pulls the number of microphones
    q = cuda.blockIdx.x #pulls the row of the block in the grid which is microphone position
    n  = cuda.blockIdx.y*L+l #pulls the flattened position of the thread in the first row of the CUDA grid

    
    #Has the first thread in each block calculate and store the microphone positions while all other threads wait
    if l==0:
        theta_mic = 2*np.pi*q/Q
        shared_mic_positions[0] = R_mic*math.cos(theta_mic) #x-position of q microphone
        shared_mic_positions[1] = R_mic*math.sin(theta_mic) #y-position of q microphone
    cuda.syncthreads()

    #Each thread will now compute one of the distances and store it in memory
    theta_speaker = 2*np.pi*l/L
    x_speaker = R_speaker*math.cos(theta_speaker) #x-position of speaker
    y_speaker = R_speaker*math.sin(theta_speaker) #y-position of speaker

    Shared_Distance_array[l] = np.float32(math.sqrt((shared_mic_positions[0]-x_speaker)**2+(shared_mic_positions[1]-y_speaker)**2)) #compute the distance between the mic and the speaker
    cuda.syncthreads()

    #Cancels is we are outside the grid. It is necessary to call this check here rather than earlier to ensure
    #that the entire Shared_Distance_Array gets populated before the E_tensor values are calculated
    if n>=N:
        return

    for i in range(L):
        E_tensor[i,n,q] = 1j*0.25*hankel1_0_device(K_arr[n]*Shared_Distance_array[i])
        

