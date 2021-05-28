import numpy as np
import math
from numba import cuda, jit, complex128, int16, int64, float32,float64, void
import timeit
import matplotlib.pyplot as plt 

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

@cuda.jit(float64(float32),fastmath=True,device = True)
def bessj1(x):
    '''Calculate the first order bessel function of the first kind for a scalar by applying the approximations
    described in Numerical Recipes.
        Parameters:
        -----------
        x (float32) - Input value for the function
        Assumptions:
        ------------
        The input is assumed to be positive as for the case of wave domain ANC this is acceptable
        Returns:
        --------
        ans (float64) - The value of J_1 at the value x
    '''

    #For x<=8 we apply the rational fitting function given in (Hart,1968) and listed in Numerical Recipes
    if x <= 8:
        y = x*x
        Num = x*(72362614232.0+y*(-7895059235.0+y*(242396853.1+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))))
        Den = 144725228442.0+y*(2300535178.0+y*(18583304.74+y*(99447.43394+y*(376.9991397+y*1.0))))
        ans = Num/Den
        return np.float64(ans)

    #For x>8 we apply the approximation forms described in Numerical Recipes
    else:
        z= 8/x
        y = z*z
        X_1 = x-2.356194491 #x-pi/4

        Term1 = 1.0+y*(0.183105e-2+y*(-0.351639649e-4+y*(0.245752017e-5+y*(-0.240337019e-6))))
        Term2 = 0.04687499995+y*(-0.2002690873e-3+y*(0.8449199096e-5+y*(-0.88228987e-6+y*0.105787412e-6)))

        ans = math.sqrt(0.636619772/x)*(math.cos(X_1)*Term1-z*math.sin(X_1)*Term2)
        return np.float64(ans)

@cuda.jit(void(int64,float32[:],float64[:,:]),fastmath=True)
def bessj_kernel(m,x_arr,J):
    '''Calculates the mth order bessel function of the first kind on CUDA.
        Parameters:
        -----------
        m (int64) - The order of the bessel function
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at, should be a device array
        J (float64[:,:]) - An N by m device matrix of the value of the bessel function at each input point passed for output
        Assumptions:
        ------------
        We are only calling Bessel functions of positive order, as this is all that is required for WD-ANC
    '''

    bw = cuda.blockDim.x #Block width
    bx = cuda.threadIdx.x #position in block
    gx = cuda.blockIdx.x #position of block in grid
    pos = bx+gx*bw
    if pos >= x_arr.shape[0]:
        return
    x = x_arr[pos]

    #implements the base cases
    if m==0:
        J[pos,0] = bessj0(x)
    elif m==1:
        J[pos,0] = bessj0(x)
        J[pos,1] = bessj1(x)

    #performs iterative recursion to calculate the higher order Bessel functions
    else:
        Tox = 2.0/x #We store this instead of always calculating it
        J[pos,0] = bessj0(x)
        J[pos,1] = bessj1(x)
        

        if x>m: #applies the upwards recurrence algorithm for large x
            for n in range(1,m):
                  J[pos,n+1] = n*Tox*J[pos,n]-J[pos,n-1]
        
        else:
            #Applies the downwards recurrence Miller algorithm for small values
            ACC = 100
            M = np.int64(2*(m + np.int64(math.sqrt(ACC*m))/2))
            J_n = 1e-321
            J_nplus1 = 0.0
            Sum = 0.0
            even = True #Boolean stores if we are in an even or an odd state

            while M>=0:
                temp = M*Tox*J_n-J_nplus1
                J_nplus1 = J_n
                J_n = temp
                if even: #accumulates the sum for the normalisation constant
                    Sum += J_n
                even = not even #flips the boolean
                if (M<=m) and (M>=2): #Saves the unnormalised sums
                    J[pos,M] = J_nplus1
                M -= 1

            Sum = (2.0*Sum-J_n) #calculates the full normalisation constant

            for n in range(2,m+1):
                if math.isnan(Sum): #Sets to zero for nan values, since these are caused by being extremely close the the singularity at the origin
                    J[pos,n] = 0.0
                else:
                    J[pos,n] = J[pos,n]/Sum #Normalises the values

def bessj(m,x_arr):
    '''Calculates the bessel functions of the first kind on CUDA up to and including order m
        Parameters:
        -----------
        m (int64) - The order of the bessel function
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at
        Returns:
        --------
        J (float64[:,:]) - The values of all of the positive bessel functions at each input point up to m
                               with the arangement J[x_n,m] = J_m(x_n)
    '''
    TPB = 128 #the number of threads per block
    N = x_arr.shape[0] #The number of points for the function to be evaluated at
    BPG = N//TPB + 1

    #Moving arrays to GPU memory
    J_device = cuda.device_array(shape = (N,m+1),dtype=np.float64)
    x_device = cuda.to_device(x_arr)
    
    #calls our CUDA kernel
    bessj_kernel[BPG,TPB](m,x_device,J_device)

    J = np.empty(shape = (N,m+1),dtype=np.float64)
    J_device.copy_to_host(J)

    return J

@cuda.jit(void(float32[:],complex128[:]),fastmath = True)
def hankel1_0_kernel(x_arr,H):
    '''Calculates the zeroth order hankel function of the first kind at an array of array of positive real input values on GPU
        Parameters:
        -----------
        x_arr (float32[:]) - Array of points to evaluate the hankel function at
        H (complex128[:]) - Array of values of the zeroth-order hankel function of the first kind passed for output
    '''
    bw = cuda.blockDim.x #Block width
    bx = cuda.threadIdx.x #position in block
    gx = cuda.blockIdx.x #position of block in grid
    pos = bx+gx*bw #flattened position of thread in grid
    x = x_arr[pos]

    if pos >= x_arr.shape[0]: #Skips if we are outside of the maximum element in the array
        return

    H[pos] = bessj0(x)+1j*bessy0(x) #Each thread computes the hankel function separately at each point

def hankel1_0(x_arr):
    '''Calculates the zeroth order hankel function of the first kind at an array of array of positive real input values on GPU. This funciton
        is a wrapper for a full kernel call which actually calculates the values of the function
        Parameters:
        -----------
        x_arr (float32[:]) - Array of points to evaluate the hankel function at
        Returns:
        --------
        H (complex128[:]) - Array of values of the zeroth-order hankel function of the first kind passed for output
    '''
    N = x_arr.shape[0]
    TPB = 32 #Will assign 32 threads to every block
    BPG = N//TPB + 1 #Assigns enough blocks to ensure we run over all of the position values

    x_device  = cuda.to_device(x_arr) #Create array in GPU memory
    H_device = cuda.device_array(N,dtype = np.complex128)

    hankel1_0_kernel[BPG,TPB](x_device,H_device) #Call the CUDA kernel

    H = H_device.copy_to_host() #Copy the computed values back from GPU memory

    return H
   



    


