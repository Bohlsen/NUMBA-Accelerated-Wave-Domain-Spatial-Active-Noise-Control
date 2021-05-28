import numpy as np
import math
from numba import cuda, jit, complex128, int16, int64, float32, float64, void
import timeit
import matplotlib.pyplot as plt 

@jit(float64(float32),nopython=True,fastmath=True)
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

@jit(float64(float32),nopython = True,fastmath = True)
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


@jit(float64(float32),nopython=True,fastmath=True)
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

@jit(float64[:](int64,float32),nopython=True,fastmath=True)
def bessjm(m,x):
    '''Calculate the nth order bessel function of the first kind for a scalar by applying the recursive 
    approximations described in Numerical Recipes.
        Parameters:
        -----------
        m (int64) - order of the bessel function being called
        x (float32) - Input value for the function
        Assumptions:
        ------------
        The input is assumed to be positive as for the case of wave domain ANC this is acceptable.
        Returns:
        --------
        ans (float64[:]) - The value of all J_m's up to m for a fixed value of x
    '''
    if m==0:
        Js = np.zeros(m+1,dtype=np.float64)
        Js[0] = bessj0(x)
    elif m==1:
        Js = np.zeros(m+1,dtype=np.float64)
        Js[0] = bessj0(x)
        Js[1] = bessj1(x)
    else:
        Js = np.zeros(m+1,dtype=np.float64)
        Tox = np.float64(2.0/x) #We store this instead of always calculating it
        Js[0] = bessj0(x)
        Js[1] = bessj1(x)

        if x>m:#Applies the upwards recurrence for large values
            for n in range(1,m):
                Js[n+1] = n*Tox*Js[n]-Js[n-1]

        else:
            Js_temp = np.zeros(m+1,dtype=np.float64)
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
                if M<=m and (M>=2): #Saves the unnormalised sums
                    Js_temp[M] = J_nplus1
                M -= 1

            Sum = (2.0*Sum-J_n)
            for n in range(2,m+1):
                if Sum == np.float64(0.0):
                    Js[n] = 0.0
                else:
                    Js[n] = Js_temp[n]/Sum
            
    ans = Js
    
    return ans


@jit(float64[:,:](int64,float32[:]),nopython=True,fastmath=True)
def bessj(m,x_arr):
    '''Calculates the bessel functions of the first kind on CPU up to and including order m
        Parameters:
        -----------
        m (int64) - The order of the bessel function
        x_arr (float32[:]) - The input points for the bessel function to be evaluated at
        Returns:
        --------
        J (float32[:,:]) - The values of all of the positive bessel functions at each input point up to m
                               with the arangement J[x_n,m] = J_m(x_n)
    '''

    y_arr = np.zeros(shape = (x_arr.shape[0],m+1),dtype = np.float64)

    for i in range(x_arr.shape[0]):
        y_arr[i] = bessjm(m,x_arr[i])

    return y_arr

@jit(complex128[:](float32[:]),nopython = True,fastmath = True)
def hankel1_0(x_arr):
    '''Calculates the zeroth order hankel function of the first kind at an array of array of positive real input values
        Parameters:
        -----------
        x_arr (float32[:]) - Array of points to evaluate the hankel function at
        Returns:
        --------
        H (complex128[:]) - Array of values of the zeroth-order hankel function of the first kind
    '''
    H = np.zeros(x_arr.shape[0],dtype = np.complex128)

    for i in range(x_arr.shape[0]):
        H[i] = bessj0(x_arr[i])+1j*bessy0(x_arr[i])

    return H


