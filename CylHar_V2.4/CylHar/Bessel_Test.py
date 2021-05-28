import numpy as np
import math
from numba import cuda, jit, complex64, int16, int64, float32, void
import timeit
import matplotlib.pyplot as plt 
from scipy.stats import linregress

from CylHar import Bessel_Controller as Bes

def computational_complexity_test_bessel():
    x_max_arr = np.linspace(0.1,10001,25,dtype=np.float32)
    CPU_Average_arr = np.zeros_like(x_max_arr)
    CPU_Std_dev_arr = np.zeros_like(x_max_arr)
    GPU_Average_arr = np.zeros_like(x_max_arr)
    GPU_Std_dev_arr = np.zeros_like(x_max_arr)
    index = 0

    for x_max in x_max_arr:
        x_arr = np.arange(0.01,x_max,0.01).astype(np.float32)

        print('x_max = ',x_max,' Bessel CPU test results:')
        times = np.zeros(10,dtype = np.float64)
        for i in range(10):
            start = timeit.default_timer()
            Bes.bessj(50,x_arr,CUDA=False) #we perform the test for a 50th order bessel function somewhat arbitrarily
            end = timeit.default_timer()
            times[i] = end-start
        average = np.average(times)
        print('average time = ',average,'\n')  
        CPU_Average_arr[index] = average #Store the average value of the runtime
        CPU_Std_dev_arr[index] = np.std(times)


        print('x_max = ',x_max,' Bessel CUDA test results:')
        times = np.zeros(10,dtype = np.float64)
        for i in range(10):
            start = timeit.default_timer()
            Bes.bessj(50,x_arr,CUDA=True) #we perform the test for a 50th order bessel function somewhat arbitrarily
            end = timeit.default_timer()
            times[i] = end-start
        average = np.average(times)
        print('average time = ',average,'\n')  
        GPU_Average_arr[index] = average #Store the average value of the runtime
        GPU_Std_dev_arr[index] = np.std(times)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(x_max_arr,CPU_Average_arr)
    CPU_average_line = mc*x_max_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(x_max_arr,GPU_Average_arr)
    GPU_average_line = mg*x_max_arr+yintg


    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(x_max_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(x_max_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(x_max_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(x_max_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('$x_{max}$')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Runtime required to calculate $J_m(x)$ over an array of positions')
    plt.legend()
    plt.show()

def computational_complexity_test_hankel():
    x_max_arr = np.linspace(0.1,10001,25,dtype=np.float32)
    CPU_Average_arr = np.zeros_like(x_max_arr)
    CPU_Std_dev_arr = np.zeros_like(x_max_arr)
    GPU_Average_arr = np.zeros_like(x_max_arr)
    GPU_Std_dev_arr = np.zeros_like(x_max_arr)
    index = 0

    for x_max in x_max_arr:
        x_arr = np.arange(0.01,x_max,0.01).astype(np.float32)

        print('x_max = ',x_max,' Hankel CPU test results:')
        times = np.zeros(10,dtype = np.float64)
        for i in range(10):
            start = timeit.default_timer()
            Bes.hankel1_0(x_arr,CUDA=False) #we perform the test for a 50th order bessel function somewhat arbitrarily
            end = timeit.default_timer()
            times[i] = end-start
        average = np.average(times)
        print('average time = ',average,'\n')  
        CPU_Average_arr[index] = average #Store the average value of the runtime
        CPU_Std_dev_arr[index] = np.std(times)


        print('x_max = ',x_max,' Hankel CUDA test results:')
        times = np.zeros(10,dtype = np.float64)
        for i in range(10):
            start = timeit.default_timer()
            Bes.hankel1_0(x_arr,CUDA=True) #we perform the test for a 50th order bessel function somewhat arbitrarily
            end = timeit.default_timer()
            times[i] = end-start
        average = np.average(times)
        print('average time = ',average,'\n')  
        GPU_Average_arr[index] = average #Store the average value of the runtime
        GPU_Std_dev_arr[index] = np.std(times)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(x_max_arr,CPU_Average_arr)
    CPU_average_line = mc*x_max_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(x_max_arr,GPU_Average_arr)
    GPU_average_line = mg*x_max_arr+yintg
    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(x_max_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(x_max_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(x_max_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(x_max_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('$x_{max}$')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Runtime required to calculate $H_0^{(1)}(x)$ over an array of positions')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    computational_complexity_test_hankel()


