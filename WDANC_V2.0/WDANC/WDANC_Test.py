import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.stats import linregress
from WDANC import WDANC_G as G
from WDANC import WDANC_C as C
from CylHar import WaveField_G as WF


def Gradient_Descent_Complexity():
    max_run_count = 10
    #Q_arr = np.arange(16,272,16,dtype = np.int64)
    #L_arr = 3*Q_arr
    Q_arr = np.arange(4,96,8,dtype = np.int64)
    L_arr = 3*Q_arr

    CPU_times_arr = np.zeros(Q_arr.shape[0],dtype = np.float64)
    CPU_std_arr = np.zeros(Q_arr.shape[0],dtype = np.float64)
    GPU_times_arr = np.zeros(Q_arr.shape[0],dtype = np.float64)
    GPU_std_arr = np.zeros(Q_arr.shape[0],dtype = np.float64)

    Params_file_cached = open('Test_Params.txt','r')
    Params_file_cached_content = Params_file_cached.readlines()
    Params_file_cached.close()
    for i in range(Q_arr.shape[0]):
        Q_str = 'Q = {'+str(Q_arr[i])+'}                \n'
        L_str = 'L = {'+str(L_arr[i])+'}                \n'
        Params_file_content = Params_file_cached_content[:2]+[Q_str]+[Params_file_cached_content[3]]+[L_str]+Params_file_cached_content[5:]

        Params_file = open('Test_Params.txt','w')
        Params_file.writelines(Params_file_content)
        Params_file.close()

        CPU_times = np.zeros(max_run_count,dtype = np.float32)
        GPU_times = np.zeros(max_run_count,dtype = np.float32)
        for run_number in range(max_run_count):
            start = default_timer()
            C.WDANC_Controller(100, 1)
            end = default_timer()
            CPU_times[run_number]= end-start

            start = default_timer()
            G.WDANC_Controller(100, 1)
            end = default_timer()
            GPU_times[run_number]= end-start

        average = np.average(CPU_times)
        print('Q = '+str(Q_arr[i])+' CPU average time = ',average,'\n')  
        CPU_times_arr[i] = average #Store the average value of the runtime
        CPU_std_arr[i] = np.std(CPU_times)

        average = np.average(GPU_times)
        print('Q = '+str(Q_arr[i])+' GPU average time = ',average,'\n')  
        GPU_times_arr[i] = average #Store the average value of the runtime
        GPU_std_arr[i] = np.std(GPU_times)

    Params_file = open('Test_Params.txt','w')
    Params_file.writelines(Params_file_cached_content)
    Params_file.close()

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(Q_arr,CPU_times_arr)
    CPU_average_line = mc*Q_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(Q_arr,GPU_times_arr)
    GPU_average_line = mg*Q_arr+yintg
    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(Q_arr,CPU_times_arr,yerr = CPU_std_arr,fmt = 'kx',ecolor = 'grey')
    ax.plot(Q_arr,CPU_average_line,'b',label = 'CPU')
    ax.errorbar(Q_arr,GPU_times_arr,yerr = GPU_std_arr,fmt = 'ko',ecolor = 'grey')
    ax.plot(Q_arr,GPU_average_line,'g',label = 'CUDA')
    ax.set_xlabel('Q')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('End to end gradient descent runtime for 100 iterations for varying Q')
    plt.legend()
    plt.show()

def Gradient_Descent_Complexity_2():
    max_run_count = 5
    Iterations_arr = np.arange(1,257,16,dtype = np.int64)

    CPU_times_arr = np.zeros(Iterations_arr.shape[0],dtype = np.float64)
    CPU_std_arr = np.zeros(Iterations_arr.shape[0],dtype = np.float64)
    GPU_times_arr = np.zeros(Iterations_arr.shape[0],dtype = np.float64)
    GPU_std_arr = np.zeros(Iterations_arr.shape[0],dtype = np.float64)

    Params_file_cached = open('Test_Params.txt','r')
    Params_file_cached_content = Params_file_cached.readlines()
    Params_file_cached.close()
    for i in range(Iterations_arr.shape[0]):
        CPU_times = np.zeros(max_run_count,dtype = np.float32)
        GPU_times = np.zeros(max_run_count,dtype = np.float32)
        for run_number in range(max_run_count):
            start = default_timer()
            C.WDANC_Controller(Iterations_arr[i], 1)
            end = default_timer()
            CPU_times[run_number]= end-start

            start = default_timer()
            G.WDANC_Controller(Iterations_arr[i], 1)
            end = default_timer()
            GPU_times[run_number]= end-start

        average = np.average(CPU_times)
        print('Iteration# '+str(Iterations_arr[i])+' CPU average time = ',average,'\n')  
        CPU_times_arr[i] = average #Store the average value of the runtime
        CPU_std_arr[i] = np.std(CPU_times)

        average = np.average(GPU_times)
        print('Iteration# '+str(Iterations_arr[i])+' GPU average time = ',average,'\n')  
        GPU_times_arr[i] = average #Store the average value of the runtime
        GPU_std_arr[i] = np.std(GPU_times)

    Params_file = open('Test_Params.txt','w')
    Params_file.writelines(Params_file_cached_content)
    Params_file.close()

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(Iterations_arr,CPU_times_arr)
    CPU_average_line = mc*Iterations_arr+yintc
    print('CPU Slope = ',mc, ' s/iteration')
    mg, yintg, rg, pg, stderrg = linregress(Iterations_arr,GPU_times_arr)
    GPU_average_line = mg*Iterations_arr+yintg
    print('CPU Slope = ',mg, ' s/iteration')
    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(Iterations_arr,CPU_times_arr,yerr = CPU_std_arr,fmt = 'kx',ecolor = 'grey')
    ax.plot(Iterations_arr,CPU_average_line,'b',label = 'CPU')
    ax.errorbar(Iterations_arr,GPU_times_arr,yerr = GPU_std_arr,fmt = 'ko',ecolor = 'grey')
    ax.plot(Iterations_arr,GPU_average_line,'g',label = 'CUDA')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Cumuluative gradient descent runtime for Q = 64, L=192, against iteration number')
    plt.legend()
    plt.show()

def NoiseReductionAgainstFrequency():
    CALPHA,CK_arr = C.WDANC_Controller(250,1.0)

    CNoise_level_array = np.zeros_like(CK_arr).astype(np.float64)

    for i in range(CK_arr.shape[0]):
        Noise_level_ini = WF.noise_level(CALPHA[0,i],CK_arr[i])
        Noise_level_fin = WF.noise_level(CALPHA[-1,i],CK_arr[i]) 
        CNoise_level_array[i] = 10*np.log10(Noise_level_fin/Noise_level_ini)

    GALPHA,GK_arr = G.WDANC_Controller(250,1.0)

    GNoise_level_array = np.zeros_like(GK_arr).astype(np.float64)

    for i in range(1,GK_arr.shape[0]):
        Noise_level_ini = WF.noise_level(GALPHA[0,i],GK_arr[i])
        Noise_level_fin = WF.noise_level(GALPHA[-1,i],GK_arr[i])
        GNoise_level_array[i] = 10*np.log10(Noise_level_fin/Noise_level_ini)




    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots()
    ax.plot(GK_arr*343/(2*np.pi),GNoise_level_array,color = 'b',label = 'CPU')
    ax.plot(CK_arr*343/(2*np.pi),CNoise_level_array,color = 'g',linestyle = '--',label = 'CUDA')
    ax.set_ylim(-300,0)
    ax.set_xlabel('Frequency  (Hz)')
    ax.set_ylabel('Noise Reduction (db)')
    ax.set_title('Noise level reduction against frequency after 100 iterations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Gradient_Descent_Complexity_2()



