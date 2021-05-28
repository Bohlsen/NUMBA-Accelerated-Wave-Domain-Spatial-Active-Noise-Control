import timeit
import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
from scipy.optimize import curve_fit

from WaveFieldSynth import WaveFieldSynth_C as C
from WaveFieldSynth import WaveFieldSynth_G as G
from WaveFieldSynth import SoundFields
from CylHar import WaveField_G as WF
from FFT import FFT_Controller as F

def computational_complexity_test_f():
    fmax_arr = np.arange(128,1256,12,dtype=np.int64) #Array of many possible maximum frequencies numbers
    CPU_Average_arr = np.zeros_like(fmax_arr).astype(np.float64)
    CPU_Std_dev_arr = np.zeros_like(fmax_arr).astype(np.float64)
    GPU_Average_arr = np.zeros_like(fmax_arr).astype(np.float64)
    GPU_Std_dev_arr = np.zeros_like(fmax_arr).astype(np.float64)


    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Q = 128
    R = np.float32(1.0) #radius of the hypothetical microphone array
    Wave_Vector = np.array([1,0],dtype=np.float64) #direction of the plane wave
    Max_Run_Count =50

    Times_CPU = np.zeros(Max_Run_Count).astype(np.float64)
    Times_GPU = np.zeros(Max_Run_Count).astype(np.float64)

    index = 0
    for fmax in fmax_arr:
        #tests the calculation time for the compute_wavefield function
        
        f = np.linspace(0.1,fmax,1024,dtype = np.float32) #array of frequency waves
        omega = 2*np.pi*f
        K_arr = omega/343 #computes omega/c as the wavenumber array
    
        Vocal_Spectrum_Shortened = Vocal_Spectrum[:1024]#Chops the spectrum off to enough samples to fill the array, the actual content of this is irrelevant

        #Computing the original soundfield
        E = SoundFields.plane_wave_fast(Q,R,K_arr,Vocal_Spectrum_Shortened,Wave_Vector)

        for run_number in range(Max_Run_Count):

            start = timeit.default_timer() #Times each run
            C.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_CPU[run_number] = end-start

            start = timeit.default_timer() #Times each run
            G.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_GPU[run_number] = end-start

            

        print('CPU test results:',fmax)
        CPU_Average = np.average(Times_CPU)
        CPU_Average_arr[index] = CPU_Average
        CPU_Std_dev_arr[index] = np.std(Times_CPU)
        print('CPU: Average time over '+str(Max_Run_Count)+' runs  = ',CPU_Average)

        print('GPU test results:',fmax)
        GPU_Average = np.average(Times_GPU)
        GPU_Average_arr[index] = GPU_Average
        GPU_Std_dev_arr[index] = np.std(Times_GPU)
        print('GPU: Average time over '+str(Max_Run_Count)+' runs  = ',GPU_Average)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(fmax_arr,CPU_Average_arr)
    CPU_average_line = mc*fmax_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(fmax_arr,GPU_Average_arr)
    GPU_average_line = mg*fmax_arr+yintg


    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(fmax_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(fmax_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(fmax_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(fmax_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('$f_{max}$')
    ax.set_ylabel('Runtime (ms)')
    ax.set_ylim([0,max(CPU_Average_arr)*1e3+1])
    ax.set_title('Runtime required to perform the cylindrical harmonics decomposition at 1024 frequencies with Q=128')
    plt.legend()
    plt.show()

def computational_complexity_test_f_constant_density():
    fmax_arr = np.arange(128,512,4,dtype=np.int64) #Array of many possible maximum frequencies numbers
    CPU_Average_arr = np.zeros_like(fmax_arr).astype(np.float64)
    CPU_Std_dev_arr = np.zeros_like(fmax_arr).astype(np.float64)
    GPU_Average_arr = np.zeros_like(fmax_arr).astype(np.float64)
    GPU_Std_dev_arr = np.zeros_like(fmax_arr).astype(np.float64)


    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Q = 128
    R = np.float32(1.0) #radius of the hypothetical microphone array
    Wave_Vector = np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype=np.float64) #direction of the plane wave
    Max_Run_Count =50

    Times_CPU = np.zeros(Max_Run_Count).astype(np.float64)
    Times_GPU = np.zeros(Max_Run_Count).astype(np.float64)

    index = 0
    for fmax in fmax_arr:
        #tests the calculation time for the compute_wavefield function
        
        f = np.arange(0.1,fmax,0.25,dtype = np.float32) #array of frequency waves
        omega = 2*np.pi*f
        K_arr = omega/343 #computes omega/c as the wavenumber array
    
        Vocal_Spectrum_Shortened = Vocal_Spectrum[:1024]#Chops the spectrum off to enough samples to fill the array, the actual content of this is irrelevant

        #Computing the original soundfield
        E = SoundFields.plane_wave_fast(Q,R,K_arr,Vocal_Spectrum_Shortened,Wave_Vector)

        for run_number in range(Max_Run_Count):

            start = timeit.default_timer() #Times each run
            C.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_CPU[run_number] = end-start

            start = timeit.default_timer() #Times each run
            G.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_GPU[run_number] = end-start

            

        print('CPU test results:',fmax)
        CPU_Average = np.average(Times_CPU)
        CPU_Average_arr[index] = CPU_Average
        CPU_Std_dev_arr[index] = np.std(Times_CPU)
        print('CPU: Average time over '+str(Max_Run_Count)+' runs  = ',CPU_Average)

        print('GPU test results:',fmax)
        GPU_Average = np.average(Times_GPU)
        GPU_Average_arr[index] = GPU_Average
        GPU_Std_dev_arr[index] = np.std(Times_GPU)
        print('GPU: Average time over '+str(Max_Run_Count)+' runs  = ',GPU_Average)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    def quadratic_fit(x,a,b,c):
        return a*x**2+b*x+c
        

    Cparams,Cparams_errors = curve_fit(quadratic_fit,fmax_arr,CPU_Average_arr,)
    CPU_average_line = quadratic_fit(fmax_arr,Cparams[0],Cparams[1],Cparams[2])

    Gparams,Gparams_errors = curve_fit(quadratic_fit,fmax_arr,GPU_Average_arr,)
    GPU_average_line = quadratic_fit(fmax_arr,Gparams[0],Gparams[1],Gparams[2])


    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(fmax_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(fmax_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(fmax_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(fmax_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('$f_{max}$')
    ax.set_ylabel('Runtime (ms)')
    ax.set_ylim([0,max(CPU_Average_arr)*1e3+1])
    ax.set_title('Runtime required to perform cylindrical harmonics decomposition with Q=128')
    plt.legend()
    plt.show()

def computational_complexity_test_samples():
    sample_arr = np.arange(128,4096,64,dtype=np.int64) #Array of many possible sample numbers
    CPU_Average_arr = np.zeros_like(sample_arr).astype(np.float64)
    CPU_Std_dev_arr = np.zeros_like(sample_arr).astype(np.float64)
    GPU_Average_arr = np.zeros_like(sample_arr).astype(np.float64)
    GPU_Std_dev_arr = np.zeros_like(sample_arr).astype(np.float64)

    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Q = 128
    R = np.float32(1.0) #radius of the hypothetical microphone array
    Wave_Vector = np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype=np.float64) #direction of the plane wave
    Max_Run_Count =10

    Times_CPU = np.zeros(Max_Run_Count).astype(np.float64)
    Times_GPU = np.zeros(Max_Run_Count).astype(np.float64)

    index = 0
    for sample_number in sample_arr:
        #tests the calculation time for the compute_wavefield function
        
        f = np.linspace(0.1,1024,sample_number,dtype = np.float32) #array of frequency waves
        omega = 2*np.pi*f
        K_arr = omega/343 #computes omega/c as the wavenumber array
    
        Vocal_Spectrum_Shortened = Vocal_Spectrum[:sample_number]#Chops the spectrum off to enough samples to fill the array, the actual content of this is irrelevant

        #Computing the original soundfield
        E = SoundFields.plane_wave_fast(Q,R,K_arr,Vocal_Spectrum_Shortened,Wave_Vector)

        for run_number in range(Max_Run_Count):

            start = timeit.default_timer() #Times each run
            C.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_CPU[run_number] = end-start

            start = timeit.default_timer() #Times each run
            G.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_GPU[run_number] = end-start

            

        print('CPU test results:',sample_arr[index])
        CPU_Average = np.average(Times_CPU)
        CPU_Average_arr[index] = CPU_Average
        CPU_Std_dev_arr[index] = np.std(Times_CPU)
        print('CPU: Average time over '+str(Max_Run_Count)+' runs  = ',CPU_Average)

        print('GPU test results:',sample_arr[index])
        GPU_Average = np.average(Times_GPU)
        GPU_Average_arr[index] = GPU_Average
        GPU_Std_dev_arr[index] = np.std(Times_GPU)
        print('GPU: Average time over '+str(Max_Run_Count)+' runs  = ',GPU_Average)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(sample_arr,CPU_Average_arr)
    CPU_average_line = mc*sample_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(sample_arr,GPU_Average_arr)
    GPU_average_line = mg*sample_arr+yintg


    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(sample_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(sample_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(sample_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(sample_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('$n_{samples}$')
    ax.set_ylabel('Runtime (ms)')
    ax.set_ylim([0,max(CPU_Average_arr)*1e3+1])
    ax.set_title('Runtime required to perform cylindrical harmonics decomposition at up to 1024Hz with Q=128')
    plt.legend()
    plt.show()



def computational_complexity_test_Q():
    #tests the calculation time for the compute_wavefield function
    Q_arr = np.arange(1,1025,48,dtype=np.int64)
    CPU_Average_arr = np.zeros_like(Q_arr).astype(np.float64)
    CPU_Std_dev_arr = np.zeros_like(Q_arr).astype(np.float64)
    GPU_Average_arr = np.zeros_like(Q_arr).astype(np.float64)
    GPU_Std_dev_arr = np.zeros_like(Q_arr).astype(np.float64)

    sample_number = 2048 #number of samples in frequency
    R = np.float32(1.0) #radius of the hypothetical microphone array
    f = np.linspace(0.1,1024,sample_number,dtype = np.float32) #array of frequency waves
    omega = 2*np.pi*f
    K_arr = omega/343 #computes omega/c as the wavenumber array

    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Wave_Vector = np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype=np.float64) #direction of the plane wave
    Max_Run_Count =5

    Times_CPU = np.zeros(Max_Run_Count).astype(np.float64)
    Times_GPU = np.zeros(Max_Run_Count).astype(np.float64)

    index = 0
    for Q in Q_arr:
        #tests the calculation time for the compute_wavefield function
        
        f = np.linspace(0.1,1024,sample_number,dtype = np.float32) #array of frequency waves
        omega = 2*np.pi*f
        K_arr = omega/343 #computes omega/c as the wavenumber array
    
        Vocal_Spectrum_Shortened = Vocal_Spectrum[:sample_number]#Chops the spectrum off to enough samples to fill the array, the actual content of this is irrelevant

        #Computing the original soundfield
        E = SoundFields.plane_wave_fast(Q,R,K_arr,Vocal_Spectrum_Shortened,Wave_Vector)

        for run_number in range(Max_Run_Count):

            start = timeit.default_timer() #Times each run
            C.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_CPU[run_number] = end-start

            start = timeit.default_timer() #Times each run
            G.compute_wavefield(E,K_arr,R)
            end = timeit.default_timer()
            Times_GPU[run_number] = end-start

            

        print('CPU test results:',Q_arr[index])
        CPU_Average = np.average(Times_CPU)
        CPU_Average_arr[index] = CPU_Average
        CPU_Std_dev_arr[index] = np.std(Times_CPU)
        print('CPU: Average time over '+str(Max_Run_Count)+' runs  = ',CPU_Average)

        print('GPU test results:',Q_arr[index])
        GPU_Average = np.average(Times_GPU)
        GPU_Average_arr[index] = GPU_Average
        GPU_Std_dev_arr[index] = np.std(Times_GPU)
        print('GPU: Average time over '+str(Max_Run_Count)+' runs  = ',GPU_Average)

        index +=1

    #Performs the linear regression and genarates the associated exact points
    mc, yintc, rc, pc, stderrc = linregress(Q_arr,CPU_Average_arr)
    CPU_average_line = mc*Q_arr+yintc

    mg, yintg, rg, pg, stderrg = linregress(Q_arr,GPU_Average_arr)
    GPU_average_line = mg*Q_arr+yintg


    
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots()
    ax.errorbar(Q_arr,CPU_Average_arr*1e3,yerr = CPU_Std_dev_arr*1e3,fmt = 'kx',ecolor = 'grey')
    ax.plot(Q_arr,CPU_average_line*1e3,'b',label = 'CPU')
    ax.errorbar(Q_arr,GPU_Average_arr*1e3,yerr = GPU_Std_dev_arr*1e3,fmt = 'ko',ecolor = 'grey')
    ax.plot(Q_arr,GPU_average_line*1e3,'g',label = 'CUDA')
    ax.set_xlabel('Number of microphones (Q)')
    ax.set_ylabel('Runtime (ms)')
    ax.set_ylim([0,max(CPU_Average_arr)*1e3+1])
    ax.set_title('Runtime required to perform cylindrical harmonics decomposition at up to 1024Hz for variable Q')
    plt.legend()
    plt.show()


def reproduction_test():
    #tests the calculation time for the compute_wavefield function
    sample_number = 2048 #number of samples in frequency
    R = np.float32(1.0) #radius of the hypothetical microphone array
    f = np.linspace(0.1,1024,sample_number,dtype = np.float32) #array of frequency waves
    omega = 2*np.pi*f
    K_arr = omega/343 #computes omega/c as the wavenumber array
    Max_Run_Count =10

    Q = 64
    Wave_Vector = np.array([1.0,0.0])

    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Vocal_Spectrum_Shortened = Vocal_Spectrum[:2048]
    
    #Computing the original soundfield
    E = SoundFields.plane_wave_fast(Q,R,K_arr,Vocal_Spectrum_Shortened,Wave_Vector)
    #E = SoundFields.Greens_Function_C(Q,R,1.1,0,16,K_arr)#+SoundFields.Greens_Function(Q,R,1.1,4,16,K_arr,True)

    print('CPU test results:')
    total = 0
    for run_number in range(Max_Run_Count):
        start = timeit.default_timer() #Times each run
        Alpha = C.compute_wavefield(E,K_arr,R)
        end = timeit.default_timer()
        total += end-start
    print('CPU: Average time over '+str(Max_Run_Count)+' runs  = ',total/Max_Run_Count)

    WF.Plot_Wave_Field_k(Alpha[2047],K_arr[2047])
    

    #Computes the reproduced soundfield
    e_CPU = np.zeros(sample_number,dtype=np.complex128)
    index = 0
    for k in K_arr:
        Field = WF.Wave_Field_k(Alpha[index],k)
        e_CPU[index] = Field[0][999]
        index+=1

    #Produces the testing plots
    plt.rcParams['figure.figsize'] = (16,9)
    fig,ax = plt.subplots(2,2)
    
    ax[0][0].plot(f,10*np.log10(np.abs(np.transpose(E)[0])**2),'b') #Plots the original fourier domain signal at the first mic
    ax[0][0].set_title('Original Spectrum (CPU)')
    ax[0][0].set_xlabel('Frequency (Hz)')
    ax[0][0].set_ylabel('Energy (db)')
    ax[0][1].plot(f,10*np.log10(np.abs(e_CPU)**2),'g') #Plots the final fourier domain signal at the first mic
    ax[0][1].set_title('Reproduced Spectrum (CPU)')
    ax[0][1].set_xlabel('Frequency (Hz)')
    ax[0][1].set_ylabel('Energy (db)')

    
    print('GPU test results:')
    total = 0
    for run_number in range(Max_Run_Count):
        start = timeit.default_timer()
        Alpha = G.compute_wavefield(E,K_arr,R)
        end = timeit.default_timer()
        total += end-start
    print('GPU: Average time over '+str(Max_Run_Count)+' runs  = ',total/Max_Run_Count)

    #Computes the reproduced soundfield
    e_GPU = np.zeros(sample_number,dtype=np.complex128)
    index = 0
    for k in K_arr:
        Field = WF.Wave_Field_k(Alpha[index],k)
        e_GPU[index] = Field[0][999]
        index+=1


    ax[1][0].plot(f,10*np.log10(np.abs(np.transpose(E)[0])**2),'b') #Plots the original fourier domain signal at the first mic
    ax[1][0].set_title('Original Spectrum (GPU)')
    ax[1][0].set_xlabel('Frequency (Hz)')
    ax[1][0].set_ylabel('Energy (db)')
    ax[1][1].plot(f,10*np.log10(np.abs(e_GPU)**2),'g') #Plots the final fourier domain signal at the first mic
    ax[1][1].set_title('Reproduced Spectrum (GPU)')
    ax[1][1].set_xlabel('Frequency (Hz)')
    ax[1][1].set_ylabel('Energy (db)')
    print('Finished')

    plt.show()

if __name__ == '__main__':
    computational_complexity_test_f_constant_density()
