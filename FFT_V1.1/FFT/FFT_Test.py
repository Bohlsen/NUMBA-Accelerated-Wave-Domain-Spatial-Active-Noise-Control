import FFT_Controller as FFT
from scipy.io import wavfile
import timeit
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
#tests the DFT for a particular test audio file

    print('CPU test results:')
    average = 0
    for i in range(10):
        start = timeit.default_timer()
        FFT.FFT_From_WAV('Test.wav',CUDA = False)
        end = timeit.default_timer()
        average += end-start
        print(end-start)
    print('average time = ',average/10,'\n')  


    print('CUDA test results:')
    average = 0
    for i in range(10):
        start = timeit.default_timer()
        FFT.FFT_From_WAV('Test.wav',CUDA = True)
        end = timeit.default_timer()
        average += end-start
        print(end-start)
    print('average time = ',average/10,'\n')  

