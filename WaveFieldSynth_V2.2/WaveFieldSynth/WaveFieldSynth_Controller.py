import numpy as np
import timeit
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from WaveFieldSynth import WaveFieldSynth_C as C
from WaveFieldSynth import WaveFieldSynth_G as G
from WaveFieldSynth import SoundFields
from FFT import FFT_Controller as F


#Unfinished, should probably be done properly
def Compute_Alpha_Test(Q,CUDA=False):
    f = np.linspace(0.1,850,256,dtype = np.float32)
    omega = 2*np.pi*f
    K_arr = omega/343 #computes omega/c
    N = K_arr.shape[0]

    Vocal_Spectrum , sample_rate, number_of_samples = F.FFT_From_WAV('Test.wav',CUDA=True) #We pull the non-zero part of the spectrum of my voice
    Vocal_Spectrum_Shortened = Vocal_Spectrum[:2048]

    E = SoundFields.plane_wave_fast(64,1.0,K_arr,Vocal_Spectrum_Shortened,[1/np.sqrt(2),1/np.sqrt(2)])+SoundFields.plane_wave(64,1.0,K_arr,Vocal_Spectrum_Shortened,[1/np.sqrt(2),-1/np.sqrt(2)])
    
    Alpha = compute_wavefield(E,K_arr,np.float32(1))
    plt.imshow(np.log10(np.abs(Alpha)**2))
    plt.show()
    WF.Plot_Wave_Field_k(Alpha[1024],K_arr[1024])