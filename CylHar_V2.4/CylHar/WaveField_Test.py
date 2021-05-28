import timeit
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from CylHar import WaveField_G as G
from CylHar import WaveField_C as C


if __name__ == '__main__':
#tests the calculation time for the WaveField calculation function
    M = 100
    alpha = np.zeros(2*M+1,dtype=np.complex128)
    alpha[M] = 1
    for m in range(1,M+1):
        alpha[M+m] = 1/(m**2)
        alpha[M-m] = -1/(m**2)

     
    total = 0
    for run_number in range(1,302):
        start = timeit.default_timer()
        Z = C.Wave_Field_k(alpha,np.float32(run_number)/10)
        end = timeit.default_timer()
        total += end-start
        if run_number % 30 == 0:
            print(end-start)
    print('Total time of 300 runs  =',total)

    #Plots the wave field as a function of space as a visualisation
    r = np.linspace(0.1,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    R, Thet = np.meshgrid(r,thet)
    X = R*np.cos(Thet)
    Y = R*np.sin(Thet) 
    plt.rcParams['figure.figsize'] = (16,16)
    fig = plt.figure() 
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X,Y,np.real(Z), cmap = cm.get_cmap('coolwarm'), linewidth=0,antialiased = True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    

    print('GPU test results:')
    total = 0
    for run_number in range(1,302):
        start = timeit.default_timer()
        Z = G.Wave_Field_k(alpha,np.float32(run_number)/10)
        end = timeit.default_timer()
        total +=  end-start
        if run_number % 30 == 0:
            print(end-start)
    print('Total time of 300 runs  =',total)

    #Plots the wave field as a function of space as a visualisation
    r = np.linspace(0.1,1,1000).astype(np.float32)
    thet = np.linspace(0.0,2*np.pi,1000).astype(np.float32)
    R, Thet = np.meshgrid(r,thet)
    X = R*np.cos(Thet)
    Y = R*np.sin(Thet) 
    plt.rcParams['figure.figsize'] = (16,16)
    fig = plt.figure() 
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X,Y,np.real(Z), cmap = cm.get_cmap('coolwarm'), linewidth=0,antialiased = True)
    #ax.set_zlim(-1.5,1.5)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
