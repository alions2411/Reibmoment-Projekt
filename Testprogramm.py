# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:36:29 2021

@author: Alex
"""
import numpy as np
from numpy.linalg import inv
import h5py
import matplotlib.pyplot as plt
from Funktionen import Identifikation, R_square
import pandas as pd

df_y = pd.DataFrame()
df_y_hat = pd.DataFrame()
df_y_hat_single = pd.DataFrame()
df_Stribeck = pd.DataFrame()
df_time = pd.DataFrame()


Datensatz_rg12_load_20 = ['data/rg12_30_20.hdf5','data/rg12_60_20.hdf5',
                          'data/rg12_120_20.hdf5','data/rg12_240_20.hdf5']

Datensatz_rg14_load_20 = ['data/rg14_30_20.hdf5','data/rg14_60_20.hdf5',
                          'data/rg14_120_20.hdf5','data/rg14_240_20.hdf5']

Datensatz_rg15_load_20 = ['data/rg15_30_20.hdf5','data/rg15_60_20.hdf5',
                          'data/rg15_120_20.hdf5','data/rg15_240_20.hdf5']

Datensatz_rg16_load_20 = ['data/rg16_30_20.hdf5','data/rg16_60_20.hdf5',
                          'data/rg16_120_20.hdf5','data/rg16_240_20.hdf5']

R2 = np.zeros(4)


for i in range(4):
    y, y_hat, Stribeck, time = Identifikation(Datensatz_rg16_load_20[i])
    df_y[i] = y
    df_y_hat[i] = y_hat
    df_Stribeck[i] = Stribeck
    R2[i] = R_square(y, y_hat)
    df_time[i] = time
    print(R2[i])  

# =============================================================================
# Plots
# =============================================================================
for i in range(4):
    plt.figure(i)
    plt.plot(df_time[:-1][i],df_y[i])
    plt.plot(df_time[:-1][i],df_y_hat[i])
    plt.grid()
    plt.xlabel('Zeit in s')
    plt.ylabel('Amplitude in A')
    plt.legend(['Strom','geschätzter Strom'])
    plt.title('Strom und geschätzter Strom RG16')
    plt.savefig(f'Strom_RG16{i}.pdf')
    
    
