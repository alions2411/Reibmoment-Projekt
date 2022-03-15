# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:42:31 2022

@author: Alex
"""

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


df_Datensatz = pd.DataFrame()


Datensatz_rg12_load_20 = ['data/rg12_30_20.hdf5','data/rg12_60_20.hdf5',
                          'data/rg12_120_20.hdf5','data/rg12_240_20.hdf5']
df_Datensatz['rg12'] = Datensatz_rg12_load_20

Datensatz_rg14_load_20 = ['data/rg14_30_20.hdf5','data/rg14_60_20.hdf5',
                          'data/rg14_120_20.hdf5','data/rg14_240_20.hdf5']
df_Datensatz['rg14'] = Datensatz_rg14_load_20

Datensatz_rg15_load_20 = ['data/rg15_30_20.hdf5','data/rg15_60_20.hdf5',
                          'data/rg15_120_20.hdf5','data/rg15_240_20.hdf5']
df_Datensatz['rg15'] = Datensatz_rg15_load_20

Datensatz_rg16_load_20 = ['data/rg16_30_20.hdf5','data/rg16_60_20.hdf5',
                          'data/rg16_120_20.hdf5','data/rg16_240_20.hdf5']
df_Datensatz['rg16'] = Datensatz_rg16_load_20

# rg12 nach Lasten sortiert
df_Datensatz_rg = pd.DataFrame()
Datensatz_rg12_20 = ['data/rg12_30_20.hdf5','data/rg12_60_20.hdf5','data/rg12_120_20.hdf5','data/rg12_240_20.hdf5']
Datensatz_rg12_40 = ['data/rg12_60_40.hdf5','data/rg12_120_40.hdf5','data/rg12_240_40.hdf5']
Datensatz_rg12_80 = ['data/rg12_120_80.hdf5','data/rg12_240_80.hdf5']

#rg14 nach Last sortiert
Datensatz_rg14_20 = ['data/rg14_30_20.hdf5','data/rg14_60_20.hdf5','data/rg14_120_20.hdf5','data/rg14_240_20.hdf5']
Datensatz_rg14_40 = ['data/rg14_60_40.hdf5','data/rg14_120_40.hdf5','data/rg14_240_40.hdf5']
Datensatz_rg14_80 = ['data/rg14_120_80.hdf5','data/rg14_240_80.hdf5']

#rg15 nach Last sortiert
Datensatz_rg15_20 = ['data/rg15_30_20.hdf5','data/rg15_60_20.hdf5','data/rg15_120_20.hdf5','data/rg15_240_20.hdf5']
Datensatz_rg15_40 = ['data/rg15_60_40.hdf5','data/rg15_120_40.hdf5','data/rg15_240_40.hdf5']
Datensatz_rg15_80 = ['data/rg15_120_80.hdf5','data/rg15_240_80.hdf5']

#rg16 nach Last sortiert
Datensatz_rg16_20 = ['data/rg16_30_20.hdf5','data/rg16_60_20.hdf5','data/rg16_120_20.hdf5','data/rg16_240_20.hdf5']
Datensatz_rg16_40 = ['data/rg16_60_40.hdf5','data/rg16_120_40.hdf5','data/rg16_240_40.hdf5']
Datensatz_rg16_80 = ['data/rg16_120_80.hdf5','data/rg16_240_80.hdf5']

# nach Last sortierte Datensätze zusammenfügen

#df_Datensatz_rg['rg12'] = Datensatz_rg12_20 + Datensatz_rg12_40 + Datensatz_rg12_80
#df_Datensatz_rg['rg14'] = Datensatz_rg14_20 + Datensatz_rg14_40 + Datensatz_rg14_80
#df_Datensatz_rg['rg15'] = Datensatz_rg15_20 + Datensatz_rg15_40 + Datensatz_rg15_80
df_Datensatz_rg['rg16'] = Datensatz_rg16_20 + Datensatz_rg16_40 + Datensatz_rg16_80


rpm = [30,60,120,240]

df_y = pd.DataFrame()
df_y_hat = pd.DataFrame()
df_Stribeck = pd.DataFrame()
df_time = pd.DataFrame()
df_R2 = pd.DataFrame()

df_y1 = pd.DataFrame()
df_y_hat1 = pd.DataFrame()
df_Stribeck1 = pd.DataFrame()
df_time1 = pd.DataFrame()
df_R21 = pd.DataFrame()



for i in range(9):
        y1, y_hat1, Stribeck1, time1 = Identifikation(df_Datensatz_rg['rg16'].loc[i])
        df_y1[i] = y1
        df_y_hat1[i] = y_hat1
        df_Stribeck1[i] = Stribeck1
        df_R21[i] = R_square(y1, y_hat1)
        df_time1[i] = time1

# =============================================================================
# Plots
# =============================================================================


plt.figure(18)
plt.plot(df_Stribeck1.iloc[0,0:4],df_Stribeck1.iloc[1,0:4])
plt.plot(df_Stribeck1.iloc[0,4:7],df_Stribeck1.iloc[1,4:7])
plt.plot(df_Stribeck1.iloc[0,7:9],df_Stribeck1.iloc[1,7:9])
plt.grid()
plt.xlabel('Drehzahl in 1/min')
plt.ylabel('Drehmoment in Nm')
plt.legend(['Last = 20 bar','Last = 40 bar','Last = 80 bar'])
plt.title('Stribeck Kurven Lager rg16')
plt.savefig('Stribeck_rg16.pdf')
