# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:45:38 2021

@author: Alex
"""
import numpy as np
from numpy.linalg import inv
import h5py
import matplotlib.pyplot as plt


def Identifikation(Datensatz): 
    # =============================================================================
    # Daten einlesen
    # =============================================================================
    f = h5py.File(Datensatz, 'r')
    keys = list(f.keys())

    # buffer = Messkarte
    # cv = contact voltage
    load = f['load']
    set_load = f['set_load']
    rpm = f['rpm']
    set_rpm = f['set_rpm']
    signal = f['signal']
    mc = f['mc']
    time = f['time']
    
    # =============================================================================
    # Arrays erstellen
    # =============================================================================
    time = np.array(time)
    load = np.array(load)
    set_load = np.array(set_load)
    rpm = np.array(rpm)
    set_rpm = np.array(set_rpm)
    mc = np.array(mc)

    signal = np.array(signal)
    signal_rms = np.zeros(len(signal))
    for i in range(0,len(signal)):
        signal_rms[i] = np.sqrt(np.mean(signal[i]**2))
    
    rpm_set = np.max(set_rpm)
    


    # =============================================================================
    # System identifizieren  mit y(k) =  b*u1(k) + c*u2(k) + d*u3(k) + n(k)
    # =============================================================================

    N = len(mc)
    Phi = np.zeros((729,3))
    Phi[:,0] = signal_rms[1:N]
    Phi[:,1] = rpm[1:N]
    Phi[:,2] = load[1:N]

    y = mc[1:N] # 

    A = np.matmul(Phi.T,Phi) # Fischer Matrix
    theta_hat = np.matmul(inv(A),Phi.T)
    theta_hat = np.dot(theta_hat,y)
    y_hat = np.dot(Phi,theta_hat)
    
    # =============================================================================
    # Drehmoment und Reibung bestimmen
    # =============================================================================
    
    # Indizes für RMS Berechnung vom Drehmoment suchen
    array_start = []
    array_ende = []

    for i in time:
        if np.isclose(i,20,atol=0.1) == 1:
            array_start += [i]
        if np.isclose(i,40,atol=0.1) == 1:
            array_ende += [i]
         
            index_start = int(np.argwhere(time == array_start[0]))
            index_ende = int(np.argwhere(time == array_ende[0]))

    load_time = time[index_start:index_ende]
    
    # Drehmoment bestimmen
    #Mrollen = np.linspace(0.38,1.2,len(rpm))
    km = 0.8
    Mr = (y_hat*km) # - Mrollen[1:N]
    Mr = Mr[index_start:index_ende]
    Mr_rms = np.sqrt(np.mean(Mr**2))

    Stribeck = [rpm_set,Mr_rms]

    return y, y_hat, Stribeck, time



def R_square(y,y_hat):
    S1 = 0
    S2 = 0
    for i in range(0,len(y)):
        S1 = S1+(y_hat[i]-np.mean(y))**2
        S2 = S2+(y[i]-np.mean(y))**2
    R2 = S1/S2
    R2 = round(R2,2)
    return R2 
    
