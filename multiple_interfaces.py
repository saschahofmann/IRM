#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def multiple_interfaces(h, w,n0,n1,n2,n3, dm, S , D):
    r12 = (n1 - n2)/ (n1+n2)
    r23 = (n2- n3)/ (n2+n3)
    gamma = r23/r12*(1-r12**2)
    delta = 4*np.pi*n2 * dm/w
    h0 = - w/(4*np.pi*n1)*np.arctan(gamma * np.sin(delta)/(1 + gamma*np.cos(delta)))
    I = S/2 - D * np.cos(4*np.pi/w*(h-h0))
    return I
def func(X, y0, A, h0 ):
    h, n1, w1 = X
    return y0 - A *np.cos(4*n1*np.pi/w1*(h-h0))
    
# Refractive Index and wave lengths
n0 = 1.515
n1 = 1.34
n2 = 1.486
n3 = 1.37
w1 = 488
w2 = 635
dm = 4 
r01 = (n0 - n1)/(n0 + n1)
r12 = (n1 - n2)/ (n1+n2)
r23 = (n2- n3)/ (n2+n3)
h = np.linspace(0,200,2000)
R_norm = np.absolute( 1 + (1 - r01**2)*np.exp(4j * np.pi* n1 *h/w1)*((r12 + r23*(1 - 
                   r12**2)*np.exp(4j * np.pi *n2*dm/w1)))/r01)**2 - 1
X = (h, n1, w1)
popt, pcov = curve_fit(func, X, R_norm)
y01, A1, h01 = popt
X2 =(h, n1, w2)
R_norm2 = np.absolute( 1 + (1 - r01**2)*np.exp(4j * np.pi* n1 *h/w2)*((r12 + r23*(1 - 
                   r12**2)*np.exp(4j * np.pi *n2*dm/w2)))/r01)**2 - 1
popt, pcov = curve_fit(func, X, R_norm2)
y0 = r23/r12*(1-r12**2)
d2 = 4*np.pi*n2*dm/w1
A = 2*r12/r01*(r01**2-1)*np.sqrt(1 +  y0**2 + 2 *y0*np.cos(d2))
h0 = - w1/(4*np.pi*n1)*np.arctan(y0 * np.sin(d2)/(1 + y0*np.cos(d2)))


plt.figure(1)
plt.plot(R_norm, R_norm2)