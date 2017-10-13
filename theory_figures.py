#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:47:45 2017

@author: paluchlab
"""
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import gridspec

def dw(h, D1, S1, S2, D2, w1, w2, n):
    I1 = 0.5*(S1 - D1*np.cos(4*np.pi*n/ w1 * h ))
    I2 = 0.5*(S2 - D2*np.cos(4*np.pi*n/ w2 * h ))
    return I1,I2


n0 = 1.515
n1 = 1.34
n2 = 1.48
n3 = 1.42
w1 = 488
w2 = 642

# Theory plots
h = np.linspace(0, 250, 1000)
I1, I2 = dw(h, 1, 1, 1, 1, w1, w2, n1 )

#f, ax = plt.subplots(1,2,figsize=(30,5))
##gs = gridspec.GridSpec(2,1, width_ratios = [1])
##plt.suptitle('Periodicity summand substituted'+ '\n' + r'$2 I2 = S2 - D2* cos(\frac{4 \pi n h (w1^2 -w2^2)}{w1^2w2} +\frac{w1}{w2}arccos(\frac{S1 - 2I1}{D1}))$')
##ax1=plt.subplot(gs[0])
#ax[0].plot(h, I1, c = "#369E4B")# 024EA0
#ax[0].plot(h, I2, c = "#be1621")
#ax[0].set_xlim(0,h.max())
#ax[0].set_ylim(0,1.1)
#ax[0].set_xlabel(r'$\lambda$ [nm]')
#ax[0].set_ylabel('Normalised Intensity')
##plt.subplot(gs[1], aspect = 'equal', adjustable='box-forced')
#ax[1].plot(I1, I2)
#ax[1].set_aspect(1)
#
#ind = np.where(h == 100)[0][0]
#
#ind2 = np.where(h == 200)[0][0]
#ind3 = np.where(h == 300)[0][0]
#ax[1].annotate('100', xy = (I1[ind], I2[ind]) )
#ax[1].annotate('200', xy = (I1[ind2], I2[ind2]))
#ax[1].annotate('300', xy = (I1[ind3], I2[ind3]))
#plt.xlabel("Intensity 1")
#plt.ylabel("Intensity 2")
#plt.savefig('irm_theory.pdf', format='pdf', dpi=1000)

plt.figure(1)
plt.plot(h, I1, c = "#369E4B", label = str(w1) +' nm')# 024EA0
plt.plot(h, I2, c = "#be1621", label = str(w2) +' nm' )
#plt.axes().set_aspect(1.,)
plt.xlim(0,h.max())
plt.ylim(0,1.1)
plt.xlabel('d [nm]')
plt.ylabel('Normalised Intensity')
plt.legend(loc = 1)
plt.savefig('irm_twosine.pdf', format='pdf', dpi=1000)

plt.figure(2)
n1 = [1.32, 1.33, 1.34, 1.36, 1.38]
color = ['b', 'r', 'g', 'violet', 'y']
for i in xrange(len(n1)):

    I1, I2 = dw(h, 1, 1, 1, 1, w1, w2, n1[i] )
    plt.plot(I1, I2, c =color[i])

    for j in [50,100,150, 200, 250]:
    
        ind = np.argmin(abs(j-h))
        #plt.annotate(str(i), xy= (I1[ind], I2[ind]),  xytext = (I1[ind]-0.025, I2[ind]- 0.05) )
        plt.plot(I1[ind], I2[ind], ls ='', marker ='+', markersize = 12,  c =color[i])
plt.axes().set_aspect('equal',)


#ind = np.where(h == 50)[0][0]
#ind2 = np.where(h == 100)[0][0]
#ind3 = np.where(h == 150)[0][0]
#ind4 = np.where(h == 200)[0][0]
#ind5 = np.where(h == 250)[0][0]
#plt.annotate('50', xy = (I1[ind], I2[ind]) )
#plt.annotate('100', xy = (I1[ind2], I2[ind2]))
#plt.annotate('150', xy = (I1[ind3], I2[ind3]))
#plt.annotate('200', xy = (I1[ind4], I2[ind4]))
#plt.annotate('250', xy = (I1[ind4], I2[ind4]))
plt.xlabel("Intensity 1")
plt.ylabel("Intensity 2")
plt.savefig('irm_twoint.pdf', format='pdf', dpi=1000)