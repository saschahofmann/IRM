# -*- coding: utf-8 -*-
"""Multiple Interefaces"""
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shapely.geometry as geom
# Multiple Interfaces
def func(X, y0, A, h0 ):
    h, n1, w = X
    return y0 - A *np.cos(4*n1*np.pi/w*(h-h0))

def multiple_interfaces(h, w,n0,n1,n2,n3, dm):
    r01 = (n0 - n1)/(n0 + n1)
    r12 = (n1 - n2)/ (n1+n2)
    r23 = (n2- n3)/ (n2+n3)
    R_norm = np.absolute( 1 + (1 - r01**2)*np.exp(4j * np.pi* n1 *h/w)*((r12 + r23*(1 - 
                       r12**2)*np.exp(4j * np.pi *n2*dm/w)))/r01)**2 - 1
    X = (h, n1, w)
    popt, pcov = curve_fit(func, X, R_norm)
    y0, A, h0 = popt                     
    return y0, A, h0

# Calculate Max/Min via Fresnel Coefficients
    
    
def MP(channel1, channel2, I11, I12, n0, n1, n2, n3 ,w1, w2, dm):    

    channel1 = (channel1-I11)/I11
    channel2 = (channel2- I12)/I12
  
    
    h = np.linspace(0,200,2000)

    y01, A1, h01 = multiple_interfaces(h, w1, n0, n1, n2, n3, dm)
    y02, A2, h02 = multiple_interfaces(h, w2, n0, n1, n2, n3, dm)
    X1 = (h, n1, w1)
    X2 = (h, n1, w2)
    I1= func(X1, y01, A1, h01)
    I2= func(X2, y02, A2, h02)
    
    
    
    
    z = np.column_stack((I1, I2, h))
    line = geom.LineString(z)
    point_on_line = np.zeros(channel1.shape)
    point_on_line2 = np.zeros(channel1.shape)
    height_img = np.zeros(channel1.shape)
    min1 = np.min(channel1) 
    min2 = np.min(channel2) 
    for i in range(channel1.shape[0]):
        print i
        for j in range(channel1.shape[1]):
            if channel1[i,j] >= min1 and channel2[i,j] >= min2:
                point = geom.Point(channel1[i,j], channel2[i,j])
                points = line.interpolate(line.project(point))
                point_on_line[i,j] = points.x
                point_on_line2[i,j] = points.y
                height_img[i,j] = points.z
    #height_img[height_img>65] = 65
    #height_img[height_img< 15]= 15           

    return height_img, I1, I2, channel1, channel2
                