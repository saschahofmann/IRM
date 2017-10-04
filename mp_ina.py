#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import shapely.geometry as geom
from scipy.optimize import curve_fit

# Multiple Interfaces
def func(X, y0, A, h0):

    h, n1, w, INA = X
    a = np.arcsin(INA/n1)
    y = 4* np.pi/w*h*np.sin(a/2)**2
    #return y0 - A *np.sin(y)/y*np.cos(4*n1*np.pi/w*(h*(1-np.sin(a/2)**2)-h0))
    return y0 - A *np.cos(4*n1*np.pi/w*(h-h0))

def mp_ina(h, w,n0,n1,n2,n3, dm, INA):
    r01 = (n0 - n1)/(n0 + n1)
    r12 = (n1 - n2)/ (n1+n2)
    r23 = (n2- n3)/ (n2+n3)
    R_norm = np.absolute( 1 + (1 - r01**2)*np.exp(4j * np.pi* n1 *h/w)*((r12 + r23*(1 - 
                       r12**2)*np.exp(4j * np.pi *n2*dm/w)))/r01)**2 - 1
    X = (h, n1, w, INA)
    popt, pcov = curve_fit(func, X, R_norm)
    y0, A, h0 = popt                     
    return y0, A, h0
    
    
def inamp_main(channel1, channel2, I11, I12, n0, n1, n2, n3 ,w1, w2, dm, INA):    

    channel1 = (channel1-I11)/I11
    channel2 = (channel2- I12)/I12
  
    
    h = np.linspace(0.01,200,2000)

    y01, A1, h01 = mp_ina(h, w1, n0, n1, n2, n3, dm, INA)
    y02, A2, h02 = mp_ina(h, w2, n0, n1, n2, n3, dm, INA)
    X1 = (h, n1, w1, INA)
    X2 = (h, n1, w2, INA)
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
                

