#!/usr/bin/env python2
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import shapely.geometry as geom
# Min-Max-Method
def IncidentIntensity(I1, n0, n1):
    """Calculates the theoretical incident intensity based on the background
    intensity"""
    inc_int = I1/((n1 - n0)/(n1 + n0))**2
    return inc_int

def get_Max_and_Min(I1, n0, n1, n2):
    """Calculates the min and maximum for 2 interfaces based on the
    IncidentIntensity"""

    I0 = IncidentIntensity(I1, n0, n1)
    I2 = (I0-I1)*((n1-n2)/(n1+n2))**2
    I_max = I1 + I2 + 2 * np.sqrt(I1*I2)
    I_min = I1 + I2 - 2 * np.sqrt(I1*I2)
    return I_max, I_min

def dw(h, D1, S1, S2, D2, w1, w2, n):
    I1 = 0.5*(S1 - D1*np.cos(4*np.pi*n/ w1 * h ))
    I2 = 0.5*(S2 - D2*np.cos(4*np.pi*n/ w2 * h ))
    return I1,I2    

def minmaxmethod(channel1, channel2, I11, I12, n0, n1, n3 ,w1, w2):
    I1_max, I1_min = get_Max_and_Min(I11, n0, n1, n3)
    I2_max, I2_min = get_Max_and_Min(I12, n0, n1, n3)
    S1, D1, S2, D2 = I1_max + I1_min, I1_max - I1_min,I2_max + I2_min, I2_max - I2_min,
    
    h = np.linspace(0,200,2000)
    I1, I2 = dw(h, S1, D1, S2, D2,w1, w2, n1) 
        
    z = np.column_stack((I1, I2, h))
    line = geom.LineString(z)
    point_on_line = np.zeros(channel1.shape)
    point_on_line2 = np.zeros(channel1.shape)
    height_img = np.zeros((4,4))
#    height_img = np.zeros(channel1.shape)
#    min1 = np.min(channel1) 
#    min2 = np.min(channel2) 
#
#    for i in range(channel1.shape[0]):
#        print i
#        for j in range(channel1.shape[1]):
#            if channel1[i,j] >= min1 and channel2[i,j] >= min2:
#                point = geom.Point(channel1[i,j], channel2[i,j])
#                points = line.interpolate(line.project(point))
#                point_on_line[i,j] = points.x
#                point_on_line2[i,j] = points.y
#                height_img[i,j] = points.z
#    #height_img[height_img>65] = 65
#    #height_img[height_img< 15]= 15           
    

    return height_img, I1, I2, channel1, channel2


