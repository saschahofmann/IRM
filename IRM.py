# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.optimize import fsolve
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import skimage.io as io
#import lmfit
from scipy.optimize import fminbound
from mpl_toolkits import axes_grid1
from scipy.stats import gaussian_kde
from scipy.misc import imsave


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def IncidentIntensity(I1, n0, n1):
    inc_int = I1/((n1 - n0)/(n1 + n0))**2
    return inc_int

def get_Max_and_Min(I1, n0, n1, n2):
    # Correct for PDMS-Glass interface

    I0 = IncidentIntensity(I1, n0, n1)
    I2 = (I0-I1)*((n1-n2)/(n1+n2))**2
    I_max = I1 + I2 + 2 * np.sqrt(I1*I2)
    I_min = I1 + I2 - 2 * np.sqrt(I1*I2)
    print I0, I2
    return I_max, I_min
    


def height(I, D, S, w, n):
    quotient = (S-2*I)/D
    h = np.zeros(I.shape)
    mask = abs(quotient ) <= 1
    h[mask] = w/(4*np.pi*n)*np.arccos(quotient[mask])
    return h


def dwh(I1, I2, D1, S1, D2, S2, w1, w2, n):
    h = w1**2 * w2/(4*np.pi*n*( w1**2 - w2**2)) * (np.arccos((S2 - 2*I2)/D2) 
                                 + w2/w1*np.arccos((S1 - 2*I1)/D1))
    return h

def multiple_interfaces(h, w,n0,n1,n2,n3, dm, S , D):
    r12 = (n1 - n2)/ (n1+n2)
    r23 = (n2- n3)/ (n2+n3)
    gamma = r23/r12*(1-r12**2)
    delta = 4*np.pi*n2 * dm/w
    h0 = - w/(4*np.pi*n1)*np.arctan(gamma * np.sin(delta)/(1 + gamma*np.cos(delta)))
    I = S/2 - D * np.cos(4*np.pi*n1/w*(h-h0))
    return I
    
def dw(h, D1, S1, S2, D2, w1, w2, n):
    I1 = 0.5*(S1 - D1*np.cos(4*np.pi*n/ w1 * h ))
    # Initial I2
    I2 = 0.5*(S2 - D2*np.cos(4*np.pi*n/ w2 * h ))
    # Formula derived in the paper
    # I2 = D1 - S2 * np.cos(w1/w2 * np.arccos((D2 - I1)/S1))
    # h from I1 substituted in I2
    #I3 = (S2 - D2 * np.cos(w1/w2 * np.arccos((S1 - 2*I1)/D1)))/2
    # I2 derived from substituting the periodicity summand
    #I3 = 0.5*(S2 - D2 * np.cos(4*np.pi*n*h*(w1**2 -w2**2)/(w1**2*w2)
    #                           + w2/w1 * arccos((S1 - 2*I1)/D1, h, w1/(4*n))))
    return I1,I2

def arccos(x, h, p):
    y = np.arccos(x)
    for i in range(1,int(round(h.max()/p + 0.5))):
        if i%2 != 0:
            y[h>= i * p] = (i+1) * np.pi - np.arccos(x[h>= i* p])
        else:
            y[h>= i * p] = i * np.pi + np.arccos(x[h>= i*p])

    return y

def theoretic_Intensity(h,n, S, D, w):
    return  0.5*(S - D*np.cos(4*np.pi*n/ w * h ))

def minimize_height(w1, w2, I1, I2, S1, D1, S2, D2, n):
    k = np.outer(np.arange(6), np.ones(6))
    h1 = w1/(4*np.pi* n)*(np.arccos((S1 - 2*I1)/D1) + k * w1/(2*n))
    h2 = w2/(4*np.pi* n)*(np.arccos((S2 - 2*I2)/D2) + k.T * w2/(2*n))
    min_ind = np.unravel_index(np.argmin(abs(h1 - h2)), h1.shape)
    h = (h1[min_ind]+ h2[min_ind])/2
    return h, abs(k[min_ind] )

def mp_get_Max_and_Min(I1, n0, n1, n2, n3):

    I0 = IncidentIntensity(I1, n0, n1)
    I2 = (I0-I1)*((n1-n2)/(n1+n2))**2
    I3 = (I0 - I1 - I2)*((n2-n3)/(n2+n3))**2
    I_max = I1 + I2 + I3 + 2 * np.sqrt(I1*I2) + 2 * np.sqrt(I1*I3) + 2*np.sqrt(I2*I3)
    I_min = I1 + I2 - (2 * np.sqrt(I1*I2) + 2 * np.sqrt(I1*I3) + 2*np.sqrt(I2*I3))
    return I_max, I_min

def res(h, I1, I2, S1, D1, S2, D2, w1, w2, n):
    res = np.sqrt((I1 - theoretic_Intensity(h, n1, S1, D1, w1))**2 
                        + (I2 - theoretic_Intensity(h, n, S2, D2, w2))**2)
    return res
# Refractive Index and wave lengths
n0 = 1.515
n1 = 1.34
n2 = 1.486
n3 = 1.37
w1 = 488
w2 = 635
dm = 4 

# Calculate Max/Min via Fresnel Coefficients

I11 = 1493.683
I12 = 1446.922
int_max1, int_min1 = mp_get_Max_and_Min(I11, n0, n1, n2, n3)
int_max2, int_min2 = mp_get_Max_and_Min(I12, n0, n1, n2, n3)

# Image In-read
directory = 'Height Measurements/'
filename = "1209_6x8_WT_006b.tif"
original = io.imread(directory + filename)
channel1 = original[0]
channel2 = original[1]
"""
# Get Min/Max from Image
int_min1 = np.min(channel1)
int_max1 = np.max(channel1)
int_min2 = np.min(channel2)
int_max2 = np.max(channel2)
"""
D1 = int_max1 - int_min1
S1 = int_max1 + int_min1
D2 = int_max2 - int_min2
S2 = int_max2 + int_min2


# Distance with Shapely
import shapely.geometry as geom

h = np.linspace(0,200,2000)
#I1, I2 = dw(h, D1, S1,S2, D2, w1, w2, n1)
I1 = multiple_interfaces(h,w1, n0, n1, n2, n3, dm, S1, D1)
I2 = multiple_interfaces(h,w2, n0, n1, n2, n3, dm, S2, D2)

plt.figure(1)
plt.plot(h, I1)
plt.plot(h, I2)
plt.show()
"""
I1, I2 = dw(h, 3300, 4800,5200, 3650, w1, w2, n1)
x = channel1.flatten()
y = channel2.flatten()
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=2, edgecolor='')
ax.plot(I1,I2)
"""
z = np.column_stack((I1, I2, h))
line = geom.LineString(z)
point_on_line = np.zeros(channel1.shape)
point_on_line2 = np.zeros(channel1.shape)
height_img = np.zeros(channel1.shape)
for i in range(channel1.shape[0]):
    print i
    for j in range(channel1.shape[1]):
        if channel1[i,j] >= int_min1 and channel2[i,j] >= int_min2:
            point = geom.Point(channel1[i,j], channel2[i,j])
            points = line.interpolate(line.project(point))
            point_on_line[i,j] = points.x
            point_on_line2[i,j] = points.y
            height_img[i,j] = points.z
#height_img[height_img>65] = 65
#height_img[height_img< 15]= 15           
x1 = channel1.flatten()
x2 = point_on_line.flatten()
y1 = channel2.flatten()
y2 = point_on_line2.flatten()
fig, ax = plt.subplots(2,1, figsize=(16, 16))
import matplotlib.patches as patches
try:
    while True:
        im = ax[0].imshow(height_img, cmap = 'inferno')
        xy = plt.ginput(2)
        xlength = xy[1][0] -xy[0][0]
        ylength =xy[1][1]- xy[0][1]
        rect = patches.Rectangle(xy[0], xlength, ylength, fill = False)
        print rect
        ax[0].add_patch(rect)
        section1 = channel1[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
        section2 = channel2[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
        ax[1].clear()
        ax[1].plot(section1,section2, ls = '', marker = ',')
        plt.plot(I1, I2)
        ax[1].set_aspect(1)
        ax[1].axis([np.min(channel1), np.max(channel1), np.min(channel2), np.max(channel2)])
        add_colorbar(im)
        
            
            
except KeyboardInterrupt:
    pass
#im = ax[0].imshow(height_img, cmap = 'inferno')
#xy = plt.ginput(2)
#xlength = xy[1][0] -xy[0][0]
#ylength =xy[1][1]- xy[0][1]
#rect = patches.Rectangle(xy[0], xlength, ylength, fill = False)
#print rect
#ax[0].add_patch(rect)
#section1 = channel1[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
#section2 = channel2[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
#ax[1].plot(section1,section2, ls = '', marker = ',')
##ax[1].plot(I1, I2)
##ax[1].plot([x1, x2], [y1, y2],c = 'r' , alpha = 0.5)
##ax[1].scatter(x1, y1, marker='+', c = 'b')
##ax[1].scatter(x2, y2, marker='+', c = 'g')
##ax[1].axis([np.min((int_min2, int_min1)),np.max((int_max1, int_max2)), np.min((int_min2, int_min1)),np.max((int_max1, int_max2))])
#ax[1].set_aspect(1)
#add_colorbar(im)

# Save ac
img = Image.fromarray(height_img)   # Creates a PIL-Image object
save_name = filename.replace('.tif', '') + '_height.tif'
img.save(directory + save_name)

#fig, ax = plt.subplots(3,1, figsize=(16, 16))
#ax[0].imshow(channel1, cmap = 'gray')
#ax[0].axis( 'off')
#ax[1].imshow(channel2, cmap = 'gray')
#ax[1].axis( 'off')
#im = ax[2].imshow(height_img, cmap = 'inferno')
#ax[2].axis( 'off')
#add_colorbar(im)
#plt.savefig('irm_image.pdf', format ='pdf', dpi =1000)
"""
# For Loop-Plot with Scipy
height_img = np.zeros(channel1.shape)
for i in range(channel1.shape[0]):
    print i
    for j in range(channel1.shape[1]):
        height_img[i,j] = fminbound(res, 0,400, args = ( channel1[i,j],channel2[i,j],S1,D1,S2, D2, w1, w2, n1) )
                  #method =  'bounded', bounds = [0,800]).x

h = np.linspace(0,300,300)
I1, I2 = dw(h, D1, S1,S2, D2, w1, w2, n1)
i1, i2 = dw(height_img, D1, S1,S2, D2, w1, w2, n1)
fig, ax = plt.subplots(4,1, figsize=(16, 16))
ax[0].imshow(channel1)
ax[0].set_title('488 nm')
ax[1].imshow(channel2)
ax[1].set_title('559 nm')
ax[2].imshow(height_img)
ax[3].plot(channel1,channel2, ls = '', marker = ',')
ax[3].plot(I1, I2)
ax[3].plot(i1, i2, ls = '', marker='+', c = 'r')
ax[3].axis([np.min((int_min2, int_min1)),np.max((int_max1, int_max2)), np.min((int_min2, int_min1)),np.max((int_max1, int_max2))])
ax[3].set_aspect(1)






import matplotlib.patches as patches
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(channel1)
ax[0,0].set_title('488 nm')
ax[0,1].imshow(channel2)
ax[0,1].set_title('559 nm')
#xy = plt.ginput(2)
#xlength = xy[1][0] -xy[0][0]
#ylength =xy[1][1]- xy[0][1]
#rect = patches.Rectangle(xy[0], xlength, ylength)
#ax[0].add_patch(rect)
#ax[0].add_patch(rect)


# Average of heights
height_img = np.zeros(channel1.shape)
diff = np.zeros(channel1.shape)
for i in range(channel1.shape[0]):
    print "Row: " + str(i)
    for j in range(channel1.shape[1]):
        if channel1[i,j] >= int_min1 and channel2[i,j] >= int_min2:
            height_img[i,j], diff[i,j] = minimize_height(w1, w2, channel1[i,j], channel2[i,j], S1, D1, S2, D2, n1)
ax[1,0].imshow(height_img)
ax[1,1].imshow(diff)            
# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(I1,I2)
ax.plot(channel1, channel2, ls = '', marker = ',')#, height_img)     

section1 = channel1[int(xy[1][1]): int(xy[0][1]), int(xy[0][0]): int(xy[1][0])]
section2 = channel2[int(xy[1][1]): int(xy[0][1]), int(xy[0][0]): int(xy[1][0])]


int_min1 = np.min(section1)
int_max1 = np.max(section1)
D1 = int_max1 - int_min1
S1 = int_max1+ int_min1

int_min2 = np.min(section2)
int_max2 = np.max(section2)
D2 = int_max2 - int_min2
S2 = int_max2 + int_min2

height_img = height(channel1, D1, S1, w1, n1)
height_img2 = height(channel2, D2, S2, w2, n2)


im = ax[1,0].imshow(height_img)
im2 = ax[1,1].imshow(height_img2)
ax[1,0].set_title('h1 [nm]')
ax[1,1].set_title('h2 [nm]')
add_colorbar(im)
add_colorbar(im2)

     
average_channel = plt.imread("average_channel.tif")
# Scaling
max_int = np.max(original)
scalefac = 255.0/ max_int
scaled= scalefac * original
thresh = scaled.astype(np.uint8)
# Default Minimum 

# Minimize with Lmfit
par = get_h_default(n1, S1, D1, w1, S2, D2, w2)
height_img = np.zeros(section1.shape)
# For Loop with lmfit
for i in range(section1.shape[0]):
    print "Row: " + str(i)
    for j in range(section1.shape[1]):
        fit_result = lmfit.minimize(residual, par, 
                                    kws = {'I1': section1[i, j], 'I2': section2[i, j]})
        height_img[i,j] = fit_result.params["h"].value
plt.figure()
plt.imshow(height_img)
fit_result = lmfit.minimize(residual, par, kws = {'I1': channel1[85, 191:192], 'I2': channel2[85, 191:192]})
h_img = fit_result.params["h"].value
print h_img







# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot(I1,I2, img_h)
ax = fig.add_subplot(212, projection='3d')
ax.plot(I1,I2, h)



ch1_height = height(channel1, D1, S1, w1, n1)
ch2_height = height(channel2, D2, S2, w2, n1)
ch_diff = ch1_height - ch2_height

fig, ax = plt.subplots(3,1)
ax[0].imshow(ch1_height)
ax[1].imshow(ch2_height)
ax[2].imshow(ch_diff)
#Find channel
kernel = np.ones((4,4),np.uint8)
__, thresh = cv2.threshold(thresh ,50, 1,cv2.THRESH_BINARY)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 5)
closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 4 )

laplacian = abs(cv2.Laplacian(original,cv2.CV_64F, ksize =5))
max_lap = np.max(laplacian)
laplacian = np.uint8(255.0/ max_lap * laplacian)
__, cache = cv2.threshold(laplacian ,33, 1,cv2.THRESH_BINARY)
closing2 = cv2.morphologyEx(cache, cv2.MORPH_CLOSE,kernel, iterations = 3)
opening2 = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel, iterations = 1)
closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE,kernel, iterations = 4)

mask_ind = closing == 0

masked_img = np.copy(original)
masked_img[mask_ind]= 6000
min_int = np.min(masked_img)
print min_int
print 
thresh_ind = masked_img > min_int
thresh = np.copy(masked_img)
thresh[thresh_ind]= 0 

plt.figure()
plt.subplot(311)
plt.imshow(average_channel, cmap = "gray", alpha = 1)
plt.subplot(312)
plt.imshow(thresh)
#plt.imshow(masked_img, cmap = "gray", alpha = 0.4)
plt.subplot(313)
plt.imshow(original, cmap = "gray")

max_int = np.max(thresh) 
min_int = np.min(thresh)
__, max_Image = cv2.threshold(thresh ,max_int-1, 255,cv2.THRESH_BINARY)
__, min_Image = cv2.threshold(thresh, min_int, 255, cv2.THRESH_BINARY_INV)




inc_int = IncidentIntensity(intensity_mean, n0, n1)
p_img = original/ inc_int


def IRM(d, n0, n1, n2, w, p):
    diff = ((n1**2 * (n2-n0)**2 *np.cos(np.pi*2/w* n1 *d)**2 + (n1**2 -n0*n2)**2 
          *np.sin(np.pi/w*2*n1*d)**2)/(n1**2 * (n2+n0)**2 
          *np.cos(np.pi*2/w* n1 *d)**2 + (n1**2 +n0*n2)**2 
          *np.sin(np.pi/w*2*n1*d)**2) - p)
    return diff
    
def findIntersection(p, n0, n1, n2, w, x0 =0):

    return fsolve(IRM,x0, args = (n0, n1, n2, w, p))
"""