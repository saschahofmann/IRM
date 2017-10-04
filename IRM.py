# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.optimize import curve_fit
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



# Refractive Index and wave lengths
n0 = 1.525    # https://us.vwr.com/assetsvc/asset/en_US/id/17619754/contents
n1 = 1.34
n2 = 1.486
n3 = 1.36     # http://onlinelibrary.wiley.com/store/10.1002/cyto.a.20134/asset/20134_ftp.pdf?v=1&t=j809eogl&s=adcee703002c2b2030641711cd89e2b5fe800d94&systemMessage=Wiley+Online+Library+will+be+unavailable+on+Saturday+7th+Oct+from+03.00+EDT+%2F+08%3A00+BST+%2F+12%3A30+IST+%2F+15.00+SGT+to+08.00+EDT+%2F+13.00+BST+%2F+17%3A30+IST+%2F+20.00+SGT+and+Sunday+8th+Oct+from+03.00+EDT+%2F+08%3A00+BST+%2F+12%3A30+IST+%2F+15.00+SGT+to+06.00+EDT+%2F+11.00+BST+%2F+15%3A30+IST+%2F+18.00+SGT+for+essential+maintenance.+Apologies+for+the+inconvenience+caused+.
w1 = 488
w2 = 635
dm = 4 
ina = 1.2
method = 'INA_MP'



I11 = 1485.554
I12 = 1497.267
# Image In-read
directory = 'Height Measurements/'
filename = "1209_6x8_WT_006b.tif"
original = io.imread(directory + filename)
channel1 = original[0] 
channel2 = original[1] 
if method =='MP':
    import mp
    height_img = mp.MP(channel1, channel2, I11, I12, n0, n1, n2, n3 ,w1, w2, dm)
    
elif method == 'normal':
    import minmax
    height_img = minmax.minmaxmethod(channel1, channel2, I11, I12, n0, n1, n3, w1, w2)
    
elif method == 'INA':
    import INA
    height_img = INA.INA(channel1, channel2, ina, I11, I12, n0, n1, n3 ,w1, w2)
elif method =='INA_MP':
    import mp_ina
    height_img, I1, I2, channel1, channel2 = mp_ina.inamp_main(channel1, channel2, I11, I12, n0, n1, n2, n3 ,w1, w2, dm, ina)
    
    
# Save as
img = Image.fromarray(height_img)   # Creates a PIL-Image object
save_name = filename.replace('.tif', '') + '_height.tif'
img.save(directory + save_name)


fig, ax = plt.subplots(2,1, figsize=(16, 16))
ax[0].imshow(height_img, cmap = 'inferno')
import matplotlib.patches as patches
try:
    while True:
        xy = plt.ginput(2)
        xlength = xy[1][0] -xy[0][0]
        ylength =xy[1][1]- xy[0][1]
        rect = patches.Rectangle(xy[0], xlength, ylength, fill = False)
        ax[0].add_patch(rect)
        section1 = channel1[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
        section2 = channel2[int(xy[0][1]): int(xy[1][1]), int(xy[0][0]): int(xy[1][0])]
        ax[1].clear()
        ax[1].plot(section1,section2, ls = '', marker = ',')
        ax[1].plot(I1, I2)
        ax[1].set_aspect(1)
        ax[1].axis([np.min(channel1), np.max(channel1), np.min(channel2), np.max(channel2)])
        
except KeyboardInterrupt:
    pass


"""
#Density plot
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
#fig, ax = plt.subplots(3,1, figsize=(16, 16))
#ax[0].imshow(channel1, cmap = 'gray')
#ax[0].axis( 'off')
#ax[1].imshow(channel2, cmap = 'gray')
#ax[1].axis( 'off')
#im = ax[2].imshow(height_img, cmap = 'inferno')
#ax[2].axis( 'off')
#add_colorbar(im)
#plt.savefig('irm_image.pdf', format ='pdf', dpi =1000)


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