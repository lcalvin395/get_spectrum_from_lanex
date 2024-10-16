import matplotlib.pyplot as plt
import numpy as np

from scipy import signal, ndimage

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def gaus(x,a,x0,sigma,bkg):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
from fnmatch import fnmatch

import os
import glob

def find_tiff_files(folder_path):
    # Ensure the folder path ends with a separator
    if not folder_path.endswith(os.path.sep):
        folder_path += os.path.sep

    # Use glob to find all .tiff files in the folder
    tiff_files = glob.glob(folder_path + '*.tif')

    return tiff_files

folder_path = '/Users/lukecalvin/2023/ELI-NP DATA/espec/20231128/run_03_(5_shots_after_2cm_shots)/untitled folder/'
tiff_files = find_tiff_files(folder_path)

print(len(tiff_files))

shot_numbers = np.array(range(len(tiff_files)))
print(shot_numbers)

fig, ax = plt.subplots(1,1,figsize=(5,3),dpi=150)
for n,i in enumerate(tiff_files):
    img = plt.imread(str(i))
    img_filt = ndimage.median_filter(img,4 )
    img_filt = img_filt[150:400,0:590] # Crop the image to remove the damage spot or xray region
    
    ProjY = np.sum(img_filt,1) #pixel counts projection onto the y-axis (non-dispersion axis)
    y_axis_pxl = np.array(range(len(ProjY))) # y axis array in pixel
    
    extra_ProjY = ProjY[y_axis_pxl>0] # By eye we know that the main signal in this case is towards the right of 200.So just selecting this region.
    extra_y_axis_pxl = y_axis_pxl[y_axis_pxl>0] 
    mean = extra_y_axis_pxl[np.argmax(extra_ProjY)] # The position of the peak (after we truncate the axis(>200) to avoid the peaks at the left). This will be used as input paramter to fit the gaussian.
    
    popt,pcov = curve_fit(gaus,extra_y_axis_pxl,extra_ProjY,p0=[0.9e7,mean,10,100], maxfev=5000)
    
    Amp, Max_pos,sigma,bkg = popt
    waist = sigma*2
    FWHM = waist*np.sqrt(2*np.log(2))
    
    waist_factor = 2 # a factor to multiply waist when selecting the background region
    bkg_y_bottom_select = [int(Max_pos-waist_factor*waist-10), int(Max_pos-waist_factor*waist)]
    bkg_bottom = img_filt[bkg_y_bottom_select[0]:bkg_y_bottom_select[1],:] 

    bkg_y_top_select = [int(Max_pos+waist_factor*waist), int(Max_pos+waist_factor*waist+10)]
    bkg_top = img_filt[bkg_y_top_select[0]:bkg_y_top_select[1],:]
    # print(np.shape(bkg_top))
    # print(np.shape(bkg_bottom))
    # print(bkg_y_bottom_select[0],bkg_y_bottom_select[1])
    bkg = (bkg_bottom + bkg_top)/2
    bkg = np.mean(bkg,axis=0)
    
    
    ROI_y = [int(Max_pos-waist_factor*waist),int(Max_pos+waist_factor*waist)] 

    img_ROI = img_filt[ROI_y[0]:ROI_y[1],:] #selecting the region of interest. beyind this the bkg was selected

    img_ROI = img_ROI-bkg # subtracting background
    img_ROI[img_ROI<0] = 0
    ProjX = np.sum(img_ROI,0) # projection onto the x-axis (dispersion axis)
    ProjY = np.sum(img_ROI,1) # projection onto the y-axis (non-dispersion axis)
    x_axis_pxl = np.array(range(len(ProjX)))
    y_axis_pxl = np.array(range(len(ProjY)))
    
    if fnmatch(os.path.basename(str(i)), '*8*.tiff'):
        ax.plot(x_axis_pxl, ProjX*100,label=os.path.basename(str(i)))
        print(os.path.basename(str(i)))
    else:
        ax.plot(x_axis_pxl, ProjX,label=os.path.basename(str(i)))
        
    
    

    
    
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()