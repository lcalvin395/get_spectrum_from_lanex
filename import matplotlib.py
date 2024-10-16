import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from scipy import signal, ndimage

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

import cv2
import matplotlib.pyplot as plt

def gaus(x,a,x0,sigma,bkg):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg


# Load the image
img = cv2.imread('/Users/lukecalvin/2023/ELI-NP DATA/espec/20231128/espec refererences 20231121/reference_no_filters_30ms.tif') 

# Create a copy of the image
img_copy = np.copy(img)

# # Convert to RGB so as to display via matplotlib
# # Using Matplotlib we can easily find the coordinates
# # of the 4 points that is essential for finding the 
# # transformation matrix
# img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

plt.imshow(img_copy)




#All points are in format [cols, rows]
pt_A = [653, 483]
pt_B = [1470, 1614]
pt_C = [1754, 1311]
pt_D = [892, 131]



# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))
 

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])


# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)
out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_NEAREST)
#img = out
#plt.imshow(out)
img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

lanex_size_x_mm = 290

fig, ax = plt.subplots()

pc = ax.pcolor(img)
ax.set_title('Raw image')
ax.set_xlabel('Dispersion axis (horizontal) [pxls]')
ax.set_ylabel('Non-dispersion axis (vertical) [pxls]')
#plt.colorbar(pc)
plt.show()