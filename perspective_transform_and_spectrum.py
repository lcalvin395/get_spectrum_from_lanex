import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from scipy import signal, ndimage

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

import cv2
import skimage as ski
import matplotlib.pyplot as plt

import scipy.ndimage
import math
import matplotlib.ticker as ticker



#MATPLOTLIB INTERACTIVE MODE TURNED OFF (FOR PLOTS)#

plt.ioff()

####################################################


def gaus(x,a,x0,sigma,bkg):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg



### reference image file ###
#file='reference_no_filters_30ms.tif'
############################


path='/Users/lukecalvin/2023/ELI-NP DATA/espec/20231124/run_07/'
file='Espec_#0010_000001.tif'
bckgrnd_file='Espec_#0001_000001.tif'

# Load the background image
bckimg = ski.io.imread('%s%s'%(path,bckgrnd_file)) 
#print(bckimg)
# Load the image
img = ski.io.imread('%s%s'%(path,file)) 

#print(img[423][822]) 
#print(bckimg[423][822]) 
# Create a copy of the image
img_copy = np.copy(img)
bckgrnd_subtrct=input("Perform background subtraction?\ny/n:")
if bckgrnd_subtrct=='y':
    for w in range(len(img)):
        for q in range(len(img[1])):
            if img[w][q]>=bckimg[w][q]:
                img[w][q]=img[w][q]-bckimg[w][q] 
            else:
                img[w][q]=0
else:
    print('No background subtraction.')


#plt.imshow(img)
#plt.show()

# # Convert to RGB so as to display via matplotlib
# # Using Matplotlib we can easily find the coordinates
# # of the 4 points that is essential for finding the 
# # transformation matrix
# img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

#plt.imshow(img_copy)
#plt.savefig('%stest12_resolution_10905767688670'%(path),bbox_inches='tight',)



#All points are in format [cols, rows]
pt_A = [653, 483]
pt_B = [1470, 1614]
pt_C = [1754, 1311]
pt_D = [892, 131]



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
out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
#img = out

plt.imshow(out)
#out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

plt.savefig('%sperspective_fixed_%s'%(path,file),bbox_inches='tight', dpi=2000)
plt.close()
lanex_size_x_mm = 290


#fig, ax = plt.subplots()

#pc = ax.pcolor(img)
#ax.set_title('Raw image')
#ax.set_xlabel('Dispersion axis (horizontal) [pxls]')
#ax.set_ylabel('Non-dispersion axis (vertical) [pxls]')
#plt.colorbar(pc)
#plt.savefig('%sraw_image_%s'%(path,file),bbox_inches='tight', dpi=2000)
#plt.show()



'''x0, y0 = 196, 0 # These are in _pixel_ coordinates!!
x1, y1 = 196, 1460
num = 1000
x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

# Extract the values along the line, using cubic interpolation 
zi = scipy.ndimage.map_coordinates(out, np.vstack((x,y)))

#-- Plot...
fig, axes = plt.subplots(nrows=2)
axes[0].imshow(out)
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(zi)

plt.show()'''



#############################################

###    GOOD WAY FOR PLOTTING PROFILE   ###

"""start = (0, 196) #Start of the profile line row=100, col=0
end = (1460, 196) #End of the profile line row=100, col=last

profile = ski.measure.profile_line(out, start, end)

fig, ax = plt.subplots(1, 2)
ax[0].set_title('Image')
ax[0].imshow(out)
ax[0].plot([start[1], end[1]], [start[0], end[0]], 'r')
ax[1].set_title('Profile')
ax[1].plot(profile)
plt.show()"""

#############################################



##############################
## distance pixel calibration ##

start_cm=0
end_cm=25
start_pixel=0
end_pixel=834

pixel_cm_ratio=(end_cm/end_pixel)/100   #multiply the pixel value by this to change to meters


##############################

##############################
## energy pixel calibration ##


dipole_strt_to_scrn=0.82
length_of_dipole=0.3
thetal=0
me=9.11*(10**-31)
c=3*(10**8)
e=1.6*(10**-19)
B=1
metres=[]

Energy=np.arange(150000,250000000,1175)
for E in range(150000,250000000,1175):
    #print(E)
    joules=((float(E)/1000)*1000000)*(1.6*(10**-19))
    radius=np.sqrt(joules*(joules+2*me*c**2))/(e*c*B)
    xP=0.3
    yP=radius-np.sqrt((radius**2)-xP**2)
    xC=((xP**2)+(yP**2))/(2*xP)
    #print(type(xP))
    metres.append(round((((dipole_strt_to_scrn-xC)*yP)/(xP-xC+yP*math.tan(thetal))),5))
#print(metres)













##############################




###### these points are in [rows,colunms] ######

centre=ndimage.maximum_position(out) #position of max pixel

prof_pt_a=[0,centre[1]-35]
prof_pt_b=[1460,centre[1]-35]
prof_pt_c=[1460,centre[1]+35]
prof_pt_d=[0,centre[1]+35]


prof_thick=np.sqrt(((prof_pt_d[1]-prof_pt_a[1])**2)+((prof_pt_d[0]-prof_pt_a[0])**2))

#profile = out[range((start[2]-prof_thick/2),(start[2]+prof_thick/2)),:]
profile=[]
for l in range(prof_pt_a[1],prof_pt_d[1]):
    profile.append(out[:,l])
#profile = out[:,196]
#profile=img_copy[:,950]

new=np.zeros(len(profile[0]))

#print(profile[0])
#print(new)
for i in range(0,int(prof_thick)):
    new=new+profile[i]
    #print(new)

new_x=np.arange(0,len(new))
new_x=new_x*pixel_cm_ratio

'''new[:] = new[::-1]'''

plot_energy=np.zeros(len(new))

plot_energy=70.823*(new_x**(-0.945)) #energy to distance off beam axis at screen calculated in excel file and fit found

plot_energy = np.round(plot_energy)
'''plot_energy[:]=plot_energy[::-1]'''
'''for m in range(0,len(new_x)):
    if m==1460:
        continue
    plot_energy[m]=int(plot_energy[m])
    print(plot_energy[m])'''

print(new_x[len(new_x)-1])


#print(new_x)
#print(profile) 
#new=new/prof_thick
fig, ax = plt.subplots(1,2)
ax[0].set_title('Image')
ax[0].imshow(out)
ax[0].plot([prof_pt_a[1],prof_pt_b[1]], [prof_pt_a[0],prof_pt_b[0]], 'r')
ax[0].plot([prof_pt_b[1],prof_pt_c[1]], [prof_pt_b[0],prof_pt_c[0]], 'r')
ax[0].plot([prof_pt_c[1],prof_pt_d[1]], [prof_pt_c[0],prof_pt_d[0]], 'r')
ax[0].plot([prof_pt_d[1],prof_pt_a[1]], [prof_pt_d[0],prof_pt_a[0]], 'r')

ax[1].set_title('Profile')
ax[1].plot(new_x,new)
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (72.59335631/(x**0.93984962))))







############ Calculating charge of the spectrum ##################


pi=3.14159265359 #
pgos=7.44 #g/cm^3 scintillator is composed of amixture of phosphor powder (Gd2O2S:Tb)
hs=33*(10**-3) # phosphor surface loading
epsdEbydx=180 # yield of kinetic energy of an electron which is transformed into visible light into the scintillator expressed in unit of pure gadolinium oxysul-fide (GOS) thickness
sigx=hs/pgos*math.cos(pi/4) #s the equivalent thickness of pure GOS crossed by an electron
Eph=2.27*(10**-6) #MeV the energy of one photon emitted at 546 nm.
dNcrbydNel=(1/Eph)*epsdEbydx*sigx # The number of photons Ncr created in the scintillator at the central wavelength per incident electron

ζ=0.22

gthetaCCD=math.cos(pi/4)/pi
print(gthetaCCD)
sigomega=1*(10**-4)
qlens=0.95
qIf=0.05
qIf2=0.96
qIR=0.85
qfibre=0.37

dNcollbydNcr=ζ*gthetaCCD*sigomega*qlens*qIf*qIf2*qfibre*qIR

QE=0.58
r=0.46

dNctsbydNcoll=QE/r

pixelsize=6.5*(10**-3)
print(new[500])
new[:]=new[:]/(pixelsize*(dNctsbydNcoll*dNcollbydNcr*dNcrbydNel))
print(new[500])
##########################################################

fig, ax=plt.subplots()
ax.plot(plot_energy,new)
#plt.xticks(new_x, plot_energy)
#plt.locator_params(axis='x',tight=True, nbins=11)
ax.set_xlim(0,2500)
#ax.set_xticklabels(plot_energy.astype(int))

print('total electrons:',sum(new))
print('total charge:',round((sum(new)*(1.6*(10**-19)*10**9)),3),'nC')



plt.savefig('%sSpectrum_%s'%(path,file),bbox_inches='tight', dpi=1000)
plt.show()

#print(new_x)
#print(plot_energy)




