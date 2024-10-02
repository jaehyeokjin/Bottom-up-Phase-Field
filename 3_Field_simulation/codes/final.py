import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

import os
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.interpolate import InterpolatedUnivariateSpline as uvs
import sys

import math
snapshot_dir = 'snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

prop = FontProperties(size=14)

nx, ny = 130, 60
# Centered region thickness
thickness = 65

# Adjust the boundary for tanh decay computation
shift = 5  # Shift inward by 5 units on both sides
dx, dy = 0.5e-6, 0.5e-6 # spacing of computational grid [m]
eee = -1.0*7.699448E+04 # driving force of growth of phase B: g_A - g_B [J/m3]. Double checked that it should be negative
www = 13.37131577*abs(eee)
delta = 4.*dx # interfacial thickness [m]
amobi = 1.e-6 # interfacial mobilitiy [m4/(Js)]
ram = 0.1 # paraneter which deternines the interfacial area or 0.1
bbb = 2.*np.log((1.+(1.-2.*ram))/(1.-(1.-2.*ram)))/2.  # The constant b = 2.1972
sigma = delta*www/(6*bbb)
aaa   = np.sqrt(3.*delta*sigma/bbb) # gradient energy coefficient  "a"[(J/m)^(1/2)]
pmobi = amobi*math.sqrt(2.*www)/(6.*aaa) # mobility of phase-field [m3/(Js)]

#fname = sys.argv[1]
#ffile = open(fname,'r')
#xval=[]
#fval=[]
#gval=[]
#for line in ffile:
#    line_e = line.split()
#    xval.append(float(line_e[0]))
#    gval.append(float(line_e[1]))
#    fval.append(float(line_e[2]))
#ffile.close()
#f_dot = uvs(xval,fval,k=5)
#g_dot = uvs(xval,gval,k=5)
def f_dot(x):
    #value = -321.708268143431*x**8 + 1057.37527407473*x**7 - 1373.10667255304*x**6 + 914.430802025268*x**5 - 365.859832608942*x**4 + 110.215990261692*x**3 - 24.4893390250982*x**2 + 4.3313685540494*x - 0.0581860364595567
    #value = 95.4120271845815*x**6 - 270.684409709018*x**5 + 274.771273139203*x**4 - 122.208234316267*x**3 + 24.2447479147958*x**2
    value = x**2*(x - 1)**2*(406.096365931681*x**2 - 312.054265569108*x + 69.9995996612168)
    #value = 30.0 * (1-x)**2 * x**2
    return value

def g_dot(x):
    #value_2 = -110.5792*x**7 + 318.0156*x**6 - 353.9784*x**5 + 196.4455*x**4 - 62.8776*x**3 + 14.2065*x**2 - 2.1044*x + 0.1861
    #value_2 = 24.4682800543111*x**5 - 57.8471965094707*x**4 + 46.9764707898103*x**3 - 15.6700648253102*x**2 + 2.07251049065949*x
    value_2 = x*(92.035982494651*x**4 - 212.328865871469*x**3 + 166.230498138655*x**2 - 51.2257536286395*x + 5.28813886680301)
    #value_2 = 2.0*x*(1.0-x)*(1.0-2.0*x)
    return value_2

dt = dx*dx/(5.*pmobi*aaa*aaa)/2 # time increment for a time step [s]
nsteps = 2001 # total number of time step
#p  = np.full((nsteps,nx,ny),0.25) # phase-field variable
p  = np.zeros((nsteps,nx,ny)) # phase-field variable

# Center the initialization in the x-direction
x_center_start = (nx - thickness) // 2
x_center_end = x_center_start + thickness

# Adjusted boundaries
left_boundary = x_center_start + shift
right_boundary = x_center_end - shift
for i in range(nx):
    for j in range(ny):
        # Calculate the radial distance for decay effect around the central band
        if left_boundary <= i <= right_boundary:
            r = -1.0*shift*dx  # Inside the shifted central band, no decay
        else:
            # Calculate distance from the nearest edge of the shifted central band
            if i <left_boundary:
                r = (x_center_start-i)*dx
            else:
                r = (i-x_center_end)*dx
#            r = min(abs(i - left_boundary), abs(i - right_boundary))*dx
        # Initialize p with a tanh profile that decays outside the shifted central band
        p[0, i, j] = 0.5 * (1. - np.tanh(np.sqrt(2. * www) / (2. * aaa) * r))
def do_timestep(p):
    for t in range(nsteps-1):
        for j in range(ny):
            for i in range(nx):
                ip = i + 1
                im = i - 1
                jp = j + 1
                jm = j - 1
                if ip > nx - 1:
                    ip = nx -1
                if im < 0:
                    im = 0
                if jp > ny - 1:
                    jp = ny -1
                if jm < 0:
                    jm = 0
                p[t+1,i,j] = p[t,i,j] + pmobi * ( -1.0*f_dot(p[t,i,j])*eee-g_dot(p[t,i,j])*www+  aaa*aaa*((p[t,ip,j] - 2*p[t,i,j] + p[t,im,j])/dx/dx + (p[t,i,jp] - 2*p[t,i,j] + p[t,i,jm])/dy/dy) ) * dt
do_timestep(p)

#x = np.linspace(0, nx, nx)
#y = np.linspace(0, ny, ny)
#x, y = np.meshgrid(y, x)
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(y, x) 
fig = plt.figure(figsize=(12,8), dpi=100)
#fig.set_dpi(100)
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    if i % 100 == 0:
        ax.clear()
        ax.set_ylim([0, nx])
        ax.set_xlim([0, ny])
        ax.set_title('Example Plot', fontproperties=prop)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(prop)
        x_ticks = [0, 10, 20, 30, 40, 50, 60]
        y_ticks = [0, 20, 40, 60, 80, 100, 120]
        
        # Apply half scale factor and convert to integer
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(0.5 * val)}" for val in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(0.5 * val)}" for val in y_ticks])

        #ax.set_xlabel('X Axis', fontproperties=prop)
        #ax.set_ylabel('Y Axis', fontproperties=prop)
        #ax.set_zlabel('Phi', fontproperties=prop)
        ax.plot_surface(X, Y, p[i, :, :], rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmax=1, vmin=0)
        ax.set_zlim(0, 1)
        ax.view_init(18,-54,0)
        ax.set_box_aspect([ny/nx, 1, 0.5])  
        # Save each frame to file
        plt.savefig(f'{snapshot_dir}/snapshot_step_{i:04d}.pdf', format='pdf')

anim = animation.FuncAnimation(fig, animate, frames=nsteps-1, interval=10, repeat=False)


plt.show()
