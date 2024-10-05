import copy
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as uvs


# Initial variables
filename="../../../trj/aa.trj"
#filename="./aa.trj"
f = open(filename, 'r')
out_data = open("final_energy.hist", 'w')
k = 0
count_time = 0
index = [i for i in range(0, 2048)]
index_temp = []
coord = []
force = []
peratom = []
peratom1 = []
peratom2 = []
perprob=[]
pernorm=[]
pernormprob=[]
id_type = []
time_element = 0
# READ TABLE
rho=[]
#f_fl=[]
f_fs=[]
f_f0=[]
f_fw=[]
f_fepsilon=[]
#fl=open("../table/fl.table",'r')
#for line in fl:
#	rho.append(float(line.split()[1]))
#	f_fl.append(float(line.split()[2]))
fs=open("../fs.dat",'r')
for line in fs:
	rho.append(float(line.split()[0]))
	f_fs.append(float(line.split()[1]))
f0=open("../f0.dat",'r')
for line in f0:
	f_f0.append(float(line.split()[1]))
fw=open("../fw.dat",'r')
for line in fw:
	f_fw.append(float(line.split()[1]))
fepsilon=open("../fepsilon.dat",'r')
for line in fepsilon:
	f_fepsilon.append(float(line.split()[1]))
#spl_fl = uvs(rho,f_fl,k=5)
spl_f0 = uvs(rho,f_f0,k=5)
spl_fs = uvs(rho,f_fs,k=5)
spl_fw = uvs(rho,f_fw,k=5)
spl_fepsilon = uvs(rho,f_fepsilon,k=5)
#result
fs_avg=[]
f0_avg=[]
#fl_avg=[]
fw_avg=[]
fepsilon_avg=[]
# PBC consideration
xlo = xhi = ylo = yhi = zlo = zhi = 0.0
box_size_x = box_size_y = box_size_z = 0.0
axis_read = 0
axis_temp = 0
# Wrapping function

def scale_position(pos, lo, hi):
    scaled_pos = pos * (hi-lo)
    return(scaled_pos)

def distance(a, b):
    dist = ((a[0]-b[0])**2.0 + (a[1]-b[1])**2.0 + (a[2]-b[2])**2.0)**0.5
    return(dist)
###########################################################
############### Trajectory Processing ...  ################
###########################################################

for line in f:
    line_element = line.split()
    if(line_element[0] == 'ITEM:'):
        k = (k+1) % 4
    if(k == 1 and len(line_element) == 1):
        time_element = int(line_element[0])
        index_temp = []
        coord = []
        force = []
        id_type = []
        #out_data.write(line)
    if(k == 2 and len(line_element) == 1):
        n_mole = "2048\n"
        #out_data.write(n_mole)
    if(k == 3 and len(line_element) == 2):
        if (axis_read == 0):
            xlo = float(line_element[0])
            xhi = float(line_element[1])
            box_size_x = xhi-xlo
            axis_temp = 1
            pbc_str = '%2.5f %2.5f\n' % (0.0, box_size_x)
        if (axis_read == 1):
            ylo = float(line_element[0])
            yhi = float(line_element[1])
            box_size_y = yhi-ylo
            axis_temp = 2
            pbc_str = '%2.5f %2.5f\n' % (0.0, box_size_y)
        if (axis_read == 2):
            zlo = float(line_element[0])
            zhi = float(line_element[1])
            box_size_z = zhi-zlo
            axis_temp = 0
            pbc_str = '%2.5f %2.5f\n' % (0.0, box_size_z)
        axis_read = axis_temp
        #out_data.write(pbc_str)
    if(k == 0 and line_element[0] != 'ITEM:'):  # Scan for all
        box_lengtha = " %2.5f %2.5f %2.5f\n" % (box_size_x, box_size_y, box_size_z)
        coord.append([float(line_element[2])-xlo, float(line_element[3])-ylo, float(line_element[4])-zlo])
        id_type.append(int(line_element[1]))
        force.append([float(line_element[5]), float(line_element[6]), float(line_element[7])])
        index_temp.append(int(line_element[0]))
        if ((time_element%1000 == 0)and (len(index_temp) == 2048)):  # Start the mapping process
            print_arg = "Time step: %d is starting! with %f %f %f \n" % (time_element, xlo, ylo, zlo)
            perprob_val = 0.0
            density = [0.0 for _ in range(0,2048)]
            phi = [0.0 for _ in range(0,2048)]
            density_grad = [0.0 for _ in range(0,2048)]
            for i in range(0, 2048):
                perprob_count = 0.0
                grad_x = 0.0
                grad_y = 0.0
                grad_z = 0.0
                for j in range(0, 2048):
                    if i!=j :
                        delta = [coord[i][0]-coord[j][0],coord[i][1]-coord[j][1],coord[i][2]-coord[j][2]]
                        while(delta[0] > box_size_x/2.0 or delta[0] < -0.5 * box_size_x):
                            if delta[0] > box_size_x/2.0:
                                delta[0] -= box_size_x
                            elif delta[0] < -0.5 * box_size_x:
                                delta[0] += box_size_x
                        while(delta[1] > box_size_y/2.0 or delta[1] < -0.5 * box_size_y):
                            if delta[1] > box_size_y/2.0:
                                delta[1] -= box_size_y
                            elif delta[1] < -0.5 * box_size_y:
                                delta[1] += box_size_y
                        while(delta[2] > box_size_z/2.0 or delta[2] < -0.5 * box_size_z):
                            if delta[2] > box_size_z/2.0:
                                delta[2] -= box_size_z
                            elif delta[2] < -0.5 * box_size_z:
                                delta[2] += box_size_z
                        distval = delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]
                        vval1 = math.exp(-distval/800)
                        grad_x += vval1 * -2.0/800*delta[0]
                        grad_y += vval1 * -2.0/800*delta[1]
                        grad_z += vval1 * -2.0/800*delta[2]
                        perprob_count += vval1
                density[i] = perprob_count
                phi[i] = 0.5*(1-math.tanh((perprob_count-1132)/2))
                density_grad[i] = (grad_x*grad_x+grad_y*grad_y+grad_z*grad_z)*(0.25*(1.0-(math.tanh((perprob_count-1132)/2)*math.tanh((perprob_count-1132)/2))))*(0.25*(1.0-(math.tanh((perprob_count-1132)/2)*math.tanh((perprob_count-1132)/2))))
                # perprob_val += 0.5*(1-math.tanh((perprob_count-1132)/2))
            #perprob.append(perprob_val/2048.0)
            # Clean up the variables
            fs_val = 0.0
            f0_val = 0.0
            fw_val = 0.0
            fepsilon_val = 0.0
            fs_cnt = 0.0
            f0_cnt = 0.0
            fw_cnt = 0.0
            fepsilon_cnt = 0.0
            for i in range(0,2048):
                fs_cnt += phi[i]*phi[i]*phi[i]*(10-15*phi[i]+6*phi[i]*phi[i])
                f0_cnt += (1.0-phi[i]*phi[i]*phi[i]*(10-15*phi[i]+6*phi[i]*phi[i]))
                fs_val += phi[i]*phi[i]*phi[i]*(10-15*phi[i]+6*phi[i]*phi[i])*spl_fs(phi[i])
                f0_val += (1.0-phi[i]*phi[i]*phi[i]*(10-15*phi[i]+6*phi[i]*phi[i]))*spl_f0(phi[i])
                fw_cnt += phi[i]*phi[i]*(1-phi[i])*(1-phi[i])
                fw_val += phi[i]*phi[i]*(1-phi[i])*(1-phi[i])*spl_fw(phi[i])
                fepsilon_cnt += density_grad[i]
                fepsilon_val += density_grad[i]*spl_fepsilon(phi[i])
            fs_avg.append([fs_cnt,fs_val])
            f0_avg.append([f0_cnt,f0_val])
            fw_avg.append([fw_cnt,fw_val])
            fepsilon_avg.append([fepsilon_cnt,fepsilon_val])
            coord = []
            index_temp = []
            print_arg = "Time step: %d is done! \n" % time_element
            print(print_arg)

str_out = "index fs fw fepsilon\n"
out_data.write(str_out)    
for i in range(0,len(fs_avg)):
    fs_val = fs_avg[i][1]/fs_avg[i][0]
    f0_val = f0_avg[i][1]/f0_avg[i][0]
    fw_val = fw_avg[i][1]/fw_avg[i][0]
    fepsilon_val = fepsilon_avg[i][1]/fepsilon_avg[i][0]
    #str_out = "%d %.6e %.6e %.6e %.6e\n" % (i+1,fs_val,fl_val,fw_val,fepsilon_val)
    str_out = "%d %.6e %.6e %.6e %.6e\n" % (i+1,fs_val,f0_val, fw_val,fepsilon_val)
    out_data.write(str_out)
fin_fs_val = np.sum(np.array(fs_avg), axis=0)[1]/np.sum(np.array(fs_avg), axis=0)[0]
fin_f0_val = np.sum(np.array(f0_avg), axis=0)[1]/np.sum(np.array(f0_avg), axis=0)[0]
#fin_fl_val = np.sum(np.array(fl_avg), axis=0)[1]/np.sum(np.array(fl_avg), axis=0)[0]
fin_fw_val = np.sum(np.array(fw_avg), axis=0)[1]/np.sum(np.array(fw_avg), axis=0)[0]
fin_fepsilon_val = np.sum(np.array(fepsilon_avg), axis=0)[1]/np.sum(np.array(fepsilon_avg), axis=0)[0]
#str_out = "%.6e %.6e %.6e %.6e\n" %(fin_fs_val,fin_fl_val,fin_fw_val,fin_fepsilon_val)
str_out = "%.6e %.6e %.6e %.6e\n" %(fin_fs_val,fin_f0_val, fin_fw_val,fin_fepsilon_val)
print(str_out)
f.close()
fs.close()
f0.close()
#fl.close()
fw.close()
fepsilon.close()
out_data.close()
