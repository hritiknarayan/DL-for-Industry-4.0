import csv
import tensorflow as tf
import tensorflow.keras as keras
import array
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.image import NonUniformImage


np.random.seed(100)


x = []
y = []

I = range(0,59)
J = range(0,60)

for i in I:

    if i <=17:
        a = 0.5*(1-(1.1**(17-i)))
        x.append(a)
    elif i>17 and i<=37:
        a = 0.05*(i-17)
        x.append(a)
    else:
        a=1.0 -(0.5*(1-(1.1**(i-37))))
        x.append(a)

for j in J:
    if j<=26:
        b=-0.15+0.5*(1-(1.1**(26-j)))
        y.append(b)
    elif j>26 and j<=33:
        if j<=29:
            b=0.05*(j-29)
            y.append(b)
        elif j==30:
            y.append(b)
        elif j>=30:
            b=0.05*(j-30)
            y.append(b)
    elif j>33 and j<=60:
        b=0.15-0.5*(1.0-(1.1**(j-33)))
        y.append(b)

h = []
k = []


for i in range(0,59):
    if i>0 and i<58:
        a = (x[i+1] - x[i-1])/2
        h.append(a)
    else:
        h.append(0)


for j in range(0,60):
    if (j==29) :
        b = (y[j+2]-y[j-1])/2
        k.append(b)
    elif j>0 and j<59 and j!=29:
        b = (y[j+1]-y[j-1])/2
        k.append(b)
    else:
        k.append(0)




def denorm(phi_set1, lim_vals1):
    phi_set = trp(phi_set1)
    lim_vals = trp(lim_vals1)

    de_list = []
    for j in range(len(phi_set)):
        curr_list = phi_set[j]
        cur_lim = lim_vals[j]
        min_ = cur_lim[0]
        max_ = cur_lim[1]
        new_list = [min_ + (max_-min_)*(y+0.7)/1.4 for y in curr_list]
        de_list.append(new_list)

    return trp(de_list)


def trp(set_):
    set1 = [[set_[i][j] for i in range(len(set_))] for j in range(len(set_[0]))]
    return set1


def local_norm(phi_set_1):

    phi_set_ = trp(phi_set_1)
    #[[phi_set_1[i][j] for i in range(len(phi_set_1))] for j in range(len(phi_set_1[0]))]
    #phi_set = np.array(phi_set).astype(np.float)
    phi_set = [[float(phi_set_[i][j]) for j in range(len(phi_set_[0]))] for i in range(len(phi_set_))]
    lim_set = []
    norm_phi_set = []
    for b in range(len(phi_set)):
        #each case
        cur_list = phi_set[b][:]
        max_ = max(cur_list)
        min_ = min(cur_list)
        lim_set.append([min_, max_])
        new_list = [ -0.7 + 1.4*(x-min_)/(max_-min_) for x in cur_list]
        norm_phi_set.append(new_list)

    #local limits set, normalized set values
    return np.array(trp(norm_phi_set)), trp(lim_set)

#def rem_zeros(lis1, lis2):
#    ct

#x_train, y_train = rem_zeros(x_train, y_train)
#x_train, y_train, case_bcs, rand_order = randomize(x_train, y_train, case_bcs)

#x_train, y_train, rand_order = randomize(x_train, y_train)

#x_train_norm, x_lims = global_norm(x_train)
#y_train_norm, y_lims = global_norm(y_train)

#comment1
#x_train_norm_ind, x_ind_lims = local_norm(x_train)

#comment2
#print(np.array(x_train_norm_ind).shape)

#x_train_gg = x_train_norm_ind[np.newaxis]
#print(x_train_gg.shape)




__FLAG__ = 0
if __FLAG__ == 1:
    ct1= 0
    for i in I:
        if ct1 %2 == 0:
            plt.axvline(x = x[i])
            print ("sb!")
        ct1 += 1

    ct2 = 0
    for j in J:
        if ct2%2 == 0:
            plt.axhline(y = y[j])
        ct2 +=1

    plt.show()

def join_fun2(x_train, y_train):

    for x_el in x_train[0:21]:
        y_train = np.insert(y_train, 59*29+17, x_el)
    for x_el in x_train[21:]:
        y_train = np.insert(y_train, 59*29+17+21+17+17, x_el)

    phi2d = []
    i = 0
    row_ = []
    while i< len(y_train):
        if i%59 == 0 and i!= 0:
            phi2d.append(row_)
            row_ = []
            row_.append(y_train[i])
        else:
            row_.append(y_train[i])
        i+=1
    phi2d.append(row_)
    return phi2d


def vel(phi,h,vinf,alp):                  #velocity calculation
    dpdxu = []
    dpdxl = []
    v_u = []
    v_l = []
    for i in range(17,38):
        if i == 17:
            a = (phi[30][i+1] - phi[30][i])/h[i]
            b = (phi[29][i+1] - phi[29][i])/h[i]
            dpdxu.append(a)
            dpdxl.append(b)
        elif i == 37:
            a = (phi[30][i] - phi[30][i-1])/h[i]
            b = (phi[29][i] - phi[29][i-1])/h[i]
            dpdxu.append(a)
            dpdxl.append(b)
        else:
            a = (phi[30][i+1] - phi[30][i-1])/(2*h[i])
            b = (phi[29][i+1] - phi[29][i-1])/(2*h[i])
            dpdxu.append(a)
            dpdxl.append(b)
    for i in range(0,21):
        a = dpdxu[i] + vinf*np.cos(alp)
        b = dpdxl[i] + vinf*np.cos(alp)
        v_u.append(a)
        v_l.append(b)
    return v_u,v_l

#add before bcs in and after plate


def float_fun(phi_):
    phi_float = [[float(k) for k in phi_[i]] for i in range(len(phi_))]
    return phi_float



def vel_profile(phi, bcs, h, k):

    phi = float_fun(phi)

    xvel = [[0 for i in phi[0]] for j in phi]
    yvel = [[0 for i in phi[0]] for j in phi]

    vinf = bcs[2]
    alpha = bcs[1]

    r_len = range(len(phi))
    c_len = range(len(phi[0]))

    p = 0
    for i in r_len[2:-2]:
        for j in c_len[2:-2]:
                if i != 29 or i != 30:
                    dpdy = (float(phi[i+1][j])-float(phi[i-1][j]))/(2*k[i])
                    yvel[i][j] = -dpdy + p*vinf*math.sin(alpha)
                elif i == 29:
                    dpdy = (float(phi[i][j])-float(phi[i-1][j]))/(k[i])
                    yvel[i][j] = -dpdy + p*vinf*math.sin(alpha)
                elif i == 30:
                    dpdy = (float(phi[i+1][j])-float(phi[i][j]))/(k[i])
                    yvel[i][j] = -dpdy + p*vinf*math.sin(alpha)

    for i in r_len[2:-2]:
        for j in c_len[2:-2]:
                dpdx = (float(phi[i][j+1])-float(phi[i][j-1])/(2*h[j]))
                xvel[i][j] = -dpdx + p*vinf*math.cos(alpha)

                if (0): #leading and trailing points
                    if j == 17 and (i == 29 or i == 30):
                        dpdx = (float(phi[i][j+1])-float(phi[i][j])/(h[j]))
                        xvel[i][j] = dpdx + p*vinf*math.cos(alpha)
                    elif j == 37 and (i == 29 or i==30):
                        dpdx = (float(phi[i][j])-float(phi[i][j-1]))/(h[j])
                        xvel[i][j] = dpdy + p*vinf*math.cos(alpha)
                    else :
                        dpdx = (float(phi[i][j+1])-float(phi[i][j-1])/(2*h[j]))
                        xvel[i][j] = dpdx + p*vinf*math.cos(alpha)

    return xvel, yvel

def vel_mag_profile(xvel, yvel):
    #mag_profile = xvel[:]
    mag_profile = [[0 for i in xvel[0]] for j in xvel]
    for i in range(len(xvel)):
        for j in range(len(xvel[0])):
            xv = xvel[i][j]
            yv = yvel[i][j]
            mag_profile[i][j] = math.sqrt(xv**2 + yv**2)
    return mag_profile

def complete_vel(phi, bcs, h, k):
    xvel, yvel= vel_profile(phi, bcs, h, k)
    vel_mag = vel_mag_profile(xvel, yvel)

    return xvel, yvel, vel_mag

def vel_difference(v_actual, v_pred):
    diffvec=[[(v_actual[j][i]-v_pred[j][i]) for i in range(59)] for j in range(60)]
    return diffvec










def minMax(phi_set):
    phi_set = np.array(phi_set).astype(np.float)
    max_ = phi_set[0]
    min_ = phi_set[0]
    for ii in range(len(phi_set)):
        #for jj in range(len(phi_set[0])):
            if max_ < phi_set[ii]:
                max_ = phi_set[ii]
            if min_ > phi_set[ii]:
                min_ = phi_set[ii]
    return [max_, min_]




#x_vel_a, y_vel_a, vel_mag_a = complete_vel(phi_actual_5, bcsm10, h, k)
#x_vel_p, y_vel_p, vel_mag_p = complete_vel(phi_pred_5, bcsm10, h, k)
    

case_id=1



x_vel_a=np.array(np.load("cfd"+str(case_id)+"totx.npy"),dtype='float64').reshape(60,59).tolist()
y_vel_a=np.array(np.load("cfd"+str(case_id)+"toty.npy"),dtype='float64').reshape(60,59).tolist()
vel_mag_a=np.array(np.load("cfd"+str(case_id)+"tot.npy"),dtype='float64').reshape(60,59).tolist()
x_pert_a=np.array(np.load("cfd"+str(case_id)+"pertx.npy"),dtype='float64').reshape(60,59).tolist()
y_pert_a=np.array(np.load("cfd"+str(case_id)+"perty.npy"),dtype='float64').reshape(60,59).tolist()
pert_mag_a=np.array(np.load("cfd"+str(case_id)+"pert.npy"),dtype='float64').reshape(60,59).tolist()
x_vel_p=np.array(np.load("cnn"+str(case_id)+"totx.npy"),dtype='float64').reshape(60,59).tolist()
y_vel_p=np.array(np.load("cnn"+str(case_id)+"toty.npy"),dtype='float64').reshape(60,59).tolist()
vel_mag_p=np.array(np.load("cnn"+str(case_id)+"tot.npy"),dtype='float64').reshape(60,59).tolist()
x_pert_p=np.array(np.load("cnn"+str(case_id)+"pertx.npy"),dtype='float64').reshape(60,59).tolist()
y_pert_p=np.array(np.load("cnn"+str(case_id)+"perty.npy"),dtype='float64').reshape(60,59).tolist()
pert_mag_p=np.array(np.load("cnn"+str(case_id)+"pert.npy"),dtype='float64').reshape(60,59).tolist()


vx_diff = vel_difference(x_vel_a, x_vel_p)
vy_diff = vel_difference(y_vel_a, y_vel_p)
vmag_diff = vel_difference(vel_mag_a, vel_mag_p)
pertx_diff=vel_difference(x_pert_a,x_pert_p)
perty_diff=vel_difference(y_pert_a,y_pert_p)
pertmag_diff=vel_difference(pert_mag_a,pert_mag_p)

plate = [[0 for i in x_vel_a[0]] for j in x_vel_a]
for i in range(len(plate)):
    for j in range(len(plate[i])):
        if i == 29:
            if j>= 17 and j<= 37 :
                plate[i][j] = 1
                #print ("kk")

x = np.array(x)
y = np.array(y)



#added below

__FLAG2__ = 1

def vel_plot(list1, name, vlims,savename):

    fig, axs = plt.subplots(nrows=1, ncols=1,
                            constrained_layout=False)
    xmi = np.amin(x)
    xma = np.amax(x)
    ymi = np.amin(y)
    yma = np.amax(y)
    ax = axs
    im = NonUniformImage(axs, interpolation="bilinear",
                        extent=(xmi, xma, ymi, yma),
                        cmap="hsv")

    im.set_data(x, y, list1)
    
    ax.images.append(im)
    cb = fig.colorbar(im)

    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 1)
    if(savename[-1]=="f"):
        cb.set_clim(-20,20)
        
    
    ax.set_title(name)

    # To plot the plate line on the plot
    y_plate = []
    x_plate = []

    for k in x:
        if k >= 0 and k <=1 :
            x_plate.append(k)
            y_plate.append(0)
        else :
            x_plate.append(k)
            y_plate.append(None)

    plt.plot(x_plate, y_plate, color = 'black')
    plt.savefig(savename+str(case_id)+".png")
    plt.show()

def vel_lims(list1, list2):
    cur_max = max(np.amax(list1), np.amax(list2))
    cur_min = min(np.amin(list1), np.amin(list2))
    return [cur_max, cur_min]


vxl = np.array(vel_lims(x_vel_a, x_vel_p),dtype='float64').tolist()
vyl = np.array(vel_lims(y_vel_a, y_vel_p),dtype='float64').tolist()
vml = np.array(vel_lims(vel_mag_a, vel_mag_p),dtype='float64').tolist()
pertxl = np.array(vel_lims(x_pert_a, x_pert_p),dtype='float64').tolist()
pertyl = np.array(vel_lims(y_pert_a, y_pert_p),dtype='float64').tolist()
pertl = np.array(vel_lims(pert_mag_a, pert_mag_p),dtype='float64').tolist()







__displayPlots__ = 1
if __displayPlots__ == 1:


    vel_plot(x_vel_p, "Vx component predicted", vxl,"vxpred")
    vel_plot(y_vel_p, "Vy component predicted", vyl,"vypred")
    vel_plot(vel_mag_p, "Vel mag component predicted", vml,"vmagpred")

    vel_plot(x_vel_a, "Vx component actual", vxl,"vxactual")
    vel_plot(y_vel_a, "Vy component actual", vyl,"vyactual")
    vel_plot(vel_mag_a, "Vel mag component actual", vml,"vmagactual")

    vel_plot(vx_diff, "Vx component diff", vxl,"vxdiff")
    vel_plot(vy_diff, "Vy component diff", vyl,"vydiff")
    vel_plot(vmag_diff, "Vel mag component diff", vml,"vmagdiff")
    
    vel_plot(x_pert_p, "Perturbation Vx predicted", pertxl,"pertxpred")
    vel_plot(y_pert_p, "Perturbation Vy predicted", pertyl,"pertypred")
    vel_plot(pert_mag_p, "Perturbation mag predicted", pertl,"pertmagpred")
    
    vel_plot(x_pert_a, "Perturbation Vx actual", pertxl,"pertxactual")
    vel_plot(y_pert_a, "Perturbation Vy actual", pertyl,"pertyactual")
    vel_plot(pert_mag_a, "Perturbation mag actual", pertl,"pertmagactual")
    
    vel_plot(pertx_diff, "Perturbation Vx difference", pertxl,"pertxdiff")
    vel_plot(perty_diff, "Perturbation Vy difference", pertyl,"pertydiff")
    vel_plot(pertmag_diff, "Perturbation mag difference", pertl,"pertmagdiff")
    
    




