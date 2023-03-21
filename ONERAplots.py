# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:14:26 2020

@author: 91996
"""

import numpy as np

from numpy import genfromtxt

import matplotlib.pyplot as plt

xcval = genfromtxt('xcvalues.txt', delimiter=',')
#outputs=genfromtxt('CPo.txt', delimiter=',')
xcexpt = genfromtxt('xcexpt.txt', delimiter=',')
cpinp=np.load("cpinputs.npy").reshape(33)

truecp=np.load("cptrue.npy").reshape(368)
truecfx=np.load("cfxtrue.npy").reshape(368)

cnncp=np.load("cnncpnew.npy").reshape(368)
cnncfx=np.load("cnncfxnew.npy").reshape(368)

#EXPERIMENTAL CP VALUES

upper_xcp=xcexpt[11:]
lower_xcp=xcexpt[:11]
upper_cpexp=cpinp[11:]
lower_cpexp=cpinp[:11]

#XGRID

xupper=xcval[:184]
xlower=xcval[184:]

#SIMULATED CP VALUES

upper_simulcp=truecp[:184]
lower_simulcp=truecp[184:]

upper_simulcfx=truecfx[:184]
lower_simulcfx=truecfx[184:]


#CNN OUTPUTS

upper_cnncp=cnncp[:184]
lower_cnncp=cnncp[184:]

upper_cnncfx=cnncfx[:184]
lower_cnncfx=cnncfx[184:]


plt.scatter(upper_xcp,upper_cpexp,color="black",label="exp-cp")
plt.scatter(lower_xcp,lower_cpexp,color="black")
plt.plot(xlower,lower_simulcp,color="red",label="simul-cp")
plt.plot(xupper,upper_simulcp,color="red")
plt.plot(xlower,lower_cnncp,color="blue",label="cnn-cp")
plt.plot(xupper,upper_cnncp,color="blue")
plt.title("eta=0.65")
plt.xlabel("x/c")
plt.ylabel("cp")
plt.legend()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.show()

plt.plot(xlower,lower_simulcfx,color="red",label="simul-cfx")
plt.plot(xupper,upper_simulcfx,color="red")
plt.plot(xlower,lower_cnncfx,color="blue",label="cnn-cfx")
plt.plot(xupper,upper_cnncfx,color="blue")
plt.title("eta=0.65")
plt.xlabel("x/c")
plt.ylabel("cfx")
plt.legend()
plt.show()





