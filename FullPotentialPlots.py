# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:34:27 2020

@author: 91996
"""

import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

I = range(0,59)
J = range(0,60)

#GRID GENERATION

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
        
l=[]
for i in range(38,59):
    q=((x[58]-x[i])/(x[58]-x[37]))
    l.append(q)
        
        
        
        
phit = [[0.0 for i in range(0,59)] for j in range(0,60)]
vperturbx = [[0.0 for i in range(0,59)] for j in range(0,60)]
vperturby = [[0.0 for i in range(0,59)] for j in range(0,60)]
vtotx = [[0.0 for i in range(0,59)] for j in range(0,60)]
vtoty = [[0.0 for i in range(0,59)] for j in range(0,60)]
vtotsqrt = [[0.0 for i in range(0,59)] for j in range(0,60)]
        


def calculation(phi,phit,m,I,J,h,k,alp,l):
    
    

    count=0
    
    vinf=330*m #VALUE OF VINF GENERATED FROM VALUE OF M
    ww = 0.9 # Arya 26.03.2020
    
    
    flag=1
    while(flag==1):
        #cout=0
        count = count + 1
        
        
        
        #updating previous iteration value
        for i in I:
            for j in J:
                phit[j][i]=phi[j][i]
                
                
       
        
        
        
        for j in J:
            for i in I:
                print(i,j)

                

                    
                if i == 1 and (j<28 or j>31):
                    if j == 1: #For point A
                        dphix=(phit[j][i+1]-phit[j][i])/(2*h[i]) #Ann/a
                        dphiy=(phit[j+1][i]-phit[j][i])/(2*k[j]) #Ann/b
                        vxx=vinf*np.cos(alp)+dphix             #Ann/c
                        vyy=vinf*np.sin(alp)+dphiy             #Ann/d
                        vtot=vxx*vxx+vyy*vyy #Ann/e
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                                    

                    elif j == 58:  #for point B
                        dphix=(phit[j][i+1]-phit[j][i])/(2*h[i]) #Bnn/a
                        dphiy=(phit[j][i]-phi[j-1][i])/(2*k[j]) #Bnn/b
                        vxx=vinf*np.cos(alp)+dphix #Bnn/c
                        vyy=vinf*np.sin(alp)+dphiy #Bnn/d
                        vtot=vxx*vxx+vyy*vyy       #Bnn/e
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        

                    elif ((j>1 and j<28) or (j>31 and j<58)):    #for E

                        dphix=(phit[j][i+1]-phit[j][i])/(2*h[i]) #Enna
                        dphiy=(phit[j+1][i]-phi[j-1][i])/(2*k[j]) #Nnb
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                elif i==57 and (j<28 or j>31):
                    if j==1:   #for point D all exactly like Dnn/equations
                        dphix=(phit[j][i]-phi[j][i-1])/(2*h[i])
                        dphiy=(phit[j+1][i]-phit[j][i])/(2*k[j])
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                    
                    elif j==58:   #for point C all exact like Cnn/equations
                        dphix=(phit[j][i]-phi[j][i-1])/(2*h[i])
                        dphiy=(phit[j][i]-phi[j-1][i])/(2*k[j])
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    elif ((j>1 and j<28) or (j>31 and j<58)):   #for G

                        dphix=(phit[j][i]-phi[j][i-1])/(2*h[i]) #Gnn/a
                        dphiy=(phit[j+1][i]-phi[j-1][i])/(2*k[j]) #NNb
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                elif j == 1 and i >1 and i <57: #for H all same as equations listed in Hnn series
                    
                    dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i])
                    dphiy=(phit[j+1][i]-phit[j][i])/(2*k[j])
                    vxx=vinf*np.cos(alp)+dphix
                    vyy=vinf*np.sin(alp)+dphiy
                    vtot=vxx*vxx+vyy*vyy
                    
                    vperturbx[j][i]=dphix
                    vperturby[j][i]=dphiy
                    vtotx[j][i]=vxx
                    vtoty[j][i]=vyy
                    vtotsqrt[j][i]=np.sqrt(vtot)
                    
                elif j == 58 and i >1 and i < 57: #for F
                    dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #NNa
                    dphiy=(phit[j][i]-phi[j-1][i])/(2*k[j]) #FNNb
                    vxx=vinf*np.cos(alp)+dphix
                    vyy=vinf*np.sin(alp)+dphiy
                    vtot=vxx*vxx+vyy*vyy
                    
                    vperturbx[j][i]=dphix
                    vperturby[j][i]=dphiy
                    vtotx[j][i]=vxx
                    vtoty[j][i]=vyy
                    vtotsqrt[j][i]=np.sqrt(vtot)
                    
                    
                    
                #For points near the rod

                elif(j==31 and i>=17 and i<=37): #I'
                    dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #NNa
                    dphiy=(phit[j+1][i]-phit[j][i]-k[j]*vinf*np.sin(alp))/(2*k[j]) #Innb
                    vxx=vinf*np.cos(alp)+dphix
                    vyy=vinf*np.sin(alp)+dphiy
                    vtot=vxx*vxx+vyy*vyy
                    
                    vperturbx[j][i]=dphix
                    vperturby[j][i]=dphiy
                    vtotx[j][i]=vxx
                    vtoty[j][i]=vyy
                    vtotsqrt[j][i]=np.sqrt(vtot)
                    
                    dphipx=(phit[j-1][i+1]-phi[j-1][i-1])/(2*h[i]) #taken from normal nn equations
                    vpxx=vinf*np.cos(alp)+dphipx
                    vtotp=vpxx*vpxx #NEGLECTING NORMAL FLOW TO PLATE
                    
                    vperturbx[j-1][i]=dphipx
                    vperturby[j-1][i]=0
                    vtotx[j-1][i]=vpxx
                    vtoty[j-1][i]=0
                    vtotsqrt[j][i]=np.sqrt(vtotp)
                    
                    
                    
                    
                    
                elif(j==28 and i>=17 and i<=37):                                # for J'
                    dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #same as NNa
                    dphiy=(phit[j][i]-phi[j-1][i]-vinf*k[j]*np.sin(alp))/(2*k[j]) #JNNb
                    vxx=vinf*np.cos(alp)+dphix
                    vyy=vinf*np.sin(alp)+dphiy
                    vtot=vxx*vxx+vyy*vyy
                    
                    dphipx=(phit[j+1][i+1]-phi[j+1][i-1])/(2*h[i]) #same as nn equations
                    vpxx=vinf*np.cos(alp)+dphipx
                    vtotp=vpxx*vpxx #NEGLECTING NORMAL FLOW TO PLATE
                    
                    
                    vperturbx[j+1][i]=dphipx
                    vperturby[j+1][i]=0
                    vtotx[j+1][i]=vpxx
                    vtoty[j+1][i]=0
                    vtotsqrt[j][i]=np.sqrt(vtotp)
                    

                #For points in the wake

                elif(j==31 and i>37 and i<58): #For I1'

                    #l=((x[58]-x[i])/(x[58]-x[37]))
                    if i<57: #For I1' other than the point in G
                         #p=1-w+w*l[i-38]
                         #p = 1 # Arya ; forced to 1
                         p = 1 - ww + ww*l[i-38]  # Arya 26.03.2020
                         dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #NNa
                         dphiy=(phit[j+1][i]-phit[j][i]-vinf*p*k[j]*np.sin(alp))/(2*k[j]) #I1NNb
                         
                         vxx=vinf*np.cos(alp)+dphix
                         vyy=vinf*np.sin(alp)+dphiy
                         vtot=vxx*vxx+vyy*vyy
                         
                         vperturbx[j][i]=dphix
                         vperturby[j][i]=dphiy
                         vtotx[j][i]=vxx
                         vtoty[j][i]=vyy
                         vtotsqrt[j][i]=np.sqrt(vtot)
                         
                         dphix=(phit[j-1][i+1]-phi[j-1][i-1])/(2*h[i]) #same as NN equations
                        
                         
                         vxx=vinf*np.cos(alp)+dphix #NEGLECTING NORMAL FLOW TO PLATE
                         
                         vtot=vxx*vxx
                         
                         vperturbx[j-1][i]=dphix
                         vperturby[j-1][i]=0
                         vtotx[j-1][i]=vxx
                         vtoty[j-1][i]=0
                         vtotsqrt[j][i]=np.sqrt(vtot)
                         
                         
                    elif(i==57):   #For point in G

                        #p=1-w+w*l[i-38]
                        #p = 1 # Arya; forced
                        p = 1 - ww + ww*l[i-38]  # Arya 26.03.2020
                        dphix=(phit[j][i]-phi[j][i-1])/(2*h[i]) #GNNa
                        dphiy=(phit[j+1][i]-phit[j][i]-vinf*p*k[j]*np.sin(alp))/(2*k[j]) #same as I1NNb
                         
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                          #boundary condition
                        dphix=(phit[j-1][i]-phi[j-1][i-1])/(2*h[i]) #same as NN equations in G
                        
                         
                        vxx=vinf*np.cos(alp)+dphix
                       
                        vtot=vxx*vxx #NEGLECTING NORMAL FLOW
                        
                        vperturbx[j-1][i]=dphix
                        vperturby[j-1][i]=dphiy
                        vtotx[j-1][i]=vxx
                        vtoty[j-1][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                        
                elif(j==28 and i>37 and i<58):  #For J1'
                    
                    if i<57: #For J1' other than the point in G
                        #p=1-w+w*l[i-38]
                        #p = 1 # Arya ; forced
                        p = 1 - ww + ww*l[i-38]  # Arya 26.03.2020
                        dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #NNa
                        dphiy=(phit[j][i]-phi[j-1][i]-vinf*p*k[j]*np.sin(alp))/(2*k[j]) #J1NNb
                         
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        dphix=(phit[j+1][i+1]-phi[j+1][i-1])/(2*h[i]) #from nn equations
                        
                         
                        vxx=vinf*np.cos(alp)+dphix
                        
                        vtot=vxx*vxx #NEGLECTING NORMAL FLOW
                        
                        vperturbx[j+1][i]=dphix
                        vperturby[j+1][i]=dphiy
                        vtotx[j+1][i]=vxx
                        vtoty[j+1][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        
                    elif(i==57):   #For point in G
                        #p=1-w+w*l[i-38]
                        #p = 1 # Arya ; forced
                        p = 1 - ww + ww*l[i-38]  # Arya 26.03.2020
                        dphix=(phit[j][i]-phi[j][i-1])/(2*h[i]) #Gnna
                        dphiy=(phit[j][i]-phi[j-1][i]-vinf*p*k[j]*np.sin(alp))/(2*k[j]) #same as J1NNb
                         
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                       
                        
                        dphix=(phit[j+1][i]-phi[j+1][i-1])/(2*h[i]) #same as NN equations
                        vxx=vinf*np.cos(alp)+dphix
                        vtot=vxx*vxx #NEGLECTING NORMAL FLOW
                        
                        vperturbx[j+1][i]=dphix
                        vperturby[j+1][i]=dphiy
                        vtotx[j+1][i]=vxx
                        vtoty[j+1][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        

                #For points upstream

                elif(j==29 and i>0 and i<17):# For J2
                    if i>1: #For J2 not next to the border
                        
                        dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i]) #NNa
                        dphiy=(phit[j+2][i]-phi[j-1][i])/(2*k[j]) #J2NNb
                         
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                                                   
                        
                    else:   #For J2 next to the border
                        dphix=(phit[j][i+1]-phit[j][i])/(2*h[i]) #Enna
                        dphiy=(phit[j+2][i]-phi[j-1][i])/(2*k[j]) #J2NNb
                         
                        vxx=vinf*np.cos(alp)+dphix
                        vyy=vinf*np.sin(alp)+dphiy
                        vtot=vxx*vxx+vyy*vyy
                        
                        vperturbx[j][i]=dphix
                        vperturby[j][i]=dphiy
                        vtotx[j][i]=vxx
                        vtoty[j][i]=vyy
                        vtotsqrt[j][i]=np.sqrt(vtot)
                        
                        

                        
                elif(j==30 and i>0 and i<17): #For I2
                    phi[j][i]=phi[j-1][i] #according to given boundary conditions
                    vperturbx[j][i]=vperturbx[j-1][i]
                    vperturby[j][i]=vperturby[j-1][i]
                    vtotx[j][i]=vtotx[j-1][i]
                    vtoty[j][i]=vtoty[j-1][i]
                    vtotsqrt[j][i]=vtotsqrt[j-1][i]
                    

                #For interior points

                elif(i==0 or i==58 or j==0 or j==59 or j==29 or j==30 ):
                    pass #TAKEN CARE IN BOUNDARY CONDS
                else:
                    
                    #INTERIOR POINTS
                    #STANDARD FORMULAE of NN equations

                    dphix=(phit[j][i+1]-phi[j][i-1])/(2*h[i])
                    dphiy=(phit[j+1][i]-phi[j-1][i])/(2*k[j])
                    
                         
                    vxx=vinf*np.cos(alp)+dphix
                    vyy=vinf*np.sin(alp)+dphiy
                    vtot=vxx*vxx+vyy*vyy
                    
                    vperturbx[j][i]=dphix
                    vperturby[j][i]=dphiy
                    vtotx[j][i]=vxx
                    vtoty[j][i]=vyy
                    vtotsqrt[j][i]=np.sqrt(vtot)

                
                


        # updating calculations at farfield boundary points
        for i in I:
            phi[0][i]=phi[1][i]
            phi[59][i]=phi[58][i]
            vperturbx[0][i]=vperturbx[1][i]
            vperturby[0][i]=vperturby[1][i]
            vtotx[0][i]=vtotx[1][i]
            vtoty[0][i]=vtoty[1][i]
            vtotsqrt[0][i]=vtotsqrt[1][i]
            vperturbx[59][i]=vperturbx[58][i]
            vperturby[59][i]=vperturby[58][i]
            vtotx[59][i]=vtotx[58][i]
            vtoty[59][i]=vtoty[58][i]
            vtotsqrt[59][i]=vtotsqrt[58][i]
        for j in J:
            phi[j][0]=phi[j][1]
            phi[j][58]=phi[j][57]
            vperturbx[j][0]=vperturbx[j][1]
            vperturby[j][0]=vperturby[j][1]
            vtotx[j][0]=vtotx[j][1]
            vtoty[j][0]=vtoty[j][1]
            vtotsqrt[j][0]=vtotsqrt[j][1]
            vperturbx[j][58]=vperturbx[j][57]
            vperturby[j][58]=vperturby[j][57]
            vtotx[j][58]=vtotx[j][57]
            vtoty[j][58]=vtoty[j][57]
            vtotsqrt[j][58]=vtotsqrt[j][57]
            


        flag=0
            
          
        
        
        
    return vperturbx,vperturby,vtotx,vtoty,vtotsqrt

def alpha(apl):
    apl = apl*1.0
    return (apl*np.pi)/180


name="cnn1"
phi=np.load("c1cnn.npy").reshape(60,59)
m=0.3
apl=-5.0
alp=alpha(apl)

pertx,perty,totx,toty,tot=calculation(phi,phit,m,I,J,h,k,alp,l)

pert= [[np.sqrt(pertx[j][i]**2+perty[j][i]**2) for i in range(0,59)] for j in range(0,60)]

print("Reached here!"+name)

np.save(name+"totx.npy",totx)
np.save(name+"toty.npy",toty)
np.save(name+"tot.npy",tot)
np.save(name+"pertx.npy",pertx)
np.save(name+"perty.npy",perty)
np.save(name+"pert.npy",pert)

print("done")






























