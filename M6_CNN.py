# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:53:25 2020

@author: Hritik Narayan

"""


#ALL THE LIBRARIES 

import keras
from numpy import genfromtxt
from matplotlib import pyplot
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Reshape,Flatten,MaxPooling2D,AveragePooling2D,Dense,LeakyReLU,Conv2D,Conv2DTranspose,ZeroPadding2D,BatchNormalization,Dropout,Activation
import sklearn.preprocessing as preprocessing
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.models import load_model


#SOME STUFF THAT IS REQUIRED FOR TENSORFLOW-GPU SET UP

from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

set_session(sess)



from keras import optimizers

from keras import backend as K

#THE METRIC USED (MAPE)


def percent_mean_absolute_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.mean(K.abs((y_pred - y_true)) / K.mean(K.clip(K.abs(y_true),
                                                           0.01,
                                                           None)))
    return 100. * K.mean(diff)

#FORMULA USED FOR INTERPOLATION/EXTRA DATA GENERATION

def interpolation(x1,z1,x2,z2,x):
    
    return ((x2-x)*z1+(x-x1)*z2)/(x2-x1)


#THIS SECTION TAKES THE INPUT AND GENERATES THREE NEW INPUT/OUTPUT PAIRS

inputs = genfromtxt('M6expt.txt', delimiter=',')
outputs=genfromtxt('CPo.txt', delimiter=',')
#outputs=genfromtxt('CFXo.txt', delimiter=',')

Xd=inputs.reshape(6,3,11,1)
Yind=outputs.reshape(6,2,184,1)



X=inputs.reshape(6,3*11)
Yin=outputs.reshape(6,2*184)



yvalarray=[2.952035964E-01,6.494479179E-01,9.594116807E-01,1.180814385E+00,1.328416228E+00,1.416977286E+00]


yexpeta=[0.47232575715,0.811809888890,1.077493122993] #(for 0.32,0.55,0.73)


inp32=[interpolation(yvalarray[0],X[0][j],yvalarray[1],X[1][j],yexpeta[0]) for j in range(33)]
inp55=[interpolation(yvalarray[1],X[1][j],yvalarray[2],X[2][j],yexpeta[1]) for j in range(33)]
inp73=[interpolation(yvalarray[2],X[2][j],yvalarray[3],X[3][j],yexpeta[2]) for j in range(33)]

out32=[interpolation(yvalarray[0],Yin[0][j],yvalarray[1],Yin[1][j],yexpeta[0]) for j in range(2*184)]
out55=[interpolation(yvalarray[1],Yin[1][j],yvalarray[2],Yin[2][j],yexpeta[1]) for j in range(2*184)]
out73=[interpolation(yvalarray[2],Yin[2][j],yvalarray[3],Yin[3][j],yexpeta[2]) for j in range(2*184)]


inp32=np.array(inp32).reshape(3,11,1)
inp55=np.array(inp55).reshape(3,11,1)
inp73=np.array(inp73).reshape(3,11,1)

out32=np.array(out32).reshape(2,184,1)
out55=np.array(out55).reshape(2,184,1)
out73=np.array(out73).reshape(2,184,1)



Y=0.6*(Yind-np.amin(Yind))/(2*np.amax(Yind))+0.2
y32=0.6*(out32-np.amin(Yin))/(2*np.amax(Yin))+0.2
y55=0.6*(out55-np.amin(Yin))/(2*np.amax(Yin))+0.2
y73=0.6*(out73-np.amin(Yin))/(2*np.amax(Yin))+0.2

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/6)



X_train=np.array([Xd[0],Xd[1],Xd[3],Xd[4],Xd[5],inp32,inp55,inp73]).reshape(8,3,11,1)
X_test=np.array(Xd[2]).reshape(1,3,11,1)
Y_train=np.array([Y[0],Y[1],Y[3],Y[4],Y[5],y32,y55,y73]).reshape(8,2,184,1)
Y_test=np.array(Y[2]).reshape(1,2,184,1)


"""

The data is ready, by this point, and the sequential model of the CNN is given below:
    
"""

model=Sequential()

model.add(Conv2DTranspose(2,kernel_size=(1,10),input_shape=(3,11,1)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(1,20)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(1,40)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(2,50)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(4,127),padding="same"))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(1,60)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2D(2,kernel_size=(3,3)))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(2,184),padding="same"))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(2,kernel_size=(2,184),padding="same"))
model.add(LeakyReLU(alpha=0.05))

model.add(Conv2DTranspose(1,kernel_size=(2,184),padding="same"))
model.add(Activation("sigmoid"))






#OPTIMIZERS (ADAM IS USED)

sgd = optimizers.SGD(lr=0.01,momentum=0.9, decay=0.0005, nesterov=True)

adam=keras.optimizers.Adam(lr=0.0009 , beta_1=0.9, beta_2=0.999, amsgrad=True)

#MODEL COMPILATION WITH CUSTOM METRIC


model.compile(loss='mean_squared_error', optimizer=adam,metrics=[percent_mean_absolute_error])



"""

TO TRAIN THE NETWORK, I HAVE USED ALTERNATING CYCLES OF BATCH SIZE 2 and 8, as follows:
    
final=model.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=500,batch_size=2)
final=model.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=500,batch_size=8)

This is repeated multiple times.

TO CHECK THE FINAL ERROR (AFTER DENORMALIZATION), use:
    
sess.run(percent_mean_absolute_error((Y_test-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin),(model.predict(X_test)-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin)))

"""












