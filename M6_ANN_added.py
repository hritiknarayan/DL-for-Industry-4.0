# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:53:25 2020

@author: 91996

"""




# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:54:57 2020

@author: 91996
"""

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
import matplotlib.pyplot as plt



from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

set_session(sess)

scaler=preprocessing.MinMaxScaler(feature_range=(-0.7,0.7))


from keras import optimizers

from keras import backend as K


def percent_mean_absolute_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.mean(K.abs((y_pred - y_true)) / K.mean(K.clip(K.abs(y_true),
                                                           0.01,
                                                           None)))
    return 100. * K.mean(diff)

def interpolation(x1,z1,x2,z2,x):
    
    return ((x2-x)*z1+(x-x1)*z2)/(x2-x1)

inputs = genfromtxt('M6expt.txt', delimiter=',')
#outputs=genfromtxt('CPo.txt', delimiter=',')
outputs=genfromtxt('CFXo.txt', delimiter=',')

X=inputs.reshape(6,3*11)
Yin=outputs.reshape(6,2*184)

yvalarray=[2.952035964E-01,6.494479179E-01,9.594116807E-01,1.180814385E+00,1.328416228E+00,1.416977286E+00]


yexpeta=[0.47232575715,0.811809888890,1.077493122993] #(for 0.32,0.55,0.73)

interpolinp=np.array([0 for i in range(3*3*11)]).reshape(3,3*11)
interpoloutp=np.array([0 for i in range(3*2*184)]).reshape(3,2*184)

inp32=[interpolation(yvalarray[0],X[0][j],yvalarray[1],X[1][j],yexpeta[0]) for j in range(33)]
inp55=[interpolation(yvalarray[1],X[1][j],yvalarray[2],X[2][j],yexpeta[1]) for j in range(33)]
inp73=[interpolation(yvalarray[2],X[2][j],yvalarray[3],X[3][j],yexpeta[2]) for j in range(33)]

out32=[interpolation(yvalarray[0],Yin[0][j],yvalarray[1],Yin[1][j],yexpeta[0]) for j in range(2*184)]
out55=[interpolation(yvalarray[1],Yin[1][j],yvalarray[2],Yin[2][j],yexpeta[1]) for j in range(2*184)]
out73=[interpolation(yvalarray[2],Yin[2][j],yvalarray[3],Yin[3][j],yexpeta[2]) for j in range(2*184)]


Y=0.6*(Yin-np.amin(Yin))/(2*np.amax(Yin))+0.2
y32=0.6*(out32-np.amin(Yin))/(2*np.amax(Yin))+0.2
y55=0.6*(out55-np.amin(Yin))/(2*np.amax(Yin))+0.2
y73=0.6*(out73-np.amin(Yin))/(2*np.amax(Yin))+0.2



#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/6)

    

X_train=np.array([X[0],X[1],X[3],X[4],X[5],inp32,inp55,inp73]).reshape(8,3*11)
X_test=np.array(X[2]).reshape(1,3*11)
Y_train=np.array([Y[0],Y[1],Y[3],Y[4],Y[5],y32,y55,y73]).reshape(8,2*184)
Y_test=np.array(Y[2]).reshape(1,2*184)

"""

    By this point, all the data is normalized and split and ready to be fed into the ANN
    
    Sequential ANN model lies below:
    
"""





model=Sequential()

model.add(Dense(units = 10, input_dim = 33))
model.add(Activation("relu"))
model.add(Dense(units = 10))
model.add(Activation("relu"))
model.add(Dense(units = 184*2,activation = 'sigmoid'))




sgd = optimizers.SGD(lr=0.01,momentum=0.9, decay=0.00005, nesterov=True)

adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss='mean_squared_error', optimizer=adam,metrics=[percent_mean_absolute_error])


####################### This is the part where you feed zack's weights #################


weights = np.load("weights 67.npy")

def _set_weights_to_layers(model, candidate,bias = True):
        """Set model weights and bias value as candidates value
        Args:
            candidate: Candidate tensor
        """
        last_idx = 0

        # Iterate over every layer
        for layer_idx in range(0, 5, 1):
            if layer_idx == 1 or layer_idx == 3 :
                continue
           
            
            # Get layer dimensions
            w_shape = np.shape(model.layers[layer_idx].get_weights()[0])
            w_numel = np.prod(w_shape)
            if True :
                b_shape = np.shape(model.layers[layer_idx].get_weights()[1])
                b_numel = np.prod(b_shape)

            # Decode the candidate and get weight, bias matrices
            weight = candidate[last_idx:last_idx +
                               w_numel].reshape(w_shape)
            last_idx += w_numel
            if True :
                bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
                last_idx += b_numel
            else :
                bias = []

            # Set layer weight, bias
            if True :
                model.layers[layer_idx].set_weights([weight,bias])
            else :
                model.layers[layer_idx].set_weights([weight])

            
            
            
            
            
def _objective(output, expected):
    """Compute MSE between output and expected value
    Args:
        output: Output from model
        expected: Expected output
    Returns:
        MSE error as tensor
    """


    return 0.5 * np.sum(np.power(expected - output, 2), axis = 1)

def best_apply(model,X,y,weights):
    loss_best = 10**5
    index_best = 0
    for i in range(200):
        _set_weights_to_layers(model,weights[i],True)
        vec_output = model.predict(X)
        loss = np.mean(_objective(vec_output, y))
        if loss <= loss_best :
            loss_best = loss
            index_best = i
    _set_weights_to_layers(model,weights[index_best],True)  

best_apply(model,X_train,Y_train,weights)  




############ End of zack's part #########################################
"""

TO TRAIN THE NETWORK:
    
final=model.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=400,batch_size=8)


This is repeated multiple times.

TO CHECK THE FINAL ERROR (AFTER DENORMALIZATION), use:
    
sess.run(percent_mean_absolute_error((Y_test-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin),(model.predict(X_test)-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin)))

"""

mape = percent_mean_absolute_error((Y_test-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin),(model.predict(X_test)-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin))

def Results(model,X_test,Yin):
    print("###### Results  : #######")
    y_pred = (model.predict(X_test)-0.2)*2*np.amax(Yin)/0.6+np.amin(Yin)
    xcval = genfromtxt('xcvalues.txt', delimiter=',')

    truecfx=np.load("cfxtrue.npy").reshape(368)
    
    y_pred = y_pred.reshape(368)
    
    #XGRID
    
    xupper=xcval[:184]
    xlower=xcval[184:]
    
    upper_simulcfx=truecfx[:184]
    lower_simulcfx=truecfx[184:]
    
    
    #CNN OUTPUTS
    
    
    upper_cnncfx= y_pred[:184]
    lower_cnncfx= y_pred[184:]
    
    
#    
    plt.plot(xlower,lower_simulcfx,color="red",label="simul-cfx")
    plt.plot(xupper,upper_simulcfx,color="red")
    plt.plot(xlower,lower_cnncfx,color="blue",label="cnn-cfx")
    plt.plot(xupper,upper_cnncfx,color="blue")
    plt.title("eta=0.65")
    plt.xlabel("x/c")
    plt.ylabel("cfx")
    plt.legend()
    plt.show()
    plt.close()
    
Results(model,X_test,Yin)