import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from panel_allairfoil import panel

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras import backend as kb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

from tensorflow.keras import regularizers
import os

def coeff_determination(y_true, y_pred):
    '''
    custom metric to evaluate the neural network training
    it uses the coefficient of detrmination formula
    for more information : https://en.wikipedia.org/wiki/Coefficient_of_determination
    '''
    SS_res =  kb.sum(kb.square(y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def training_data(data,panel_data,aoa,re,lift,drag):
    '''
    Input:
    data ---- this contains the x and y coordinate of airfoil
    panel_data ---- this contains the CL and CD_p determined using the panel method
    aoa ---- angle of attack
    re ---- Reynolds number
    lift ---- lift coefficient from XFoil 
    drag ---- drag coefficient from XFoil
    
    Output:
    xtrain ---- input to the neural network for training (airfoil shape, aoa, re)
    ytrain ---- label of the training dataset (Xfoil lift coefficient)    
    '''
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+2))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def training_data_cp(data,panel_data,aoa,re,lift,drag):
    '''
    Input:
    data ---- this contains the x and y coordinate of airfoil
    panel_data ---- this contains the CL and CD_p determined using the panel method
    aoa ---- angle of attack
    re ---- Reynolds number
    lift ---- lift coefficient from XFoil 
    drag ---- drag coefficient from XFoil
    
    Output:
    xtrain ---- input to the neural network for training 
                (airfoil shape, cp distribution from panel method, aoa, re)
    ytrain ---- label of the training dataset (Xfoil lift coefficient)    
    '''
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+num_cp+2))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,2*npoints:2*npoints+num_cp] = panel_data[i,2:]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def training_data_cl_cd(data_xy,panel_data,aoa,re,lift,drag):
    '''
    Input:
    data_xy ---- this contains the x and y coordinate of airfoil  (8663,201,2)
    panel_data ---- this contains the CL and CD_p determined using the panel method  ( 8663, 202)
    aoa ---- angle of attack   (8663,1)                             
    re ---- Reynolds number (8663,1)
    lift ---- lift coefficient from XFoil (8663,1)
    drag ---- drag coefficient from XFoil  (8663,1)
    
    Output:
    xtrain ---- input to the neural network for training 
                (airfoil shape, CL and CD_p from panel method, aoa, re)
    ytrain ---- label of the training dataset (Xfoil lift coefficient)    
    '''
    num_samples = data_xy.shape[0]                      # num_samples = 8663
    npoints = data_xy.shape[1]                          # npoints = 201
    num_cp = npoints - 1                                # num_cp = 200
    nf = data_xy.shape[2]                               # nf = 2 ( x & y coordinates)
    xtrain = np.zeros((num_samples,npoints*nf+2+2))     # 8663 X 406
    ytrain = np.zeros((num_samples,1))                  # 8663 X 1
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data_xy[i,:,0]              # assign x coordinates for all rows & column 0 to 200
        xtrain[i,npoints:2*npoints] = data_xy[i,:,1]     # assign y coordinates for all rows & column 201 to 401
        xtrain[i,-4] = panel_data[i,0]                   # assign 1st column (CL)  to the xtrain's 4th last column         
        xtrain[i,-3] = panel_data[i,1]                   # assign 2nd column (CDp)  to the xtrain's 3rd last column 
        xtrain[i,-2] = aoa[i]                            # assign angle of attack to the xtrain's 2nd last column
        xtrain[i,-1] = re[i]                             # assign reynolds number to the last column
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def training_data_cl(data,panel_data,aoa,re,lift,drag):
    '''
    This training include only the CL values from the panel method
    Input:
    data ---- this contains the x and y coordinate of airfoil
    panel_data ---- this contains the CL and CD_p determined using the panel method
    aoa ---- angle of attack
    re ---- Reynolds number
    lift ---- lift coefficient from XFoil 
    drag ---- drag coefficient from XFoil
    
    Output:
    xtrain ---- input to the neural network for training 
                (airfoil shape,  , aoa, re)
    ytrain ---- label of the training dataset (Xfoil lift coefficient)    
    '''
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+2+1))  # only Cl from panel that's why 1 less
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,-3] = panel_data[i,0]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain
