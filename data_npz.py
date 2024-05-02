# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:49:39 2024

@author: rsarker
"""

import numpy as np



# training data

data_train = np.load('train_data_re.npz')
data_xy = data_train['data_xy']
panel_data = data_train['panel_data']
aoa = data_train['aoa']
re = data_train['re']
cl = data_train['cl']
cd = data_train['cd']
cm = data_train['cm']


print(data_xy.shape[0])
print(data_xy.shape[1])
print(data_xy.shape[2])

# testing data
data_test = np.load('test_data_re_23012.npz')
airfoil_names_test = data_test['airfoil_names_test']
data_xy_test = data_test['data_xy_test']
panel_data_test = data_test['panel_data_test']
aoa_test = data_test['aoa_test']
re_test = data_test['re_test']
cl_test = data_test['cl_test']
cd_test = data_test['cd_test']
cm_test = data_test['cm_test']

print(data_xy_test.shape[0])
print(data_xy_test.shape[1])
print(data_xy_test.shape[2])


num_samples = data_xy.shape[0]     # num_samples = 8663
npoints = data_xy.shape[1]         # npoints = 201
num_cp = npoints - 1               # npoints = 200
nf = data_xy.shape[2]              # nf = 2 ( x & y coordinates)
xtrain = np.zeros((num_samples,npoints*nf+2+2))  
ytrain = np.zeros((num_samples,1))       

for i in range(num_samples):
    xtrain[i,:npoints] = data_xy[i,:,0]
    xtrain[i,npoints:2*npoints] = data_xy[i,:,1]
    xtrain[i,-4] = panel_data[i,0]
    xtrain[i,-3] = panel_data[i,1]
    xtrain[i,-2] = aoa[i]
    xtrain[i,-1] = re[i]
    
    # ytrain[i,0] = lift[i]