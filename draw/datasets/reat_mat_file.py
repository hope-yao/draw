'''save data to hdf5 format, from Max'''

import numpy as np
import h5py
import tarfile, os
import sys
import cStringIO as StringIO
import tarfile
import time
import zlib
from scipy.io import loadmat
train_dataset1 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data1.mat').get('final_1')
train_dataset2 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data2.mat').get('final_2')
train_dataset3 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data3.mat').get('final_3')
test_dataset = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/test_data.mat').get('test')

train_features = []
train_targets_class = []
train_targets_function = []
test_features = []
test_targets_class = []
test_targets_function = []

for index, array in enumerate(train_dataset1):
    train_features.append(array.reshape(1,np.sqrt(array.shape[0]),np.sqrt(array.shape[0])))
    train_targets_class.append([0])
    train_targets_function.append([1,1,0])
for index, array in enumerate(train_dataset2):
    train_features.append(array.reshape(1,np.sqrt(array.shape[0]),np.sqrt(array.shape[0])))
    train_targets_class.append([1])
    train_targets_function.append([0, 1, 0])
for index, array in enumerate(train_dataset3):
    train_features.append(array.reshape(1,np.sqrt(array.shape[0]),np.sqrt(array.shape[0])))
    train_targets_class.append([2])
    train_targets_function.append([0, 1, 1])
for index, array in enumerate(test_dataset):
    test_features.append(array.reshape(1,np.sqrt(array.shape[0]),np.sqrt(array.shape[0])))
    test_targets_class.append([1])
    test_targets_function.append([0, 1, 0])

train_features = np.array(train_features)
train_targets_class = np.array(train_targets_class) #starts from 0
train_targets_function = np.array(train_targets_function) #starts from 0
test_features = np.array(test_features)
test_targets_class = np.array(test_targets_class)
test_targets_function = np.array(test_targets_function)

train_n, c, p1, p2 = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File('cross_function_small.hdf5', mode='w')
features = f.create_dataset('features', (n, c, p1, p2), dtype='uint8')
targets = f.create_dataset('targets', (n, 3), dtype='uint8')

features[...] = np.vstack([train_features, test_features])
# targets[...] = np.vstack([train_targets_class, test_targets_class])
targets[...] = np.vstack([train_targets_function, test_targets_function])

features.dims[0].label = 'batch'
features.dims[1].label = 'features'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'targets'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train': {'features': (0, train_n), 'targets': (0, train_n)},
    'test': {'features': (train_n, n), 'targets': (train_n, n)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()