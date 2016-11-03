import numpy as np
import h5py
import tarfile, os
import sys
import cStringIO as StringIO
import tarfile
import time
import zlib

PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarReader(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            raise StopIteration()
        name = entry.name[len(PREFIX):-len(SUFFIX)]
        fileobj = self.tfile.extractfile(entry)
        buf = zlib.decompress(fileobj.read())
        arr = np.load(StringIO.StringIO(buf))
        return arr, name

    def close(self):
        self.tfile.close()

train_dataset = NpyTarReader('draw/datasets/shapenet10_train.tar')
test_dataset = NpyTarReader('draw/datasets/shapenet10_test.tar')

train_features = []
train_targets = []
test_features = []
test_targets = []
for index, (array, name) in enumerate(train_dataset):
    if int(name[-3:])==1:
        train_features.append(array.flatten())
        train_targets.append([int(name[0:3])])
for index, (array, name) in enumerate(test_dataset):
    if int(name[-3:]) == 1:
        test_features.append(array.flatten())
        test_targets.append([int(name[0:3])])

train_features = np.array(train_features)
train_targets = np.array(train_targets)
test_features = np.array(test_features)
test_targets = np.array(test_targets)
train_n, p = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File('shapenet10.hdf5', mode='w')
features = f.create_dataset('features', (n, p), dtype='uint8')
targets = f.create_dataset('targets', (n, 1), dtype='uint8')

features[...] = np.vstack([train_features, test_features])
targets[...] = np.vstack([train_targets, test_targets])

features.dims[0].label = 'batch'
features.dims[1].label = 'features'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train': {'features': (0, train_n), 'targets': (0, train_n)},
    'test': {'features': (train_n, n), 'targets': (train_n, n)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()