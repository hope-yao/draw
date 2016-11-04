import numpy as np

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

dataset = NpyTarReader('datasets/saliency/shapenet10_test.tar')

for index, (array, name) in enumerate(dataset):
    # determine the class number from the filename 
    # make necessary modifications to array
    # save classification number at 'index' of ['targets']
    # save array into at 'index' of ['features'] as [index, 1, array]

# make shapenet10_test.npz file [['targets'] , ['features']]
   
