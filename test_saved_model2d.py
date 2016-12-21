#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import ipdb
import time
import cPickle as pickle

from argparse import ArgumentParser
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum, Scale
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import load
from leNet import *
from fuel.datasets.binarized_mnist import BinarizedMNIST


try:
    from blocks.extras import Plot
except ImportError:
    pass

import draw.datasets as datasets
from draw.draw_classify_simple import *
from draw.samplecheckpoint import SampleCheckpoint

sys.setrecursionlimit(100000)

with open('./results/potcup2d20161217-220947/potcup2d', "rb") as f:
    p = load(f, 'model')
    f.close()
if isinstance(p, Model):
    model = p

m = model.get_top_bricks()[0]
x = tensor.tensor4('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
prob, r, c, cx, cy = m.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
f = theano.function([x], [prob, r, c, cx, cy])

mnist_train = H5PYDataset('./draw/datasets/potcup2d_ux.hdf5', which_sets=('train',))
handle = mnist_train.open()
model_idx = 0
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
prob, r, c, cx, cy = f(xx)
print(prob)

################################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
# gs1.update(left=0.05, right=0.48, wspace=0.05)
ax_mnist = plt.subplot(gs1[0:3, 0:3])
ax_mnist.axis('equal')

ax_acc = plt.subplot(gs1[0:3,3:5])

ax_glimpse0 = plt.subplot(gs1[3,0])
ax_glimpse1 = plt.subplot(gs1[3,1])
ax_glimpse2 = plt.subplot(gs1[3,2])
ax_glimpse3 = plt.subplot(gs1[3,3])
ax_glimpse4 = plt.subplot(gs1[3,4])
ax_glimpse0.axis('equal')
ax_glimpse1.axis('equal')
ax_glimpse2.axis('equal')
ax_glimpse3.axis('equal')
ax_glimpse4.axis('equal')

ax_canvas0 = plt.subplot(gs1[4,0])
ax_canvas1 = plt.subplot(gs1[4,1])
ax_canvas2 = plt.subplot(gs1[4,2])
ax_canvas3 = plt.subplot(gs1[4,3])
ax_canvas4 = plt.subplot(gs1[4,4])
ax_canvas0.axis('equal')
ax_canvas1.axis('equal')
ax_canvas2.axis('equal')
ax_canvas3.axis('equal')
ax_canvas4.axis('equal')

ax_mnist.imshow(xx.reshape(28,28), cmap='Greys', interpolation='nearest')
# ax_mnist.set_xlim([0, 28])
# ax_mnist.set_ylim([0, 28])
for i in range(m.n_iter-1):
    x = cx[i]
    y = cy[i]
    ax_mnist.text(x , y, i, fontsize=15, color='red')
    import matplotlib.patches as patches
    p = patches.Rectangle(
        (x-m.read_N/2. , y-m.read_N/2.), m.read_N, m.read_N,
        fill=False, clip_on=False, color='red'
        )
    ax_mnist.add_patch(p)

t = prob[:,0,:]
ax_acc.imshow(t.transpose(), interpolation='nearest', cmap=plt.cm.viridis,extent=[0,5,10,0])
# ax_acc.xlabel('time iteration')
# ax_acc.ylabel('class index')
# ax_acc.colorbar()

import numpy
glimpse_idx = 0
glimpse0 = numpy.zeros((28,28))
canvas0 = numpy.zeros((28,28))
# glimpse_idx = glimpse_idx + 1
glimpse0 = r[glimpse_idx,:].reshape(28,28)
canvas0 = canvas0 + glimpse0
ax_glimpse0.imshow(glimpse0, cmap='Greys', interpolation='nearest')
ax_canvas0.imshow(canvas0, cmap='Greys', interpolation='nearest')
ax_glimpse0.get_xaxis().set_visible(False)
ax_glimpse0.get_yaxis().set_visible(False)
ax_canvas0.get_xaxis().set_visible(False)
ax_canvas0.get_yaxis().set_visible(False)

glimpse_idx = 1
glimpse1 = numpy.zeros((28,28))
canvas1 = numpy.zeros((28,28))
# glimpse_idx = glimpse_idx + 1
glimpse1 = r[glimpse_idx,:].reshape(28,28)
canvas1 = canvas0 + glimpse1
# ax_glimpse0.get_xaxis().set_visible(False)
# ax_glimpse0.get_yaxis().set_visible(False)
ax_glimpse1.imshow(glimpse1, cmap='Greys', interpolation='nearest')
ax_canvas1.imshow(canvas1, cmap='Greys', interpolation='nearest')
ax_glimpse1.get_xaxis().set_visible(False)
ax_glimpse1.get_yaxis().set_visible(False)
ax_canvas1.get_xaxis().set_visible(False)
ax_canvas1.get_yaxis().set_visible(False)

glimpse_idx = 2
glimpse2 = numpy.zeros((28,28))
canvas2 = numpy.zeros((28,28))
# glimpse_idx = glimpse_idx + 1
glimpse2 = r[glimpse_idx,:].reshape(28,28)
canvas2 = canvas1 + glimpse2
ax_glimpse2.imshow(glimpse2, cmap='Greys', interpolation='nearest')
ax_canvas2.imshow(canvas2, cmap='Greys', interpolation='nearest')
ax_glimpse2.get_xaxis().set_visible(False)
ax_glimpse2.get_yaxis().set_visible(False)
ax_canvas2.get_xaxis().set_visible(False)
ax_canvas2.get_yaxis().set_visible(False)

glimpse_idx = 3
glimpse3 = numpy.zeros((28,28))
canvas3 = numpy.zeros((28,28))
# glimpse_idx = glimpse_idx + 1
canvas3 = canvas2 + glimpse3
ax_glimpse3.imshow(glimpse3, cmap='Greys', interpolation='nearest')
ax_canvas3.imshow(canvas3, cmap='Greys', interpolation='nearest')
ax_glimpse3.get_xaxis().set_visible(False)
ax_glimpse3.get_yaxis().set_visible(False)
ax_canvas3.get_xaxis().set_visible(False)
ax_canvas3.get_yaxis().set_visible(False)


glimpse_idx = 4
glimpse4 = numpy.zeros((28,28))
canvas4 = numpy.zeros((28,28))
# glimpse_idx = glimpse_idx + 1
glimpse4 = r[glimpse_idx,:].reshape(28,28)
canvas4 = canvas3 + glimpse4
# ax_glimpse0.get_xaxis().set_visible(False)
# ax_glimpse0.get_yaxis().set_visible(False)
ax_glimpse4.imshow(glimpse4, cmap='Greys', interpolation='nearest')
ax_canvas4.imshow(canvas4, cmap='Greys', interpolation='nearest')
ax_glimpse4.get_xaxis().set_visible(False)
ax_glimpse4.get_yaxis().set_visible(False)
ax_canvas4.get_xaxis().set_visible(False)
ax_canvas4.get_yaxis().set_visible(False)

plt.show(True)