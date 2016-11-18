# MAX: simple RNN model w/o VAE



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
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum, Scale
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, TestOnly
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER, WEIGHT
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import load
from leNet import *
from fuel.datasets.binarized_mnist import BinarizedMNIST
from PIL import Image

try:
    from blocks.extras import Plot
except ImportError:
    pass

import draw.datasets as datasets
from draw.draw_classify_simple import *
from draw.samplecheckpoint import SampleCheckpoint

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def draw():
    from scipy.io import loadmat
    rows = 5
    cols = 20
    height, width = (28, 28)
    N = 100
    train_dataset1 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data1.mat').get('final_1')
    train_dataset2 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data2.mat').get('final_2')
    train_dataset3 = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/data3.mat').get('final_3')
    test_dataset = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/test_data.mat').get('test')
    total_height = rows * height + (rows-1)
    total_width  = cols * width + (cols-1)
    arr = test_dataset
    arr = arr - arr.min()
    # scale = (arr.max() - arr.min())
    # arr = arr / scale
    I = np.zeros((3, total_height, total_width))# highlight writing window
    I.fill(1)
    for i in xrange(N):
        r = i // cols
        c = i % cols
        this = (255*arr[i]).astype(np.uint8)
        offset_y, offset_x = r*height+r, c*width+c
        I[:, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this.reshape(1,height,width)
    out = np.dstack(I).astype(np.uint8)
    img = Image.fromarray(out)
    img.save("crosstest.png")


import tarfile
def visualize_filter():
    oldmodel_address = 'C:\Users\p2admin\Documents\Max\Projects\draw/results/cross-class-20161110-005240'  # window 5, iter 3
    with open(oldmodel_address + "/cross", "rb") as f:
        tarball = tarfile.open(oldmodel_address + "/cross", 'r')
        draw = numpy.load(tarball.extractfile(tarball.getmember('_parameters')))
        # del draw
        f.close()

    # W0 = draw['|drawclassifymodel|convolutionalsequence|conv_0.W']
    # W1 = draw['|drawclassifymodel|convolutionalsequence|conv_1.W']
    # W1 = upsample(W1, 2)
    # W1_layer0 = deconv(W0, W1)

    W1_layer0 = draw['|drawclassifymodel|convolutionalsequence|conv_1.W']

    rows = 5
    cols = 10
    height, width = (W1_layer0.shape[2], W1_layer0.shape[3])
    N = 2
    total_height = rows * height + (rows-1)
    total_width  = cols * width + (cols-1)
    arr = W1_layer0
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    arr = arr / scale
    I = np.zeros((3, total_height, total_width))# highlight writing window
    I.fill(1)
    for i in xrange(N):
        r = i // cols
        c = i % cols
        this = (255*arr[i]).astype(np.uint8)
        offset_y, offset_x = r*height+r, c*width+c
        I[:, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this.reshape(1,height,width)
    out = np.dstack(I).astype(np.uint8)
    img = Image.fromarray(out)
    img.save(oldmodel_address + "/debug.png")

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def viz2(V):
    V = V / np.max(V) * 0.3
    V[V < 0] = 0

    x = y = t = []
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if V[i, j] != 0:
                x = x + [i]
                y = y + [j]
                t = t + [V[i, j]]
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.scatter(x, y, c=t, marker='o', s=50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])
    #     plt.zlim((0,V.shape[2]))
    cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.show()

import scipy.ndimage as ndimage
def upsample(X,d):
    XX = np.zeros((X.shape[0],X.shape[1],d*X.shape[2],d*X.shape[3]))
    for i,xi in enumerate(X):
        for j,xij in enumerate(xi):
            tmp = (ndimage.zoom(xij, d))
            XX[i,j,:,:] = tmp
    return np.float32(XX)

from theano.sandbox.cuda.dnn import dnn_conv
def deconv(W0, W1):
    print(W0.shape)
    print(W1.shape)
    filter_size = W0.shape[2:4]
    num_filter = W0.shape[0]
    num_ch = W1.shape[0]
    W1_layer0 = dnn_conv(img=W1, kerns=W0.transpose(1, 0, 2, 3), border_mode='valid', subsample=(1, 1))
    W1_layer0 = np.asarray(W1_layer0.eval())
    print(W1_layer0.shape)
    return W1_layer0


def train():
    batch_size = 100
    n_iter = 1
    attention = 28
    learning_rate = 1e-4
    epochs = 300
    dataset = 'cross'
    image_size = (28, 28)
    channels = 1

    train_set = H5PYDataset('c:/users/p2admin/documents/max/projects/draw/draw/datasets/cross_function_small.hdf5', which_sets=('train',))
    test_set = H5PYDataset('c:/users/p2admin/documents/max/projects/draw/draw/datasets/cross_function_small.hdf5', which_sets=('test',))
    # train_n = train_set.num_examples
    train_n = 300
    train_stream = Flatten(
        DataStream.default_stream(train_set, iteration_scheme=SequentialScheme(range(100)+range(1000,1100)+range(2000,2100), batch_size)))
    test_stream = Flatten(
        DataStream.default_stream(test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size)))

    subdir = "./results/" + dataset + "-class-" + time.strftime("%Y%m%d-%H%M%S")
    # ----------------------------------------------------------------------

    for iteration in range(1):
        oldmodel_address = 'C:\Users\p2admin\Documents\Max\Projects\draw/results/cross-class-/cross' #window 5, iter 3
        try:
            with open(oldmodel_address, "rb") as f:
                # oldmodel = pickle.load(f)
                oldmodel = load(f, 'model')
                draw = oldmodel.get_top_bricks()[0]
                draw.n_iter = n_iter
                # del oldmodel
                f.close()
        except:
            ## initialize trained model
            draw = DrawClassifyModel(image_size=image_size, channels=channels, attention=attention)
            draw.push_initialization_config()
            # draw.conv_sequence.layers[0].weights_init = TestOnly()
            # draw.conv_sequence.layers[1].weights_init = Uniform(width=.09)
            # draw.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
            # draw.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
            draw.initialize()

        # ------------------------------------------------------------------------
        x = tensor.matrix('features') # keyword from fuel
        y = tensor.matrix('targets') # keyword from fuel

        y_hat, _, _, _, _ = draw.classify(x)

        y_hat_last = y_hat[-1,:,:] # output should be batch_size * class
        # y_hat_last = y_hat
        # # classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
        y_int = T.cast(y, 'int64')
        # recognition = -T.mean(T.log(y_hat_last)[T.arange(batch_size), y_int]) # guess (rnn_iter (16), class (10), batch_size)
        # recognition.name = "recognition"
        #
        tol = 1e-4
        recognition_convergence = (-y_hat*T.log2(y_hat+tol)).sum(axis=2).mean(axis=1).flatten()
        recognition_convergence.name = "recognition_convergence"


        # from LeNet
        recognition = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last).copy(name='recognition'))


        # recognition_all = T.as_tensor_variable([CategoricalCrossEntropy().apply(y_int.flatten(), y_hat[i,:,:]).copy(name='recognition-%s' % i) for i in range(n_iter)]).copy(name='recognition_all')
        recognition_all = T.as_tensor_variable([-T.mean(T.log(y_hat[i,:,:])[T.arange(batch_size), y_int]) for i in range(n_iter)]).copy(name='recognition_all')
        error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last).copy(name='error'))

        if np.mod(iteration, 2)==0:
            cost = recognition
        else:
            # cost = recognition + recognition_convergence.mean()
            # cost = T.mean(-theano.shared(2)*recognition_all * recognition_convergence + theano.shared(6)*recognition_all + recognition_convergence)
            cost = T.dot(-T.constant(2.0) * recognition_all * recognition_convergence + T.constant(-2.*np.log2(0.1)) * recognition_all
                         + T.constant(-np.log(0.1)) * recognition_convergence, T.constant(np.arange(n_iter)))

        # _, activated_id = T.max_and_argmax(y_hat_last, axis=1)
        # error = theano.tensor.neq(activated_id.flatten(), y_int.flatten()).sum()/float(batch_size)

        # ------------------------------------------------------------
        cg = ComputationGraph([cost])

        # W_conv = VariableFilter(roles=[WEIGHT])(cg.variables)[1]
        # cost = cost - T.constant(1000000.)*T.minimum(W_conv,T.constant(0.)).sum()
        cost.name = "cost"

        params = VariableFilter(roles=[PARAMETER])(cg.variables)

        algorithm = GradientDescent(
            cost=cost,
            parameters=params,
            step_rule=CompositeRule([
                StepClipping(10.),
                Adam(learning_rate),
            ])
            # step_rule=RMSProp(learning_rate)
            # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
            # step_rule=Scale(learning_rate=learning_rate)
        )

        # ------------------------------------------------------------------------
        # Setup monitors
        monitors = [cost, error]

        train_monitors = monitors[:]
        train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
        train_monitors += [aggregation.mean(algorithm.total_step_norm)]
        # Live plotting...
        plot_channels = [
             ["train_total_gradient_norm", "train_total_step_norm"]
        ]

        # ------------------------------------------------------------

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        plotting_extensions = []

        main_loop = MainLoop(
            model=Model(cost),
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=[
                           Timing(),
                           FinishAfter(after_n_epochs=epochs),
                           TrainingDataMonitoring(
                               train_monitors,
                               prefix="train",
                               after_epoch=True),
                           DataStreamMonitoring(
                               monitors,
                               test_stream,
                               #                updates=scan_updates,
                               prefix="test"),
                           Checkpoint("{}/{}".format(subdir, dataset), save_main_loop=False, before_training=True,
                                      after_epoch=True, save_separately=['log', 'model']),
                           ProgressBar(),
                           Printing()] + plotting_extensions)

        main_loop.run()

        W_mlp1, W_conv, W_rec = VariableFilter(roles=[WEIGHT])(cg.variables)

from scipy.io import loadmat
def test():
    oldmodel_address = 'C:\Users\p2admin\Documents\Max\Projects\draw/results/cross-class-20161109-155530/cross'  # window 5, iter 3
    with open(oldmodel_address, "rb") as f:
        draw = load(f, 'model').get_top_bricks()[0]
        f.close()

    test_dataset = loadmat('c:/users/p2admin/documents/max/projects/draw/draw/datasets/test_data.mat').get('test')
    test_features = []
    test_targets_class = []
    for index, array in enumerate(test_dataset):
        test_features.append(array.reshape(1, np.sqrt(array.shape[0]), np.sqrt(array.shape[0])))
        test_targets_class.append([1])
    test_features = np.array(test_features)
    test_targets_class = np.array(test_targets_class)

    #------------------------------------------------------------
    logging.info("Compiling classify function...")
    feature = T.ftensor4("feature")
    label = draw.classify(feature)
    do_classify = theano.function([feature], outputs=label, allow_input_downcast=True)
    #------------------------------------------------------------

    # y,r,c,cx,cy = do_classify(feature_test)
    y,_,_,_,_ = do_classify(test_features)

    y_last = y[-1,:,:] # output should be batch_size * class
    # y_hat_last = y_hat
    # # classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
    label_test = test_targets_class.astype('int64')
    # error = (MisclassificationRate().apply(label_test.flatten(), y_last)
    #          .copy(name='error_rate'))
    activated_id = np.argmax(y_last, axis=1)
    error = (activated_id.flatten()!=label_test.flatten()).sum()/float(100)

    tol = 1e-4
    recognition_convergence = (-y*np.log2(y+tol)).sum(axis=2).sum(axis=0).mean()
    print(error)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # draw()
    # train()
    # test()
    visualize_filter()