#!/usr/bin/env python

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
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

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.config import config
from blocks.serialization import load

import draw.datasets as datasets
from draw.draw_classify_simple import *

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

# these aren't paramed yet in a generic way, but these values work
ROWS = 8
COLS = 8

def img_grid(y, arr, cx, cy, global_scale=True):
    N, channels, height, width = arr.shape
    cx = cx.flatten()
    cy = cy.flatten()

    global ROWS, COLS
    rows = ROWS
    cols = COLS
    # rows = int(np.sqrt(N))
    # cols = int(np.sqrt(N))

    # if rows*cols < N:
    #     cols = cols + 1

    # if rows*cols < N:
    #     rows = rows + 1

    total_height = rows * height + 7
    total_width  = cols * width + 7

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((3, total_height, total_width))# highlight writing window
    I.fill(1)

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = (255*arr[i]).astype(np.uint8)
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height+r, c*width+c
        I[0:3, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
        c_y = np.max((np.min((cy[i], 27)), 0))
        c_x = np.max((np.min((cx[i], 27)), 0))
        I[0:3, np.round(offset_y + c_y), np.round(offset_x + c_x)] = np.array([255,0,0]).astype(np.uint8)

    # if(channels == 1):
    #     out = I.reshape( (total_height, total_width) )
    # else:
    out = np.dstack(I).astype(np.uint8)
    img = Image.fromarray(out)
    print('-'.join(['%d' % np.dot(i,np.arange(10)) for i in y]))
    return img


def run_classifier(p, subdir):
    if isinstance(p, Model):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    # get test data
    dataset = 'mnist'
    image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)

    # take a single test point
    image_id = 1000
    feature_test = data_test._data_sources[0][image_id].reshape(1,1,28,28)
    label_test = data_test._data_sources[1][image_id]
    print('gt: %d' % label_test[0])

    draw = model.get_top_bricks()[0]
    # reset the random generator
    del draw._theano_rng
    del draw._theano_seed
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling classify function...")

    feature = T.ftensor4("feature")
    label = draw.classify(feature)

    do_classify = theano.function([feature], outputs=label, allow_input_downcast=True)

    #------------------------------------------------------------
    logging.info("Sampling and saving images...")

    global ROWS, COLS
    y,r,c,cx,cy = do_classify(feature_test)
    n_iter = y.shape[0]
    r = r.reshape((n_iter,1,28,28))
    img = img_grid(y, r, cx, cy)
    img.save("{0}/result.png".format(subdir))



model_file = 'c:\users\p2admin\documents\max\projects\draw\mnist-simple-20161030-002342\mnist'
with open(model_file, "rb") as f:
    p = load(f, 'model')

subdir = 'c:\users\p2admin\documents\max\projects\draw\mnist-simple-20161030-002342/result'
if not os.path.exists(subdir):
    os.makedirs(subdir)

run_classifier(p, subdir)



