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

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy
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
from draw.draw_classify_simple import *
from blocks.serialization import load

# import blocks
# path = 'C:\Users\p2admin\Documents\Max\Projects\draw\mnist-20161009-204142\mnist'
# with open(path, "rb") as source:
#     temp = blocks.serialization.load(source, 'model')
# a = 1
#
# # with open('C:\Users\p2admin\Documents\Max\Projects\draw\mnist-20161009-204142\mnist', "rb") as f:
# #     p = pickle.load(f)

# attention = 2
# n_iter = 16
# rnn_dim = 64
# y_dim = 10
#
# channels = 1
# img_height = 28
# img_width = 28
# x_dim = channels * img_height * img_width
#
# rnninits = {
#     # 'weights_init': Orthogonal(),
#     'weights_init': IsotropicGaussian(0.01),
#     'biases_init': Constant(0.),
# }
# inits = {
#     # 'weights_init': Orthogonal(),
#     'weights_init': IsotropicGaussian(0.01),
#     'biases_init': Constant(0.),
# }
#
# # Configure attention mechanism
# if attention != "":
#     read_N = attention
#     read_N = int(read_N)
#     read_dim = x_dim
#
#     reader = AttentionReader(x_dim=x_dim, y_dim=y_dim,
#                              channels=channels, width=img_width, height=img_height,
#                              N=read_N, **inits)
#     attention_tag = "r%d" % read_N
# else:
#     read_dim = x_dim
#     reader = Reader(x_dim=x_dim, y_dim=y_dim, **inits)
#     attention_tag = "full"
#
#
# rnn = LSTM(dim=rnn_dim, name="RNN", **rnninits)
# encoder_mlp = MLP([Identity()], [(read_dim + rnn_dim), 4 * rnn_dim], name="MLP_enc", **inits)
# decoder_mlp = MLP([Identity()], [rnn_dim, y_dim], name="MLP_dec", **inits)




image_size = (28, 28)
channels = 1
attention = '5'
x_dim = 784

model_file = 'C:\Users\p2admin\Documents\Max\Projects\draw\mnist-simple-20161027-154135\mnist'
logging.info("Loading file %s..." % model_file)
with open(model_file, "rb") as f:
    model = load(f, 'model')

draw = model.get_top_bricks()[0]
# draw = DrawClassifyModel(image_size=image_size, channels=channels, attention=attention)
# draw.initialize()

# ------------------------------------------------------------------------
data_size = 2
x = tensor.matrix('features')
# x = tensor.as_tensor_variable(np.random.randn(data_size,x_dim), name='features')
# y = tensor.as_tensor_variable(np.random.randint(low=0, high=9, size=(1,data_size)), name='targets')

f = theano.function([x], draw.classify(x))

y_hat = f(np.random.randn(data_size,x_dim).astype('float32'))
y = np.random.randint(low=0, high=9, size=(1,data_size))

y_hat_last = y_hat[-1,:,:] # output should be batch_size * class
# classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
y_int = y.astype('int16')
activation = -np.mean(np.log(y_hat_last)[np.arange(data_size), y_int]) # guess (rnn_iter (16), class (10), batch_size)

activated_id = np.argmax(y_hat_last, axis=1)
error = theano.tensor.neq(activated_id.flatten(), y_int.flatten()).sum()/float(data_size)

# recognition_convergence = (-y_hat*T.log2(y_hat)).sum(axis=1).mean()
# recognition_convergence.name = "recognition_convergence"

cost = activation
# cost = activation + recognition_convergence.mean()
cost.name = "cost"