# MAX: simple RNN model w/o VAE


from __future__ import division, print_function

import sys

sys.path.append("../lib")

import logging
import theano
import theano.tensor as T
from theano import tensor
import numpy as np

from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
from blocks.bricks import Random, Initializable, MLP, Linear
from blocks.bricks.conv import Convolutional, Flattener
from blocks.bricks import Identity, Tanh, Logistic
from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum
from blocks.bricks import Tanh, Identity, Softmax
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

from attention import ZoomableAttentionWindow
from prob_layers import replicate_batch

class Reader(Initializable):
    def __init__(self, x_dim, y_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = x_dim

    def get_dim(self, name):
        if name == 'input':
            return self.y_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'y'], outputs=['r'])
    def apply(self, x, y):
        return x

class AttentionReader(Initializable):
    def __init__(self, x_dim, c_dim, channels, height, width, N, **kwargs):
        super(AttentionReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = channels * N * N

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[c_dim, 4], **kwargs) # input is the output from RNN

        self.children = [self.readout]

    def get_dim(self, name):
        if name == 'input':
            return self.c_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'c'], outputs=['r', 'cx', 'cy', 'delta', 'sigma'])
    def apply(self, x, c):
        l = self.readout.apply(c)

        center_y, center_x, delta, sigma = self.zoomer.nn2att_const_gamma(l)

        r = self.zoomer.read(x, center_y, center_x, delta, sigma)

        return r, center_x, center_y, delta, sigma

class DrawClassifyModel(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, **kwargs):
        super(DrawClassifyModel, self).__init__(**kwargs)

        self.n_iter = 16
        y_dim = 10
        rnn_dim = 16
        num_filters = 16

        rnninits = {
            # 'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
        inits = {
            # 'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
        img_height, img_width = image_size
        x_dim = channels * img_height * img_width

        # Configure attention mechanism
        read_N = attention
        read_N = int(read_N)
        read_dim = x_dim
        reader = AttentionReader(x_dim=x_dim, c_dim=rnn_dim,
                                 channels=channels, width=img_width, height=img_height,
                                 N=read_N, **inits)

        encoder_conv = Convolutional(filter_size=(read_N, read_N), num_filters=num_filters, num_channels=channels, name="CONV_enc", **inits)
        conv_dim = (read_N-read_N+1)**2*num_filters + 4 # cx, cy, delta, sigma
        encoder_mlp = MLP([Identity()], [conv_dim, 4 * rnn_dim], name="MLP_enc", **inits)
        rnn = LSTM(dim=rnn_dim, name="RNN", **rnninits)
        decoder_mlp = MLP([Softmax()], [rnn_dim, y_dim], name="MLP_dec", **inits)

        self.reader = reader
        self.encoder_conv = encoder_conv
        self.encoder_mlp = encoder_mlp
        self.rnn = rnn
        self.decoder_mlp = decoder_mlp

        self.children = [self.reader, self.encoder_conv, self.encoder_mlp, self.rnn,
                         self.decoder_mlp]

    def get_dim(self, name):
        if name == 'y':
            return 10 # for mnist
        elif name == 'h':
            return self.rnn.get_dim('states')
        elif name == 'c':
            return self.rnn.get_dim('cells')
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(DrawClassifyModel, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['y', 'h', 'c'],
               outputs=['y', 'h', 'c'])
    def apply(self, y, c, h, x, dummy):
        r, cx, cy, delta, sigma = self.reader.apply(x, c)
        a = self.encoder_conv.apply(r)
        # a_flatten = Flattener(a)
        # aa = T.flatten(a,outdim=2)
        aa = T.concatenate([T.flatten(a,outdim=2),  T.stack([cx, cy, delta, sigma]).T], axis=1)
        i = self.encoder_mlp.apply(aa)
        h, cc = self.rnn.apply(states=h, cells=c, inputs=i,
                              iterate=False)
        c = c + cc
        y = self.decoder_mlp.apply(c)

        return y, h, c

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['targets'])
    def classify(self, features):
        batch_size = features.shape[0]
        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        y, h, c = self.apply(x=features, dummy=u)

        return y
