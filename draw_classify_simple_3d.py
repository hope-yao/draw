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
from blocks.bricks import Random, Initializable, MLP, Linear, Rectifier
from blocks.bricks.conv import Convolutional, Flattener, ConvolutionalSequence, MaxPooling
from blocks.bricks import Identity, Tanh, Logistic
from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum
from blocks.bricks import Tanh, Identity, Softmax
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, Uniform
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from toolz.itertoolz import interleave
from cnn3d_bricks import Convolutional3, ConvolutionalTranspose3, Pooling3, MaxPooling3, AveragePooling3, ConvolutionalSequence3, Flattener3

from draw.attention import ZoomableAttentionWindow, ZoomableAttentionWindow3d
from draw.prob_layers import replicate_batch

class Reader(Initializable):
    def __init__(self, x_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.output_dim = x_dim

    def get_dim(self, name):
        if name == 'input':
            return self.c_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x'], outputs=['r'])
    def apply(self, x):
        return x

class AttentionReader3d(Initializable):
    def __init__(self, x_dim, c_dim, channels, height, width, depth, N, **kwargs):
        super(AttentionReader3d, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.img_depth = depth
        self.N = N
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = (height, width, depth)

        self.zoomer = ZoomableAttentionWindow3d(channels, height, width, depth, N)
        self.readout = MLP(activations=[Identity()], dims=[c_dim, 3], **kwargs) # input is the output from RNN

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

    @application(inputs=['x', 'c'], outputs=['r', 'cx', 'cy', 'cz'])
    def apply(self, x, c):
        l = self.readout.apply(c)

        center_x, center_y , center_z = self.zoomer.nn2att_const_gamma(l)

        r = self.zoomer.read_large(x, center_x, center_y, center_z)

        return r, center_x, center_y, center_z

class DrawClassifyModel3d(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, **kwargs):
        super(DrawClassifyModel3d, self).__init__(**kwargs)

        self.n_iter = n_iter

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
        conv_inits = {
            'weights_init': Uniform(width=.2),
            'biases_init': Constant(0.)
        }
        img_height, img_width, img_depth  = image_size
        self.x_dim = channels * img_height * img_width * img_depth

        # # Configure attention mechanism
        # read_N = attention
        # read_N = int(read_N)
        # read_dim = x_dim
        #
        # reader = AttentionReader(x_dim=x_dim, c_dim=rnn_dim,
        #                          channels=channels, width=img_width, height=img_height,
        #                          N=read_N, **inits)

        # encoder_conv = Convolutional(filter_size=(read_N, read_N), num_filters=num_filters, num_channels=channels, name="CONV_enc", **inits)
        # conv_dim = (read_N-read_N+1)**2*num_filters + 4 # cx, cy, delta, sigma
        # encoder_mlp = MLP([Identity()], [conv_dim, 4 * rnn_dim], name="MLP_enc", **inits)
        # rnn = LSTM(dim=rnn_dim, name="RNN", **rnninits)
        # decoder_mlp = MLP([Softmax()], [rnn_dim, y_dim], name="MLP_dec", **inits)

        # self.reader = reader
        # self.encoder_conv = encoder_conv
        # self.encoder_mlp = encoder_mlp
        # self.rnn = rnn
        # self.decoder_mlp = decoder_mlp

        # self.children = [self.reader, self.encoder_conv, self.encoder_mlp, self.rnn,
        #                  self.decoder_mlp]


#-----------------------------------------------------------------------------------------------------------------------
        # USE LeNet

        feature_maps = [16, 24] #[20, 50]
        mlp_hiddens = [250] # 500
        conv_sizes = [5, 5, 5] # [5, 5]
        pool_sizes = [2, 2, 2]
        # image_size = (28, 28)
        output_size = 10

        conv_activations = [Rectifier() for _ in feature_maps]
        mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

        num_channels = 1
        image_shape = (32, 32, 32)
        filter_sizes = [(5,5,5),(5,5,5)]
        feature_maps = feature_maps
        pooling_sizes = [(2,2,2),(2,2,2)]
        top_mlp_activations = mlp_activations
        top_mlp_dims = mlp_hiddens + [output_size]
        border_mode = 'valid'

        conv_step = None
        if conv_step is None:
            self.conv_step = (1, 1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional3(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling3(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence3(self.layers, num_channels,
                                                   image_size=image_shape, **conv_inits)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, **inits)
        self.flattener = Flattener3()

# -----------------------------------------------------------------------------------------------------------------------
        # Configure attention mechanism
        read_N = attention
        read_N = int(read_N)

        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')
        self.conv_out_dim_flatten = np.prod(conv_out_dim)
        reader = AttentionReader3d(x_dim=self.x_dim, c_dim=self.conv_out_dim_flatten,
                                 channels=channels, width=img_width, height=img_height, depth=img_depth,
                                 N=read_N, **inits)
        # reader = Reader(x_dim=self.x_dim)

        self.reader = reader

        self.children = [self.reader, self.conv_sequence, self.flattener, self.top_mlp]

        # application_methods = [self.reader.apply, self.conv_sequence.apply, self.flattener.apply,
        #                        self.top_mlp.apply]
        # super(DrawClassifyModel, self).__init__(application_methods, **conv_inits)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        # self.reader._push_allocation_config()
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


    def get_dim(self, name):
        if name == 'y':
            return 10 # for mnist_lenet
        elif name == 'c':
            return self.conv_out_dim_flatten
        elif name == 'r':
            return self.x_dim
        elif name == 'center_y':
            return 1
        elif name == 'center_x':
            return 1
        elif name == 'center_z':
            return 1
        elif name == 'delta':
            return 1
        else:
            super(DrawClassifyModel3d, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['r', 'c'],
               outputs=['y', 'r', 'c', 'cx', 'cy', 'cz'])
    def apply(self, c, r, x, dummy):
        # r, cx, cy, delta, sigma = self.reader.apply(x, c)
        # a = self.encoder_conv.apply(r)
        # # a_flatten = Flattener(a)
        # # aa = T.flatten(a,outdim=2)
        # aa = T.concatenate([T.flatten(a,outdim=2),  T.stack([cx, cy, delta, sigma]).T], axis=1)
        # i = self.encoder_mlp.apply(aa)
        # h, cc = self.rnn.apply(states=h, cells=c, inputs=i,
        #                       iterate=False)
        # c = c + cc
        # y = self.decoder_mlp.apply(c)

        rr, center_x, center_y, center_z = self.reader.apply(x, c)
        r = r + rr # combine revealed images
        batch_size = r.shape[0]
        c_raw = self.conv_sequence.apply(r.reshape((batch_size,1,32,32,32)))
        c = self.flattener.apply(c_raw)
        y = self.top_mlp.apply(c)

        return y, r, c, center_x, center_y, center_z

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['targets', 'r', 'c', 'cx', 'cy', 'cz'])
    def classify(self, features):
        batch_size = features.shape[0]
        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        # y, r, c, center_x, center_y, delta, sigma = self.apply(x=features, dummy=u)
        y, r, c, cx, cy, cz = self.apply(x=features, dummy=u)

        return y, r, c, cx, cy, cz




#=============================================================================

if __name__ == "__main__":

    import numpy

    image_size = (32,32,32)
    channels = 1
    attention = 5

    import tarfile
    tarball = tarfile.open('./draw3d.pkl', 'r')
    ps = numpy.load(tarball.extractfile(tarball.getmember('_parameters')))
    sorted(ps.keys())
    conv_W0 = ps['|drawclassifymodel3d|convolutionalsequence3|conv_0.W']
    conv_b0 = ps['|drawclassifymodel3d|convolutionalsequence3|conv_0.b']
    conv_W1 = ps['|drawclassifymodel3d|convolutionalsequence3|conv_1.W']
    conv_b1 = ps['|drawclassifymodel3d|convolutionalsequence3|conv_1.b']
    mlp_W0 = ps['|drawclassifymodel3d|mlp|linear_0.W']
    mlp_b0 = ps['|drawclassifymodel3d|mlp|linear_0.b']
    mlp_W1 = ps['|drawclassifymodel3d|mlp|linear_1.W']
    mlp_b1 = ps['|drawclassifymodel3d|mlp|linear_1.b']
    reader_W = ps['|drawclassifymodel3d|reader|mlp|linear_0.W']
    reader_b = ps['|drawclassifymodel3d|reader|mlp|linear_0.b']

    draw = DrawClassifyModel3d(image_size=image_size, channels=channels, attention=attention, n_iter=16)
    draw.push_initialization_config()
    draw.conv_sequence.layers[0].weights_init = Constant(conv_W0)
    draw.conv_sequence.layers[1].weights_init = Constant(conv_W1)
    draw.top_mlp.linear_transformations[0].weights_init = Constant(mlp_W0)
    draw.top_mlp.linear_transformations[1].weights_init = Constant(mlp_W1)
    draw.conv_sequence.layers[0].biases_init = Constant(conv_b0)
    draw.conv_sequence.layers[1].biases_init = Constant(conv_b1)
    draw.top_mlp.linear_transformations[0].biases_init = Constant(mlp_b0)
    draw.top_mlp.linear_transformations[1].biases_init = Constant(mlp_b1)
    draw.reader.readout.weights_init = Constant(reader_W)
    draw.reader.readout.biases_init = Constant(reader_b)

    draw.initialize()

    x = tensor.matrix('input') # keyword from fuel
    y = tensor.matrix('targets') # keyword from fuel
    y_hat, r, c, cx, cy, cz = draw.classify(x)
    f = theano.function([x], [y_hat,r,c,cx,cy,cz])

    from fuel.datasets.hdf5 import H5PYDataset
    train_set = H5PYDataset('./layer3D/shapenet10.hdf5', which_sets=('train',))
    test_set = H5PYDataset('./layer3D/shapenet10.hdf5', which_sets=('test',))
    handle = train_set.open()
    train_data = train_set.get_data(handle, slice(0, 10))
    handletest = test_set.open()
    test_data = test_set.get_data(handletest, slice(0, 10))
    train_features = train_data[0]

    # for [y, r, c, cx, cy, cz] in f(train_features[0,:,:,:].reshape(1,32*32*32)):
    #     print(cx, cy, cz)
    y, r, c, cx, cy, cz = f(train_features[5,:,:,:].reshape(1,32*32*32))

    '''visualize 3D data'''
    def plot_cube(ax, x, y, z, inc, a):
        "x y z location and alpha"
        ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z, alpha=a,facecolors='y')
        ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z + inc, alpha=a,facecolors='y')

        ax.plot_surface(x, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
        ax.plot_surface(x + inc, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

        ax.plot_surface([[x, x], [x + inc, x + inc]], y, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
        ax.plot_surface([[x, x], [x + inc, x + inc]], y + inc, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

    def viz2(V,cx,cy,cz):

        x = y = z = t = []
        x1 = y1 = z1 = t1 = []
        x2 = y2 = z2 = t2 = []
        x3 = y3 = z3 = t3 = []
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    if V[i, j, k] != 0:
                        if (V[i, j, k] > 1e-1):
                            x = x + [i]
                            y = y + [j]
                            z = z + [k]
                            t = t + [V[i, j, k]]
                        if i==15:
                            y1 = y1 + [j]
                            z1 = z1 + [k]
                            t1 = t1 + [V[i, j, k]]
                        if j==15:
                            x2 = x2 + [i]
                            z2 = z2 + [k]
                            t2 = t2 + [V[i, j, k]]
                        if k==15:
                            x3 = x3 + [i]
                            y3 = y3 + [j]
                            t3 = t3 + [V[i, j, k]]

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        t = np.asarray(t)
        y1 = np.asarray(y1)
        z1 = np.asarray(z1)
        t1 = np.asarray(t1)
        x2 = np.asarray(x2)
        z2 = np.asarray(z2)
        t2 = np.asarray(t2)
        x3 = np.asarray(x3)
        y3 = np.asarray(y3)
        t3 = np.asarray(t3)

        # # slice along axis
        # fig, axes = plt.subplots(nrows=2, ncols=2,)
        #
        # ax1 = axes.flat[1]
        # im = ax1.scatter(y1, z1, c=t1, marker='o', s=30)
        # plt.xlim(0, V.shape[0])
        # plt.ylim(0, V.shape[1])
        #
        # ax2 = axes.flat[2]
        # im = ax2.scatter(x2, z2, c=t2, marker='o', s=30)
        # plt.xlim(0, V.shape[0])
        # plt.ylim(0, V.shape[1])
        #
        # ax3 = axes.flat[3]
        # im = ax3.scatter(x3, y3, c=t3, marker='o', s=30)
        # plt.xlim(0, V.shape[0])
        # plt.ylim(0, V.shape[1])
        #
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(x, y, z, c=t, marker='o', s=10, alpha=0.2)
        im = ax.scatter(cx, cy, cz, c=range(1,cx.shape[0]+1,1), marker='s', s=30)
        d = 3
        for i in range(len(cx)):
            plot_cube(ax, cx[i][0]-d/2, cy[i][0]-d/2, cz[i][0]-d/2, d, i/len(cx))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.xlim(0, V.shape[0])
        plt.ylim(0, V.shape[1])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()
        plt.hold(True)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # viz2(train_features[5,:,:,:].reshape(32,32,32),cx,cy,cz)
    pp = []
    for i in range(16):
        pp = pp + [y[i][0][train_data[1][5][0]]]
    print (pp)
    print(cx)
    print(cy)
    print(cz)