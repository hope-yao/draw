# # MAX: simple RNN model w/o VAE
#
#
# from __future__ import division, print_function
#
# import sys
#
# sys.path.append("../lib")
#
# import logging
# import theano
# import theano.tensor as T
# from theano import tensor
# import numpy as np
#
# from blocks.bricks.base import application, lazy
# from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
# from blocks.bricks import Random, Initializable, MLP, Linear, Rectifier
# from blocks.bricks.conv import Convolutional, Flattener, ConvolutionalSequence, MaxPooling
# from blocks.bricks import Identity, Tanh, Logistic
# from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum
# from blocks.bricks import Tanh, Identity, Softmax
# from blocks.bricks.cost import BinaryCrossEntropy
# from blocks.bricks.recurrent import SimpleRecurrent, LSTM
# from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, Uniform
# from blocks.filter import VariableFilter
# from blocks.graph import ComputationGraph
# from blocks.roles import PARAMETER
# from blocks.monitoring import aggregation
# from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
# from blocks.extensions.saveload import Checkpoint
# from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
# from blocks.main_loop import MainLoop
# from blocks.model import Model
# from toolz.itertoolz import interleave
#
# from attention import ZoomableAttentionWindow
# from prob_layers import replicate_batch
#
# class EntropyReader(Random):
#     def __init__(self, x_dim, c_dim, channels, height, width, N, **kwargs):
#         self.batch_size = x_dim.shape[0]
#         self.img_height = height
#         self.img_width = width
#         self.N = N
#         self.x_dim = x_dim
#         self.c_dim = c_dim
#         self.output_dim = (height, width)
#
#         self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
#
#     def get_dim(self, name):
#         if name == 'input':
#             return self.c_dim
#         elif name == 'x_dim':
#             return self.x_dim
#         elif name == 'output':
#             return self.output_dim
#         else:
#             raise ValueError
#
#     @application(inputs=['y', 'img_set'], outputs=['img'])
#     def generateImg(self, y, img_set):
#         sample_size = 10
#         u = self.theano_rng.uniform(
#             size=(self.batch_size, sample_size), low=0., high=1.)
#         ysum = [y[:,:i].sum(axis=1) for i in T.arange(y.shape[1])]
#         id = [].flatten()
#         img = T.Constant([img_set[i][np.ceil(self.theano_rng.uniform(size=1,low=0,high=img_set[i].shape[0]-1))] for i in T.arange(sample_size*self.batch_size)])
#         return img.reshape((self.batch_size, sample_size, self.img_height, self.img_width))
#
#     @application(inputs=['y', 'img_set'], outputs=['img'])
#     def calEntropy(self, y, img_set):
#
#     @application(inputs=['x', 'y', 'img_set'], outputs=['r', 'cx', 'cy'])
#     def apply(self, x, y, img_set):
#         scan_x = self.img_width - self.N + 1
#         scan_y = self.img_height - self.N + 1
#         scan_set_x = T.arange(self.N/2+1,self.img_width-self.N/2, 'int64')
#         scan_set_y = T.arange(self.N / 2 + 1, self.img_width - self.N / 2, 'int64')
#         scan_set = [list(zip(scan_set_x, p)) for p in scan_set_y]
#         img = self.generateImg(y, img_set)
#         entropy = self.calEntropy(img) # entropy dim: (batch_size, scan_x, scan_y)
#         e = entropy.reshape((self.batch_size,scan_x*scan_y))
#         id = T.argmax(e,axis=1)
#         center_x, center_y = scan_set[id]
#         r = self.zoomer.read_large(x, center_y, center_x)
#
#         return r, center_x, center_y
#
# class MinEntropyModel(BaseRecurrent, Initializable, Random):
#     def __init__(self, image_size, channels, attention, **kwargs):
#         super(MinEntropyModel, self).__init__(**kwargs)
#
#         self.n_iter = 3
#         inits = {
#             # 'weights_init': Orthogonal(),
#             'weights_init': IsotropicGaussian(0.01),
#             'biases_init': Constant(0.),
#         }
#         conv_inits = {
#             'weights_init': Uniform(width=.2),
#             'biases_init': Constant(0.)
#         }
#         img_height, img_width = image_size
#         self.x_dim = channels * img_height * img_width
#
# #-----------------------------------------------------------------------------------------------------------------------
#         # USE LeNet
#
#         feature_maps = [20, 50] #[20, 50]
#         mlp_hiddens = [500] # 500
#         conv_sizes = [5, 5] # [5, 5]
#         pool_sizes = [2, 2]
#         image_size = (28, 28)
#         output_size = 10
#
#         conv_activations = [Rectifier() for _ in feature_maps]
#         mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
#
#         num_channels = 1
#         image_shape = (28, 28)
#         filter_sizes = zip(conv_sizes, conv_sizes)
#         feature_maps = feature_maps
#         pooling_sizes = zip(pool_sizes, pool_sizes)
#         top_mlp_activations = mlp_activations
#         top_mlp_dims = mlp_hiddens + [output_size]
#         border_mode = 'full'
#
#         conv_step = None
#         if conv_step is None:
#             self.conv_step = (1, 1)
#         else:
#             self.conv_step = conv_step
#         self.num_channels = num_channels
#         self.image_shape = image_shape
#         self.top_mlp_activations = top_mlp_activations
#         self.top_mlp_dims = top_mlp_dims
#         self.border_mode = border_mode
#
#         conv_parameters = zip(filter_sizes, feature_maps)
#
#         # Construct convolutional layers with corresponding parameters
#         self.layers = list(interleave([
#             (Convolutional(filter_size=filter_size,
#                            num_filters=num_filter,
#                            step=self.conv_step,
#                            border_mode=self.border_mode,
#                            name='conv_{}'.format(i))
#              for i, (filter_size, num_filter)
#              in enumerate(conv_parameters)),
#             conv_activations,
#             (MaxPooling(size, name='pool_{}'.format(i))
#              for i, size in enumerate(pooling_sizes))]))
#
#         self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
#                                                    image_size=image_shape, **conv_inits)
#
#         # Construct a top MLP
#         self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, **inits)
#         self.flattener = Flattener()
#
# # -----------------------------------------------------------------------------------------------------------------------
#
#         self.children = [self.conv_sequence, self.flattener, self.top_mlp]
#
#     @property
#     def output_dim(self):
#         return self.top_mlp_dims[-1]
#
#     @output_dim.setter
#     def output_dim(self, value):
#         self.top_mlp_dims[-1] = value
#
#     def _push_allocation_config(self):
#         self.conv_sequence._push_allocation_config()
#         conv_out_dim = self.conv_sequence.get_dim('output')
#
#         self.top_mlp.activations = self.top_mlp_activations
#         self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims
#
#
#     def get_dim(self, name):
#         if name == 'y':
#             return 10 # for mnist_lenet
#         elif name == 'c':
#             return self.conv_out_dim_flatten
#         elif name == 'r':
#             return self.x_dim
#         elif name == 'center_y':
#             return 1
#         elif name == 'center_x':
#             return 1
#         else:
#             super(MinEntropyModel, self).get_dim(name)
#
#     # ------------------------------------------------------------------------
#
#     @recurrent(sequences=['u'], contexts=['x'],
#                states=['y', 'r'],
#                outputs=['y', 'r', 'c', 'cx', 'cy'])
#     def apply(self, x, u, y, r):
#         rr, center_y, center_x = self.reader.apply(x, y, u)
#         r = T.minimum(r + rr,1.) # combine revealed images
#         batch_size = r.shape[0]
#         c_raw = self.conv_sequence.apply(r.reshape((batch_size,1,28,28)))
#         c = self.flattener.apply(c_raw)
#         y = self.top_mlp.apply(c)
#
#         return y, r, c, center_x, center_y
#
#     # ------------------------------------------------------------------------
#
#     @application(inputs=['features'], outputs=['targets', 'r', 'c', 'cx', 'cy'])
#     def classify(self, features):
#         batch_size = features.shape[0]
#         # Sample from mean-zeros std.-one Gaussian
#         u = self.theano_rng.normal(
#             size=(self.n_iter, batch_size, 1),
#             avg=0., std=1.)
#
#         y, r, c, cx, cy = self.apply(x=features, u=u)
#
#         return y, r, c, cx, cy
