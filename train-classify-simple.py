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

try:
    from blocks.extras import Plot
except ImportError:
    pass

import draw.datasets as datasets
from draw.draw_classify_simple import *
from draw.samplecheckpoint import SampleCheckpoint

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def main(name, dataset, epochs, batch_size, learning_rate, attention,
         n_iter, rnn_dim, y_dim, oldmodel, live_plotting):
    image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)

    train_stream = Flatten(
        DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size)))
    valid_stream = Flatten(
        DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
    test_stream = Flatten(
        DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, batch_size)))

    if name is None:
        name = dataset

    attention_tag = 'full'

    # Learning rate
    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e" % value)[0]
        return "%s%d" % (leading, -exp)

    lr_str = lr_tag(learning_rate)

    subdir = name + "-simple-" + time.strftime("%Y%m%d-%H%M%S")
    longname = "%s-%s-t%d-rnn%d-y%d-lr%s" % (dataset, attention_tag, n_iter, rnn_dim, y_dim, lr_str)
    pickle_file = subdir + "/" + longname + ".pkl"

    print("\nRunning experiment %s" % longname)
    print("               dataset: %s" % dataset)
    print("          subdirectory: %s" % subdir)
    print("         learning rate: %g" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("         rnn dimension: %d" % rnn_dim)
    print("           y dimension: %d" % y_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    # ----------------------------------------------------------------------

    draw = DrawClassifyModel(image_size=image_size, channels=channels, attention=attention)
    draw.push_initialization_config()
    draw.conv_sequence.layers[0].weights_init = Uniform(width=.2)
    draw.conv_sequence.layers[1].weights_init = Uniform(width=.09)
    draw.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    draw.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    draw.initialize()

    # ------------------------------------------------------------------------
    x = tensor.matrix('features') # keyword from fuel
    y = tensor.matrix('targets') # keyword from fuel

    y_hat = draw.classify(x)

    y_hat_last = y_hat[-1,:,:] # output should be batch_size * class
    # y_hat_last = y_hat
    # # classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
    y_int = T.cast(y, 'int64')
    # recognition = -T.mean(T.log(y_hat_last)[T.arange(batch_size), y_int]) # guess (rnn_iter (16), class (10), batch_size)
    # recognition.name = "recognition"
    #
    recognition_convergence = (-y_hat*T.log2(y_hat)).sum(axis=1).mean()
    recognition_convergence.name = "recognition_convergence"


    # from LeNet
    recognition = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last)
            .copy(name='recognition'))
    error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last)
                  .copy(name='error_rate'))

    cost = recognition
    # cost = recognition + recognition_convergence.mean()
    cost.name = "cost"

    # _, activated_id = T.max_and_argmax(y_hat_last, axis=1)
    # error = theano.tensor.neq(activated_id.flatten(), y_int.flatten()).sum()/float(batch_size)
    error.name = "error"

    # ------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost,
        parameters=params,
        # step_rule=CompositeRule([
        #     StepClipping(10.),
        #     Adam(learning_rate),
        # ])
        # step_rule=RMSProp(learning_rate),
        # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
        step_rule=Scale(learning_rate=learning_rate)
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
    if live_plotting:
        plotting_extensions = [
            Plot(name, channels=plot_channels)
        ]

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
                       #            DataStreamMonitoring(
                       #                monitors,
                       #                valid_stream,
                       ##                updates=scan_updates,
                       #                prefix="valid"),
                       DataStreamMonitoring(
                           monitors,
                           test_stream,
                           #                updates=scan_updates,
                           prefix="test"),
                       # Checkpoint(name, before_training=False, after_epoch=True, save_separately=['log', 'model']),
                       Checkpoint("{}/{}".format(subdir, name), save_main_loop=False, before_training=True,
                                  after_epoch=True, save_separately=['log', 'model']),
                       # SampleCheckpoint(image_size=image_size[0], channels=channels, save_subdir=subdir,
                       #                  before_training=True, after_epoch=True),
                       ProgressBar(),
                       Printing()] + plotting_extensions)

    if oldmodel is not None:
        print("Initializing parameters with old model %s" % oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_parameter_values(oldmodel.get_param_values())
        del oldmodel

    main_loop.run()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--live-plotting", "--plot", action="store_true",
                        default=False, help="Activate live-plotting to a bokeh-server")
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str, default="",
                        help="Use attention mechanism (read_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=16, help="No. of iterations")
    parser.add_argument("--rnn-dim", type=int, dest="rnn_dim",
                        default=256, help="Encoder RNN state dimension") # originally 256
    parser.add_argument("--y-dim", type=int, dest="y_dim",
                        default=10, help="Decoder  RNN state dimension") # dim should be the number of classes
    parser.add_argument("--oldmodel", type=str,
                        help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))
