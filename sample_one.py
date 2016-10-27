#!/usr/bin/env python

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.config import config
from blocks.serialization import load

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

# these aren't paramed yet in a generic way, but these values work
COLS = 64

def img_grid(arr, cx, cy, global_scale=True):
    N, channels, height, width = arr.shape

    global COLS
    cols = COLS
    # rows = int(np.sqrt(N))
    # cols = int(np.sqrt(N))

    # if rows*cols < N:
    #     cols = cols + 1

    # if rows*cols < N:
    #     rows = rows + 1

    total_height = height
    total_width  = cols * width + 63

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
        I[0:3, np.max((offset_y + cx[i],0)), np.max((offset_x + cy[i],0))] = np.array([255,0,0]).astype(np.uint8)

    # if(channels == 1):
    #     out = I.reshape( (total_height, total_width) )
    # else:
    out = np.dstack(I).astype(np.uint8)
    return Image.fromarray(out)

def generate_samples(p, subdir, output_size, channels):
    if isinstance(p, Model):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    draw = model.get_top_bricks()[0]
    # reset the random generator
    del draw._theano_rng
    del draw._theano_seed
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling sample function...")

    n_samples = T.iscalar("n_samples")
    samples = draw.sample(n_samples)

    do_sample = theano.function([n_samples], outputs=samples, allow_input_downcast=True)

    #------------------------------------------------------------
    logging.info("Sampling and saving images...")

    global COLS
    samples, cx, cy = do_sample(1)
    #samples = np.random.normal(size=(16, 100, 28*28))

    n_iter, N, D = samples.shape
    # logging.info("SHAPE IS: {}".format(samples.shape))

    samples = samples.reshape( (n_iter, N, channels, output_size, output_size) )

    # if(n_iter > 0):
    #     img = img_grid(samples[n_iter-1,:,:,:], cx[n_iter-1,:], cy[n_iter-1,:])
    #     img.save("{0}/sample.png".format(subdir))

    # for i in xrange(n_iter):
    img = img_grid(samples[:,0,:,:,:], cx[:,0], cy[:,0])
    img.save("{0}/debug.png".format(subdir))

    #with open("centers.pkl", "wb") as f:
    #    pikle.dump(f, (center_y, center_x, delta))
    # os.system("convert -delay 5 {0}/time-*.png -delay 300 {0}/sample.png {0}/sequence.gif".format(subdir))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    parser.add_argument("--channels", type=int,
                default=1, help="number of channels")
    parser.add_argument("--size", type=int,
                default=28, help="Output image size (width and height)")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = load(f, 'model')

    subdir = "c:\users\p2admin\documents\max\projects\draw\mnist-20161017-233956\sample_one"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    generate_samples(p, subdir, args.size, args.channels)



