from __future__ import print_function

import os
import math
import json
import logging
from PIL import Image
from datetime import datetime
import numpy as np
def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=4,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8) *127
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=4,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
    return im

import tensorflow as tf
def list2tensor(alist,dim=0):
    for i, list_i in enumerate(alist):
        if i == 0:
            atensor = list_i
        else:
            atensor = tf.concat([atensor, list_i], dim)
    return atensor


import dateutil.tz

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir



# xx = take_a_glimpse(x_test, np.zeros((128,2),'float32'), 17, delta = 1.0, sigma = 0.1).eval(session=sess)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(xx[100],cmap='gray',interpolation='nearest')
# plt.colorbar()
# plt.show()


# np.mean(np.abs(sess.run(y_hat,feed_dict_test)-np.tile(np.expand_dims(feed_dict_test[y],2),(1,1,5))),(0,1))

# sess.run(y_hat,feed_dict_train_fix)[4],feed_dict_train_fix[y][4]

# np.mean(np.abs(sess.run(grads[2],feed_dict_train))[0])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def glimpse_viz(xx):
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
    for i in range(T):
        x = loc[i,0,0]
        y = loc[i,0,1]
        ax_mnist.text(x , y, i, fontsize=15, color='red')
        import matplotlib.patches as patches
        p = patches.Rectangle(
            (x-read_n/2. , y-read_n/2.), read_n, read_n,
            fill=False, clip_on=False, color='red'
            )
        ax_mnist.add_patch(p)

    t = prob[:,0,:]
    ax_acc.imshow(t.transpose(), interpolation='nearest', cmap=plt.cm.viridis,extent=[0,ram.n_iter,ram.n_class,0])
    # ax_acc.xlabel('time iteration')
    # ax_acc.ylabel('class index')
    # ax_acc.colorbar()

    import numpy
    glimpse_idx = 0
    glimpse0 = numpy.zeros((28,28))
    canvas0 = numpy.zeros((28,28))
    x_start = l[glimpse_idx,0,1]-ram.read_N/2.
    x_end = l[glimpse_idx,0,1]+ram.read_N/2.
    y_start = l[glimpse_idx,0,0]-ram.read_N/2.
    y_end = l[glimpse_idx,0,0]+ram.read_N/2.
    glimpse_idx = glimpse_idx + 1
    glimpse0[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
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
    x_start = l[glimpse_idx,0,1]-ram.read_N/2.
    x_end = l[glimpse_idx,0,1]+ram.read_N/2.
    y_start = l[glimpse_idx,0,0]-ram.read_N/2.
    y_end = l[glimpse_idx,0,0]+ram.read_N/2.
    glimpse_idx = glimpse_idx + 1
    glimpse1[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
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
    x_start = l[glimpse_idx,0,1]-ram.read_N/2.
    x_end = l[glimpse_idx,0,1]+ram.read_N/2.
    y_start = l[glimpse_idx,0,0]-ram.read_N/2.
    y_end = l[glimpse_idx,0,0]+ram.read_N/2.
    glimpse_idx = glimpse_idx + 1
    glimpse2[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
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
    x_start = l[glimpse_idx,0,1]-ram.read_N/2.
    x_end = l[glimpse_idx,0,1]+ram.read_N/2.
    y_start = l[glimpse_idx,0,0]-ram.read_N/2.
    y_end = l[glimpse_idx,0,0]+ram.read_N/2.
    glimpse_idx = glimpse_idx + 1
    glimpse3[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
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
    x_start = l[glimpse_idx,0,1]-ram.read_N/2.
    x_end = l[glimpse_idx,0,1]+ram.read_N/2.
    y_start = l[glimpse_idx,0,0]-ram.read_N/2.
    y_end = l[glimpse_idx,0,0]+ram.read_N/2.
    # glimpse_idx = glimpse_idx + 1
    glimpse4[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
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

    print(rho_orig)